"""YOLO model training service with Ultralytics built-in MLflow integration.

Responsibilities
----------------
1. Auto-detect dataset source mode from dataset_manifest.json when present.
2. Validate that --pretrained-weights and --resume-from are not both set.
3. In local mode, validate that data.yaml and image directories exist.
4. Download weights / resume checkpoint from S3 to a temp dir when needed.
5. Write data.yaml to a temp dir (always deleted in a finally block).
6. Set MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME env vars so that
   the Ultralytics built-in MLflow callback handles all logging.
7. Register Ultralytics callbacks for per-epoch metric + system resource
   logging, console output, and periodic S3 checkpoint uploads.
8. Call model.train() with the full hyperparameter set.
9. Upload best.pt and last.pt to the S3 checkpoint path.
10. Evaluate best.pt and last.pt on the test split (reporting only) and
    log metrics + plots to the current MLflow run.  Skipped with a warning
    when the test split is not available.
11. Clean up temp dirs.
"""

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml

from app.models.training import TrainingParams, TrainingResult
from app.services.resource_monitor import ResourceMonitor

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_S3_URI_RE = re.compile(r"^s3://([^/]+)/(.+)$")

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Ultralytics metric keys
_METRIC_PRECISION = "metrics/precision(B)"
_METRIC_RECALL = "metrics/recall(B)"
_METRIC_MAP50 = "metrics/mAP50(B)"
_METRIC_MAP50_95 = "metrics/mAP50-95(B)"

# File names written by dataset_loading
_MANIFEST_FILENAME = "dataset_manifest.json"


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class TrainingError(Exception):
    """Raised when a training run fails unrecoverably."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class TrainingService:
    """Orchestrates a YOLO training run with Ultralytics built-in MLflow tracking.

    MLflow logging (params, metrics, artifacts) is handled entirely by the
    Ultralytics MLflow callback.  This service sets the required environment
    variables and focuses on S3 I/O and console output.

    Parameters
    ----------
    s3_client:
        A pre-constructed boto3 S3 client used for checkpoint download/upload
        and (when source='s3') image streaming.
    mlflow_tracking_uri:
        URI of the remote MLflow tracking server.
    """

    def __init__(
        self,
        s3_client: Any,
        mlflow_tracking_uri: str,
    ) -> None:
        self._s3 = s3_client
        self._mlflow_tracking_uri = mlflow_tracking_uri
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, params: TrainingParams) -> TrainingResult:
        """Execute the full training pipeline end-to-end.

        Parameters
        ----------
        params:
            Validated training parameters from the CLI.

        Returns
        -------
        TrainingResult
            Summary of the completed run.

        Raises
        ------
        TrainingError
            On any unrecoverable failure.
        """
        # Auto-detect S3 streaming mode from manifest before validation
        params = self._apply_manifest_if_present(params)

        self._validate_params(params)

        # Defer heavy imports so that unit tests can mock them easily
        from ultralytics import YOLO  # noqa: PLC0415

        Path(params.output_dir).mkdir(parents=True, exist_ok=True)

        # Configure Ultralytics' built-in MLflow callback via env vars
        os.environ["MLFLOW_TRACKING_URI"] = self._mlflow_tracking_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = params.experiment_name

        # Ultralytics reads MLFLOW_RUN (not MLFLOW_RUN_NAME) for the run name.
        # Bridge the Kubeline workflow name so each run gets a unique name.
        workflow_name = os.environ.get("KUBECORE_WORKFLOW_NAME", "")
        if workflow_name:
            os.environ["MLFLOW_RUN"] = workflow_name

        try:
            result = self._run_training(params, YOLO)
        except Exception as exc:
            raise TrainingError(f"Training failed: {exc}") from exc

        return result

    # ------------------------------------------------------------------
    # Manifest auto-detection
    # ------------------------------------------------------------------

    def _apply_manifest_if_present(self, params: TrainingParams) -> TrainingParams:
        """Read dataset_manifest.json from dataset_dir if it exists.

        When present the manifest overrides the source, s3_bucket, and
        s3_prefix fields on the params object so that the caller-supplied
        --source / --s3-bucket / --s3-prefix flags are not required.

        Returns a new (or the same) TrainingParams instance.
        """
        manifest_path = Path(params.dataset_dir) / _MANIFEST_FILENAME
        if not manifest_path.exists():
            self._logger.info(
                "No %s found in dataset_dir=%s — using source=%s (local mode)",
                _MANIFEST_FILENAME,
                params.dataset_dir,
                params.source,
            )
            return params

        try:
            with manifest_path.open() as fh:
                raw = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to parse %s: %s — falling back to source=%s",
                manifest_path,
                exc,
                params.source,
            )
            return params

        bucket = raw.get("bucket")
        prefix = raw.get("prefix")

        if not bucket or not prefix:
            self._logger.warning(
                "%s is missing 'bucket' or 'prefix' fields — ignoring manifest",
                manifest_path,
            )
            return params

        total = raw.get("total_images", "?")
        label_keys = raw.get("label_keys")

        mode_label = "manifest-only" if label_keys else "labels-only"
        self._logger.info(
            "Detected %s (bucket=%s prefix=%s total_images=%s mode=%s) — "
            "switching to S3 streaming mode",
            _MANIFEST_FILENAME,
            bucket,
            prefix,
            total,
            mode_label,
        )

        # Build a new params instance with overridden S3 fields
        update: dict[str, Any] = {
            "source": "s3",
            "s3_bucket": bucket,
            "s3_prefix": prefix,
        }
        # Signal manifest-only mode: labels must also be streamed from S3
        if label_keys:
            update["s3_stream_labels"] = True
        updated = params.model_copy(update=update)
        return updated

    # ------------------------------------------------------------------
    # Core training pipeline
    # ------------------------------------------------------------------

    def _run_training(
        self,
        params: TrainingParams,
        yolo_cls: Any,
    ) -> TrainingResult:
        """Inner pipeline: weights, data.yaml, callbacks, train(), S3 upload."""

        monitor = ResourceMonitor(gpu_index=self._parse_gpu_index(params.device))

        # Validate local dataset structure before starting the run
        if params.source == "local":
            self._validate_local_dataset(params)

        with tempfile.TemporaryDirectory(prefix="io-model-training-") as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Resolve model path (download from S3 if needed)
            model_path = self._resolve_model_path(params, tmp_path)

            # 2. Write data.yaml into the temp dir
            data_yaml_path = self._write_data_yaml(params, tmp_path)

            # 3. Build the Ultralytics YOLO model object
            model = yolo_cls(str(model_path))

            # 4. Register callbacks
            epoch_metrics: dict[str, float] = {}

            model.add_callback(
                "on_train_start",
                self._make_provenance_callback(params),
            )
            model.add_callback(
                "on_train_batch_end",
                self._make_batch_end_callback(epoch_metrics),
            )
            model.add_callback(
                "on_fit_epoch_end",
                self._make_epoch_end_callback(epoch_metrics, monitor),
            )
            model.add_callback(
                "on_train_epoch_end",
                self._make_checkpoint_callback(params),
            )
            model.add_callback(
                "on_train_end",
                self._make_train_end_callback(epoch_metrics),
            )

            # 5. Build train() kwargs — resume mode sets resume=True
            train_kwargs = self._build_train_kwargs(
                params=params,
                data_yaml_path=str(data_yaml_path),
            )

            # 5b. S3 streaming mode — inject S3PoseTrainer
            if params.source == "s3":
                from app.services.s3_pose_trainer import make_s3_pose_trainer

                cache_dir = str(tmp_path / "s3_cache")
                labels_root = str(Path(params.dataset_dir).resolve() / "labels")

                # In manifest-only mode, labels are also streamed from S3
                s3_labels_prefix: str | None = None
                if params.s3_stream_labels:
                    s3_labels_prefix = params.s3_prefix  # type: ignore[assignment]
                    self._logger.info(
                        "Manifest-only mode: labels will be streamed from S3"
                    )

                s3_trainer_cls = make_s3_pose_trainer(
                    s3_client=self._s3,
                    s3_bucket=params.s3_bucket,  # type: ignore[arg-type]
                    s3_prefix=params.s3_prefix,  # type: ignore[arg-type]
                    local_labels_root=labels_root,
                    s3_labels_prefix=s3_labels_prefix,
                    cache_dir=cache_dir,
                    cache_max_bytes=params.disk_cache_bytes,
                )
                train_kwargs["trainer"] = s3_trainer_cls
                self._logger.info(
                    "S3 streaming enabled | bucket=%s prefix=%s cache=%s (%d MB)",
                    params.s3_bucket,
                    params.s3_prefix,
                    cache_dir,
                    params.disk_cache_bytes // (1024 * 1024),
                )

            trainer = model.train(**train_kwargs)

            # 6. Determine save directory
            save_dir = Path(model.trainer.save_dir)  # type: ignore[union-attr]

            # 7. Upload final weights to S3
            best_s3_uri = self._upload_final_weights(params, save_dir)

            # 8. Export quantized / optimized models
            exported_s3_uris: dict[str, str] = {}
            if params.export.enabled:
                exported_s3_uris = self._export_models(
                    model=model,
                    params=params,
                    save_dir=save_dir,
                    data_yaml_path=str(data_yaml_path),
                )

            # 9. Post-training evaluation on the test split (reporting only)
            self._evaluate_on_test_set(
                yolo_cls=yolo_cls,
                params=params,
                save_dir=save_dir,
                data_yaml_path=str(data_yaml_path),
                tmp_path=tmp_path,
            )

        # Build result — get MLflow run_id from Ultralytics' run
        mlflow_run_id = self._get_mlflow_run_id()

        # Tag the completed MLflow run with Kubeline platform metadata
        self._tag_kubecore_metadata(mlflow_run_id)

        final_map50 = float(epoch_metrics.get("val/mAP50", 0.0))
        final_map50_95 = float(epoch_metrics.get("val/mAP50_95", 0.0))

        raw_epoch = getattr(trainer, "epoch", None)
        epochs_completed: int = raw_epoch + 1 if raw_epoch is not None else params.epochs

        return TrainingResult(
            experiment_name=params.experiment_name,
            model_variant=params.model_variant,
            mlflow_run_id=mlflow_run_id,
            best_checkpoint_local=str(save_dir / "weights" / "best.pt"),
            best_checkpoint_s3=best_s3_uri,
            epochs_completed=epochs_completed,
            final_map50=final_map50,
            final_map50_95=final_map50_95,
            exported_models=exported_s3_uris,
        )

    # ------------------------------------------------------------------
    # Callback factories
    # ------------------------------------------------------------------

    def _make_provenance_callback(
        self,
        params: TrainingParams,
    ) -> Callable[[Any], None]:
        """Return an on_train_start callback that logs dataset provenance to MLflow.

        Fires after the Ultralytics MLflow callback opens the run
        (on_pretrain_routine_end), so mlflow.active_run() is non-None.
        Implements FR-04 and CON-02: only non-None fields are logged.
        Q3 resolution: pipeline.config_hash is logged as both param and tag.
        T-05 mitigation: guarded by mlflow.active_run() check.
        """

        def on_train_start(trainer: Any) -> None:  # noqa: ARG001
            try:
                import mlflow  # noqa: PLC0415

                if mlflow.active_run() is None:
                    _logger.warning(
                        "No active MLflow run at on_train_start; "
                        "skipping provenance param logging"
                    )
                    return

                param_map = {
                    "dataset.version": params.dataset_version,
                    "dataset.lakefs_branch": params.lakefs_branch,
                    "dataset.lakefs_commit": params.lakefs_commit,
                    "pipeline.config_hash": params.config_hash,
                }
                for key, value in param_map.items():
                    if value is not None:
                        mlflow.log_param(key, value)

                # Q3: config_hash also logged as tag for experiment list filtering
                if params.config_hash is not None:
                    mlflow.set_tag("pipeline.config_hash", params.config_hash)

                _logger.info(
                    "Logged dataset provenance to MLflow | "
                    "lakefs_commit=%s config_hash=%s",
                    params.lakefs_commit or "null",
                    params.config_hash or "null",
                )
            except Exception as exc:  # noqa: BLE001
                _logger.warning("Failed to log provenance params to MLflow: %s", exc)

        return on_train_start

    @staticmethod
    def _make_batch_end_callback(
        epoch_metrics: dict[str, float],
    ) -> Callable[[Any], None]:
        """Return a callback that captures per-batch training losses."""

        def on_train_batch_end(trainer: Any) -> None:
            try:
                tloss = getattr(trainer, "tloss", None)
                if tloss is None:
                    return

                loss_names = ["box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"]
                try:
                    for idx, loss_name in enumerate(loss_names):
                        if idx < len(tloss):
                            epoch_metrics[f"train/{loss_name}"] = float(
                                tloss[idx].item()
                                if hasattr(tloss[idx], "item")
                                else tloss[idx]
                            )
                except (TypeError, IndexError):
                    return

            except Exception as cb_exc:  # noqa: BLE001
                _logger.warning("on_train_batch_end callback error: %s", cb_exc)

        return on_train_batch_end

    @staticmethod
    def _make_epoch_end_callback(
        epoch_metrics: dict[str, float],
        monitor: ResourceMonitor,
    ) -> Callable[[Any], None]:
        """Return a callback that logs per-epoch validation metrics and system resources."""

        def on_fit_epoch_end(trainer: Any) -> None:
            try:
                val_metrics_raw = getattr(trainer, "metrics", {}) or {}
                val_metrics: dict[str, float] = {
                    "val/precision": float(val_metrics_raw.get(_METRIC_PRECISION, 0.0)),
                    "val/recall": float(val_metrics_raw.get(_METRIC_RECALL, 0.0)),
                    "val/mAP50": float(val_metrics_raw.get(_METRIC_MAP50, 0.0)),
                    "val/mAP50_95": float(val_metrics_raw.get(_METRIC_MAP50_95, 0.0)),
                }
                epoch_metrics.update(val_metrics)

                current_epoch: int = getattr(trainer, "epoch", 0) + 1

                # Collect system resource snapshot and log it together with the
                # model metrics so every metric shares the same epoch step.
                system_metrics = monitor.collect()
                all_metrics = {**val_metrics, **system_metrics}
                try:
                    import mlflow  # noqa: PLC0415

                    if mlflow.active_run() is not None:
                        mlflow.log_metrics(all_metrics, step=current_epoch)
                    else:
                        _logger.debug(
                            "No active MLflow run — skipping epoch %d metric logging",
                            current_epoch,
                        )
                except Exception as log_exc:  # noqa: BLE001
                    _logger.warning(
                        "Failed to log epoch metrics to MLflow: %s", log_exc
                    )

            except Exception as cb_exc:  # noqa: BLE001
                _logger.warning("on_fit_epoch_end callback error: %s", cb_exc)

        return on_fit_epoch_end

    def _make_checkpoint_callback(
        self,
        params: TrainingParams,
    ) -> Callable[[Any], None]:
        """Return a callback that uploads a checkpoint to S3 and logs it to MLflow every N epochs."""

        def on_train_epoch_end(trainer: Any) -> None:
            try:
                epoch: int = trainer.epoch + 1  # Ultralytics epoch is 0-indexed
                if epoch % params.checkpoint_interval != 0:
                    return
                last_pt = Path(trainer.last)
                if not last_pt.exists():
                    return

                # Upload to S3
                s3_key = (
                    f"{params.checkpoint_prefix}/{params.experiment_name}/"
                    f"epoch_{epoch:04d}.pt"
                )
                self._upload_to_s3(
                    local_path=last_pt,
                    bucket=params.checkpoint_bucket,
                    key=s3_key,
                )
                _logger.info(
                    "Uploaded checkpoint to s3://%s/%s",
                    params.checkpoint_bucket,
                    s3_key,
                )

                # Log checkpoint as MLflow artifact
                try:
                    import mlflow  # noqa: PLC0415

                    if mlflow.active_run() is not None:
                        mlflow.log_artifact(
                            str(last_pt),
                            artifact_path=f"checkpoints/epoch_{epoch:04d}",
                        )
                        _logger.info(
                            "Logged epoch %d checkpoint to MLflow artifacts", epoch
                        )
                except Exception as mlflow_exc:  # noqa: BLE001
                    _logger.warning(
                        "Failed to log checkpoint to MLflow: %s", mlflow_exc
                    )

            except Exception as cb_exc:  # noqa: BLE001
                _logger.warning("S3 checkpoint upload failed: %s", cb_exc)

        return on_train_epoch_end

    @staticmethod
    def _make_train_end_callback(
        epoch_metrics: dict[str, float],
    ) -> Callable[[Any], None]:
        """Return a callback that logs a training completion summary."""

        def on_train_end(trainer: Any) -> None:
            _logger.info(
                "Training complete  best mAP50-95=%.4f  saved to %s",
                float(epoch_metrics.get("val/mAP50_95", 0.0)),
                getattr(trainer, "best", "?"),
            )

        return on_train_end

    # ------------------------------------------------------------------
    # MLflow run ID retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _get_mlflow_run_id() -> str:
        """Return the MLflow run ID from the most recent Ultralytics-managed run."""
        try:
            import mlflow  # noqa: PLC0415

            last_run = mlflow.last_active_run()
            if last_run:
                return last_run.info.run_id
            _logger.warning(
                "mlflow.last_active_run() returned None — "
                "the downstream model_registration step will not be able to "
                "link this training run. Check that the Ultralytics MLflow "
                "callback is enabled."
            )
        except Exception:  # noqa: BLE001
            _logger.warning(
                "Failed to retrieve MLflow run ID — "
                "model_registration will not be able to link this training run."
            )
        return ""

    @staticmethod
    def _tag_kubecore_metadata(run_id: str) -> None:
        """Tag the MLflow run with KUBECORE_* environment variables.

        Uses MlflowClient (not the fluent API) so it works on an already-ended
        run without triggering a new ghost run via mlflow.autolog().
        """
        if not run_id:
            return
        try:
            from mlflow.tracking import MlflowClient  # noqa: PLC0415

            kubecore_tags = {
                k.lower().replace("kubecore_", "kubecore."): v
                for k, v in os.environ.items()
                if k.startswith("KUBECORE_") and v
            }
            if kubecore_tags:
                client = MlflowClient()
                for key, value in kubecore_tags.items():
                    client.set_tag(run_id, key, value)
                _logger.info("Tagged MLflow run with %d kubecore tags", len(kubecore_tags))
        except Exception:  # noqa: BLE001
            _logger.warning("Failed to tag MLflow run with kubecore metadata")

    # ------------------------------------------------------------------
    # Post-training export / quantization
    # ------------------------------------------------------------------

    def _export_models(
        self,
        model: Any,
        params: "TrainingParams",
        save_dir: Path,
        data_yaml_path: str,
    ) -> dict[str, str]:
        """Export trained model to requested formats/precisions.

        Uploads each exported file to S3 and logs it as an MLflow artifact.
        Errors are non-fatal — training has already succeeded.

        Returns
        -------
        dict[str, str]
            Map of ``{format}_{precision}`` labels to S3 URIs.
        """
        exported: dict[str, str] = {}
        base_key = f"{params.checkpoint_prefix}/{params.experiment_name}"
        mlflow_run_id = self._get_mlflow_run_id()

        for fmt in params.export.formats:
            for precision in params.export.precisions:
                label = f"{fmt}_{precision}"

                # ONNX does not support INT8 natively via Ultralytics export
                if fmt == "onnx" and precision == "int8":
                    self._logger.warning(
                        "Skipping %s — ONNX INT8 requires a separate "
                        "onnxruntime quantization pass, not supported by "
                        "Ultralytics export. Use TensorRT for INT8.",
                        label,
                    )
                    continue

                self._logger.info(
                    "Exporting model: format=%s precision=%s", fmt, precision
                )

                try:
                    export_kwargs: dict[str, Any] = {
                        "format": fmt,
                        "imgsz": params.image_size,
                    }
                    if precision == "fp16":
                        export_kwargs["half"] = True
                    elif precision == "int8":
                        export_kwargs["int8"] = True
                        export_kwargs["data"] = data_yaml_path

                    exported_path = Path(model.export(**export_kwargs))

                    if not exported_path.exists():
                        self._logger.warning(
                            "Export produced no file for %s", label
                        )
                        continue

                    # Upload to S3
                    s3_key = f"{base_key}/{exported_path.name}"
                    self._upload_to_s3(
                        exported_path, params.checkpoint_bucket, s3_key
                    )
                    s3_uri = f"s3://{params.checkpoint_bucket}/{s3_key}"
                    exported[label] = s3_uri
                    self._logger.info("Uploaded %s to %s", label, s3_uri)

                    # Log as MLflow artifact (run is closed, use MlflowClient)
                    self._log_mlflow_artifact(
                        exported_path,
                        run_id=mlflow_run_id,
                        artifact_path="exports",
                    )

                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        "Export failed for %s: %s", label, exc
                    )

        return exported

    @staticmethod
    def _log_mlflow_artifact(
        local_path: Path, run_id: str, artifact_path: str
    ) -> None:
        """Log a file as an MLflow artifact via MlflowClient (works on ended runs)."""
        if not run_id:
            return
        try:
            from mlflow.tracking import MlflowClient  # noqa: PLC0415

            MlflowClient().log_artifact(
                run_id, str(local_path), artifact_path=artifact_path
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Failed to log artifact to MLflow: %s", exc)

    # ------------------------------------------------------------------
    # Post-training test-set evaluation
    # ------------------------------------------------------------------

    def _evaluate_on_test_set(
        self,
        yolo_cls: Any,
        params: TrainingParams,
        save_dir: Path,
        data_yaml_path: str,
        tmp_path: Path,
    ) -> None:
        """Evaluate best.pt and last.pt on the test split and log to MLflow.

        Reporting only — never fails the run.  Skipped with a warning if
        the test split is not available for the current dataset source
        mode (local: ``images/test/`` absent or empty; S3 streaming: no
        ``test`` key in ``dataset_manifest.json``).

        In S3 streaming modes the evaluation uses a configured
        :class:`S3PoseValidator` so images (and optionally labels) are
        streamed from S3 exactly as they are during training.
        """
        if not self._is_test_split_available(params):
            self._logger.warning(
                "Test split not available in source=%s mode; "
                "skipping post-training evaluation.",
                params.source,
            )
            return

        mlflow_run_id = self._get_mlflow_run_id()

        s3_validator_cls: Any = None
        if params.source == "s3":
            from app.services.s3_pose_validator import (  # noqa: PLC0415
                make_s3_pose_validator,
            )

            cache_dir = str(tmp_path / "s3_val_cache")
            labels_root = str(Path(params.dataset_dir).resolve() / "labels")
            s3_labels_prefix = params.s3_prefix if params.s3_stream_labels else None
            s3_validator_cls = make_s3_pose_validator(
                s3_client=self._s3,
                s3_bucket=params.s3_bucket,  # type: ignore[arg-type]
                s3_prefix=params.s3_prefix,  # type: ignore[arg-type]
                local_labels_root=labels_root,
                s3_labels_prefix=s3_labels_prefix,
                cache_dir=cache_dir,
                cache_max_bytes=params.disk_cache_bytes,
            )
            self._logger.info(
                "Test evaluation will stream from s3://%s/%s",
                params.s3_bucket,
                params.s3_prefix,
            )

        weights_dir = save_dir / "weights"
        checkpoints: list[tuple[str, Path]] = [
            ("best", weights_dir / "best.pt"),
            ("last", weights_dir / "last.pt"),
        ]

        for label, ckpt_path in checkpoints:
            if not ckpt_path.exists():
                self._logger.warning(
                    "Checkpoint %s not found; skipping test evaluation for '%s'.",
                    ckpt_path,
                    label,
                )
                continue

            try:
                self._logger.info(
                    "Evaluating %s on test split (checkpoint=%s)", label, ckpt_path
                )
                eval_model = yolo_cls(str(ckpt_path))
                val_kwargs: dict[str, Any] = {
                    "data": data_yaml_path,
                    "split": "test",
                    "imgsz": params.image_size,
                    "batch": params.batch_size,
                    "device": params.device,
                    "project": str(tmp_path / "test_eval"),
                    "name": label,
                    "plots": True,
                    "save_json": False,
                }
                if s3_validator_cls is not None:
                    val_kwargs["validator"] = s3_validator_cls
                results = eval_model.val(**val_kwargs)

                self._log_test_eval_results(
                    label=label,
                    results=results,
                    run_id=mlflow_run_id,
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Test evaluation for '%s' failed (non-fatal): %s",
                    label,
                    exc,
                )

    def _is_test_split_available(self, params: TrainingParams) -> bool:
        """Return True iff a test split is reachable under the current mode."""
        if params.source == "local":
            test_dir = Path(params.dataset_dir) / "images" / "test"
            if not test_dir.is_dir():
                return False
            return any(
                f.suffix.lower() in _IMAGE_EXTENSIONS for f in test_dir.iterdir()
            )

        # S3 streaming — consult dataset_manifest.json
        manifest_path = Path(params.dataset_dir) / _MANIFEST_FILENAME
        if not manifest_path.exists():
            return False
        try:
            with manifest_path.open() as fh:
                raw = json.load(fh)
        except Exception:  # noqa: BLE001
            return False
        return bool(raw.get("splits", {}).get("test"))

    def _log_test_eval_results(
        self,
        label: str,
        results: Any,
        run_id: str,
    ) -> None:
        """Log test-eval metrics and plot artifacts to the MLflow run."""
        if not run_id:
            self._logger.warning(
                "No MLflow run_id available; skipping test metric logging for '%s'.",
                label,
            )
            return

        results_dict = getattr(results, "results_dict", None) or {}
        metric_map = {
            _METRIC_PRECISION: f"test/{label}/precision",
            _METRIC_RECALL: f"test/{label}/recall",
            _METRIC_MAP50: f"test/{label}/mAP50",
            _METRIC_MAP50_95: f"test/{label}/mAP50_95",
        }

        try:
            from mlflow.tracking import MlflowClient  # noqa: PLC0415

            client = MlflowClient()
            for src_key, mlflow_key in metric_map.items():
                val = results_dict.get(src_key)
                if val is None:
                    continue
                try:
                    client.log_metric(run_id, mlflow_key, float(val))
                except Exception as exc:  # noqa: BLE001
                    self._logger.warning(
                        "Failed to log metric %s: %s", mlflow_key, exc
                    )

            save_dir = getattr(results, "save_dir", None)
            if save_dir is not None:
                save_dir_path = Path(save_dir)
                if save_dir_path.is_dir():
                    artifact_path = f"eval/test/{label}"
                    try:
                        client.log_artifacts(
                            run_id, str(save_dir_path), artifact_path=artifact_path
                        )
                        self._logger.info(
                            "Logged test/%s metrics and %s artifacts to MLflow",
                            label,
                            artifact_path,
                        )
                    except Exception as exc:  # noqa: BLE001
                        self._logger.warning(
                            "Failed to log test eval artifacts for '%s': %s",
                            label,
                            exc,
                        )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to log test eval results for '%s': %s", label, exc
            )

    # ------------------------------------------------------------------
    # GPU index parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_gpu_index(device: Any) -> "int | None":
        """Extract a single integer GPU index from a device string.

        Returns the integer index when ``device`` is a single GPU (e.g. ``"0"``
        or ``0``).  Returns ``None`` for CPU or multi-GPU strings (``"0,1"``).
        When ``device`` is ``None`` (Ultralytics auto-select), defaults to GPU 0
        if a GPU is available so that the ResourceMonitor still tracks GPU metrics
        on single-GPU machines.

        Examples
        --------
        >>> _parse_gpu_index("0")   # -> 0
        >>> _parse_gpu_index(0)     # -> 0
        >>> _parse_gpu_index("0,1") # -> None  (multi-GPU — monitor skipped)
        >>> _parse_gpu_index("cpu") # -> None
        >>> _parse_gpu_index(None)  # -> 0 if GPU available, else None
        """
        if device is None:
            from app.services.resource_monitor import gpu_available  # noqa: PLC0415

            return 0 if gpu_available() else None
        s = str(device).strip()
        if s.isdigit():
            return int(s)
        return None

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    def _validate_params(self, params: TrainingParams) -> None:
        """Raise TrainingError for mutually exclusive or missing parameters."""
        if params.pretrained_weights and params.resume_from:
            raise TrainingError(
                "--pretrained-weights and --resume-from are mutually exclusive. "
                "Use --pretrained-weights to initialise weights only (epoch 0), "
                "or --resume-from to restore the full training state."
            )

        if params.source == "s3":
            if not params.s3_bucket:
                raise TrainingError(
                    "--s3-bucket is required when --source=s3"
                )
            if not params.s3_prefix:
                raise TrainingError(
                    "--s3-prefix is required when --source=s3"
                )

        dataset_path = Path(params.dataset_dir)
        if not dataset_path.exists():
            raise TrainingError(
                f"--dataset-dir does not exist: {params.dataset_dir}"
            )
        if not dataset_path.is_dir():
            raise TrainingError(
                f"--dataset-dir is not a directory: {params.dataset_dir}"
            )

    # ------------------------------------------------------------------
    # Local dataset validation
    # ------------------------------------------------------------------

    def _validate_local_dataset(self, params: TrainingParams) -> None:
        """Pre-training sanity check for local mode datasets.

        Verifies that data.yaml, images/train/, and images/val/ all exist
        and contain at least one file each. Logs the structure for
        observability. Raises TrainingError on hard failures.
        """
        dataset_path = Path(params.dataset_dir).resolve()

        # data.yaml must exist
        data_yaml = dataset_path / "data.yaml"
        if not data_yaml.exists():
            raise TrainingError(
                f"data.yaml not found in dataset_dir: {dataset_path}. "
                "Run the dataset_loading step first."
            )
        self._logger.info("Local dataset validation | data.yaml found at %s", data_yaml)

        # Check required splits
        for split in ("train", "val"):
            images_dir = dataset_path / "images" / split
            if not images_dir.exists():
                raise TrainingError(
                    f"images/{split}/ directory not found in dataset_dir: {dataset_path}"
                )
            image_files = [
                p for p in images_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
            ]
            if not image_files:
                raise TrainingError(
                    f"images/{split}/ directory is empty (no image files): {images_dir}"
                )
            self._logger.info(
                "Local dataset validation | images/%s: %d files found",
                split,
                len(image_files),
            )

        # Log optional test split presence without raising
        test_dir = dataset_path / "images" / "test"
        if test_dir.exists():
            test_count = sum(
                1 for p in test_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
            )
            self._logger.info(
                "Local dataset validation | images/test: %d files found", test_count
            )

    # ------------------------------------------------------------------
    # Model path resolution
    # ------------------------------------------------------------------

    def _resolve_model_path(
        self,
        params: TrainingParams,
        tmp_path: Path,
    ) -> Path:
        """Determine the path passed to YOLO().

        Priority (highest first):
        1. resume_from — download from S3 if needed; return the .pt path so
           that YOLO() loads the full trainer state (resume=True handled in
           train_kwargs).
        2. pretrained_weights — download from S3 if needed; return the .pt path
           for weight-only initialisation.
        3. model_variant — return the bare variant name so that Ultralytics
           downloads COCO pretrained weights from its CDN on first use.
        """
        if params.resume_from and params.resume_from != "auto":
            return self._maybe_download_pt(params.resume_from, tmp_path, "resume")

        if params.pretrained_weights:
            return self._maybe_download_pt(
                params.pretrained_weights, tmp_path, "pretrained"
            )

        # Bare variant name: Ultralytics handles CDN download
        return Path(params.model_variant)

    def _maybe_download_pt(
        self, uri_or_path: str, tmp_path: Path, label: str
    ) -> Path:
        """Download a .pt file from S3 to tmp_path if uri starts with s3://,
        otherwise treat it as an already-local path and return it directly.
        """
        match = _S3_URI_RE.match(uri_or_path)
        if match:
            bucket = match.group(1)
            key = match.group(2)
            local_pt = tmp_path / f"{label}_weights.pt"
            self._logger.info(
                "Downloading %s weights from s3://%s/%s -> %s",
                label,
                bucket,
                key,
                local_pt,
            )
            self._s3.download_file(bucket, key, str(local_pt))
            return local_pt

        # Local path
        local_pt = Path(uri_or_path)
        if not local_pt.exists():
            raise TrainingError(
                f"{label} weights path does not exist: {uri_or_path}"
            )
        return local_pt

    # ------------------------------------------------------------------
    # data.yaml
    # ------------------------------------------------------------------

    def _write_data_yaml(self, params: TrainingParams, tmp_path: Path) -> Path:
        """Write a data.yaml to the temp directory.

        If dataset_dir already contains a data.yaml (written by dataset_loading),
        we copy it into the temp dir and update the ``path`` field to the
        absolute dataset_dir. Otherwise we generate a minimal pose-estimation
        template.

        The temp dir is cleaned up by the caller's TemporaryDirectory context
        manager, so data.yaml is always deleted after training.
        """
        dataset_path = Path(params.dataset_dir).resolve()
        source_yaml = dataset_path / "data.yaml"
        dest_yaml = tmp_path / "data.yaml"

        if source_yaml.exists():
            with source_yaml.open() as fh:
                content: dict[str, Any] = yaml.safe_load(fh) or {}
        else:
            _logger.warning(
                "No data.yaml found in dataset_dir=%s; generating a default template.",
                params.dataset_dir,
            )
            content = {
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "kpt_shape": [11, 3],
                "flip_idx": [],
                "names": {0: "spacecraft"},
            }

        # Always set path to the resolved absolute dataset directory
        content["path"] = str(dataset_path)

        # In S3 streaming mode the dataset_loading step only downloads labels.
        # Ultralytics validates that image directories exist at init time
        # (before S3PoseTrainer takes over), so we create empty stubs.
        if params.source == "s3":
            for split_key in ("train", "val", "test"):
                rel = content.get(split_key)
                if rel:
                    (dataset_path / rel).mkdir(parents=True, exist_ok=True)

        with dest_yaml.open("w") as fh:
            yaml.dump(content, fh, default_flow_style=False, sort_keys=False)

        self._logger.debug("Wrote data.yaml to %s", dest_yaml)
        return dest_yaml

    # ------------------------------------------------------------------
    # train() kwargs
    # ------------------------------------------------------------------

    def _resolve_device(self, device: "str | None") -> "str | int | None":
        """Resolve the training device, defaulting to GPU when one is present.

        Ultralytics' ``model.train(device=None)`` does NOT auto-select a GPU — it
        silently falls back to CPU. On a GPU node that means we provision a T4,
        allocate ``nvidia.com/gpu: 1`` to the pod, then train on the CPU anyway
        (observed live: trainer logged ``device=cpu`` despite the GPU being bound).
        So when the caller leaves device unset, pick GPU 0 if CUDA is actually
        available and fall back to CPU otherwise — an explicit choice, logged.
        An explicit ``--device`` (including ``"cpu"``) is always honoured.
        """
        if device is not None and str(device).strip() != "":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                self._logger.info(
                    "device unset and CUDA available — selecting GPU 0 "
                    "(Ultralytics device=None would otherwise train on CPU)"
                )
                return "0"
        except Exception as exc:  # torch import / CUDA probe failure → CPU
            self._logger.warning("CUDA probe failed (%s); training on CPU", exc)
        self._logger.info("device unset and no CUDA — training on CPU")
        return "cpu"

    def _build_train_kwargs(
        self,
        params: TrainingParams,
        data_yaml_path: str,
    ) -> dict[str, Any]:
        """Build the keyword arguments dict for model.train().

        When resume_from is 'auto' or a .pt path, Ultralytics restores full
        training state via resume=True. The data and project/name are still
        passed for path resolution.
        """
        aug = params.augmentation

        kwargs: dict[str, Any] = {
            "data": data_yaml_path,
            "epochs": params.epochs,
            "device": self._resolve_device(params.device),
            "batch": params.batch_size,
            "imgsz": params.image_size,
            "lr0": params.learning_rate,
            "cos_lr": params.cos_lr,
            "lrf": params.lrf,
            "optimizer": params.optimizer,
            "momentum": params.momentum,
            "weight_decay": params.weight_decay,
            "warmup_epochs": params.warmup_epochs,
            "warmup_momentum": params.warmup_momentum,
            "dropout": params.dropout,
            "label_smoothing": params.label_smoothing,
            "nbs": params.nbs,
            "amp": params.amp,
            "close_mosaic": params.close_mosaic,
            "workers": params.workers,
            "seed": params.seed,
            "deterministic": params.deterministic,
            "patience": params.patience,
            # Pose loss gains
            "pose": params.pose,
            "kobj": params.kobj,
            "box": params.box,
            "cls": params.cls,
            "dfl": params.dfl,
            # Output location
            "project": params.output_dir,
            "name": params.experiment_name,
            "plots": True,
            "save": True,
            "save_period": params.checkpoint_interval,
            # Augmentation
            "hsv_h": aug.hsv_h,
            "hsv_s": aug.hsv_s,
            "hsv_v": aug.hsv_v,
            "degrees": aug.degrees,
            "translate": aug.translate,
            "scale": aug.scale,
            "shear": aug.shear,
            "perspective": aug.perspective,
            "flipud": aug.flipud,
            "fliplr": aug.fliplr,
            "mosaic": aug.mosaic,
            "mixup": aug.mixup,
            "copy_paste": aug.copy_paste,
            "erasing": aug.erasing,
            "bgr": aug.bgr,
        }

        if params.freeze is not None:
            kwargs["freeze"] = params.freeze

        # Full Ultralytics resume
        if params.resume_from:
            kwargs["resume"] = True

        return kwargs

    # ------------------------------------------------------------------
    # S3 checkpoint upload
    # ------------------------------------------------------------------

    def _upload_final_weights(
        self, params: TrainingParams, save_dir: Path
    ) -> str:
        """Upload best.pt and last.pt to S3 after training completes.

        Returns
        -------
        str
            The s3:// URI of the uploaded best.pt.
        """
        base_key = f"{params.checkpoint_prefix}/{params.experiment_name}"
        best_key = f"{base_key}/best.pt"
        last_key = f"{base_key}/last.pt"

        best_pt = save_dir / "weights" / "best.pt"
        last_pt = save_dir / "weights" / "last.pt"

        best_uri = f"s3://{params.checkpoint_bucket}/{best_key}"

        if best_pt.exists():
            self._upload_to_s3(best_pt, params.checkpoint_bucket, best_key)
            self._logger.info("Uploaded best.pt to %s", best_uri)
        else:
            self._logger.warning(
                "best.pt not found at %s; skipping S3 upload. "
                "Downstream steps will not have a best checkpoint.",
                best_pt,
            )
            best_uri = ""

        if last_pt.exists():
            self._upload_to_s3(last_pt, params.checkpoint_bucket, last_key)
            self._logger.info(
                "Uploaded last.pt to s3://%s/%s",
                params.checkpoint_bucket,
                last_key,
            )

        # S3-gateway writes only STAGE objects on the lakeFS branch — commit so
        # the checkpoints are a versioned artifact (one commit captures the best/
        # last/epoch files staged during this run), not dangling changes.
        branch = params.checkpoint_prefix.split("/", 1)[0] or "main"
        self._lakefs_commit(
            params.checkpoint_bucket,
            branch,
            f"model-training: checkpoints {params.experiment_name}",
        )

        return best_uri

    def _lakefs_commit(self, repo: str, branch: str, message: str) -> None:
        """Commit staged lakeFS changes on ``branch`` (S3-gateway only stages).

        Best-effort: a failure (including 'nothing to commit') never fails the
        step — the objects are already uploaded. Uses the lakeFS REST API on the
        same endpoint as the S3 gateway with the injected lakeFS credentials.
        """
        import base64
        import urllib.request

        endpoint = os.environ.get("LAKEFS_ENDPOINT", "").rstrip("/")
        access = os.environ.get("LAKEFS_ACCESS_KEY", "")
        secret = os.environ.get("LAKEFS_SECRET_KEY", "")
        if not (endpoint and access and secret):
            self._logger.warning("lakeFS commit skipped — endpoint/creds unset.")
            return
        url = f"{endpoint}/api/v1/repositories/{repo}/branches/{branch}/commits"
        auth = base64.b64encode(f"{access}:{secret}".encode()).decode()
        req = urllib.request.Request(
            url, data=json.dumps({"message": message}).encode(), method="POST"
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Basic {auth}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                cid = str(json.loads(resp.read().decode()).get("id", "?"))
            self._logger.info("lakeFS commit %s @ %s/%s", cid[:12], repo, branch)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("lakeFS commit failed (non-fatal): %s", exc)

    def _upload_to_s3(self, local_path: Path, bucket: str, key: str) -> None:
        """Upload a single file to S3."""
        try:
            self._s3.upload_file(str(local_path), bucket, key)
        except Exception as exc:
            raise TrainingError(
                f"S3 upload failed for s3://{bucket}/{key}: {exc}"
            ) from exc
