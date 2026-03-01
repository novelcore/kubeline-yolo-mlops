"""YOLO model training service with MLflow experiment tracking.

Responsibilities
----------------
1. Validate that --pretrained-weights and --resume-from are not both set.
2. Download weights / resume checkpoint from S3 to a temp dir when needed.
3. Write data.yaml to a temp dir (always deleted in a finally block).
4. Start the background resource monitor.
5. Register Ultralytics callbacks for per-epoch MLflow metric logging and
   periodic S3 checkpoint uploads.
6. Call model.train() with the full hyperparameter set.
7. Stop the resource monitor.
8. Log artifacts (best.pt, last.pt, plots, results.csv) to MLflow.
9. Upload best.pt and last.pt to the S3 checkpoint path.
10. Clean up temp dirs.
"""

import logging
import re
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import yaml

from app.models.training import TrainingParams, TrainingResult
from app.services.resource_monitor import ResourceMonitor

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_S3_URI_RE = re.compile(r"^s3://([^/]+)/(.+)$")

# Ultralytics metric keys
_METRIC_PRECISION = "metrics/precision(B)"
_METRIC_RECALL = "metrics/recall(B)"
_METRIC_MAP50 = "metrics/mAP50(B)"
_METRIC_MAP50_95 = "metrics/mAP50-95(B)"


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class TrainingError(Exception):
    """Raised when a training run fails unrecoverably."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class TrainingService:
    """Orchestrates a YOLO training run with MLflow tracking and S3 I/O.

    Parameters
    ----------
    s3_client:
        A pre-constructed boto3 S3 client used for checkpoint download/upload
        and (when source='s3') image streaming.
    mlflow_tracking_uri:
        URI of the remote MLflow tracking server.
    resource_monitor_interval_sec:
        Seconds between resource metric samples during training.
    """

    def __init__(
        self,
        s3_client: Any,
        mlflow_tracking_uri: str,
        resource_monitor_interval_sec: int = 30,
    ) -> None:
        self._s3 = s3_client
        self._mlflow_tracking_uri = mlflow_tracking_uri
        self._monitor_interval = resource_monitor_interval_sec
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
        self._validate_params(params)

        # Defer heavy imports so that unit tests can mock them easily
        from ultralytics import YOLO  # noqa: PLC0415

        Path(params.output_dir).mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(self._mlflow_tracking_uri)
        mlflow.set_experiment(params.experiment_name)

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_name = f"{params.model_variant.removesuffix('.pt')}-{timestamp}"

        with mlflow.start_run(run_name=run_name) as active_run:
            try:
                result = self._run_with_mlflow(params, active_run, YOLO)
            except Exception as exc:
                # Tag the failed run so it is easy to filter in the UI
                try:
                    mlflow.set_tag("training.status", "FAILED")
                    mlflow.set_tag("training.error", str(exc)[:500])
                except Exception:  # noqa: BLE001
                    pass
                raise TrainingError(f"Training failed: {exc}") from exc

        return result

    # ------------------------------------------------------------------
    # Core pipeline (executed inside an active MLflow run)
    # ------------------------------------------------------------------

    def _run_with_mlflow(
        self,
        params: TrainingParams,
        active_run: Any,
        yolo_cls: Any,
    ) -> TrainingResult:
        """Inner pipeline: weights, data.yaml, callbacks, train(), artifacts."""

        monitor = ResourceMonitor(interval_sec=self._monitor_interval)

        with tempfile.TemporaryDirectory(prefix="io-model-training-") as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Resolve model path (download from S3 if needed)
            model_path = self._resolve_model_path(params, tmp_path)

            # 2. Write data.yaml into the temp dir
            data_yaml_path = self._write_data_yaml(params, tmp_path)

            # 3. Log hyperparameters and tags to MLflow
            self._log_params_and_tags(params)

            # 4. Build the Ultralytics YOLO model object
            model = yolo_cls(str(model_path))

            # 5. Register per-epoch MLflow callback
            epoch_metrics: dict[str, float] = {}

            def on_fit_epoch_end(trainer: Any) -> None:
                """Log per-epoch train + val metrics to MLflow."""
                try:
                    epoch: int = trainer.epoch
                    loss_items = getattr(trainer, "loss_items", None)

                    metrics_to_log: dict[str, float] = {}

                    # Loss items order for YOLOv8-pose:
                    # 0=box, 1=pose, 2=kobj, 3=cls, 4=dfl
                    if loss_items is not None and hasattr(loss_items, "__len__"):
                        loss_names = ["box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"]
                        for idx, loss_name in enumerate(loss_names):
                            if idx < len(loss_items):
                                val = loss_items[idx]
                                metrics_to_log[f"train/{loss_name}"] = float(val)

                    val_metrics = getattr(trainer, "metrics", {}) or {}
                    metrics_to_log["val/precision"] = float(
                        val_metrics.get(_METRIC_PRECISION, 0.0)
                    )
                    metrics_to_log["val/recall"] = float(
                        val_metrics.get(_METRIC_RECALL, 0.0)
                    )
                    metrics_to_log["val/mAP50"] = float(
                        val_metrics.get(_METRIC_MAP50, 0.0)
                    )
                    metrics_to_log["val/mAP50_95"] = float(
                        val_metrics.get(_METRIC_MAP50_95, 0.0)
                    )

                    try:
                        mlflow.log_metrics(metrics_to_log, step=epoch)
                    except Exception as mlflow_exc:  # noqa: BLE001
                        _logger.warning(
                            "MLflow metric logging failed at epoch %d: %s",
                            epoch,
                            mlflow_exc,
                        )

                    # Keep the last epoch metrics for the TrainingResult
                    epoch_metrics.update(metrics_to_log)

                except Exception as cb_exc:  # noqa: BLE001
                    _logger.warning("on_fit_epoch_end callback error: %s", cb_exc)

            # 6. Register periodic S3 checkpoint upload callback
            def on_train_epoch_end(trainer: Any) -> None:
                """Upload checkpoint to S3 every checkpoint_interval epochs."""
                try:
                    epoch: int = trainer.epoch + 1  # Ultralytics epoch is 0-indexed
                    if epoch % params.checkpoint_interval != 0:
                        return
                    last_pt = Path(trainer.last)
                    if not last_pt.exists():
                        return
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
                except Exception as cb_exc:  # noqa: BLE001
                    _logger.warning("S3 checkpoint upload failed: %s", cb_exc)

            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            # 7. Start resource monitor
            monitor.start()

            # 8. Build train() kwargs — resume mode sets resume=True
            train_kwargs = self._build_train_kwargs(
                params=params,
                data_yaml_path=str(data_yaml_path),
            )

            try:
                trainer = model.train(**train_kwargs)
            finally:
                monitor.stop()

            # 9. Determine save directory
            save_dir = Path(model.trainer.save_dir)  # type: ignore[union-attr]

            # 10. Log artifacts to MLflow
            self._log_artifacts(save_dir)

            # 11. Upload final weights to S3
            best_s3_uri = self._upload_final_weights(params, save_dir)

            # 12. Mark run as successful
            mlflow.set_tag("training.status", "SUCCEEDED")

        # Build result
        final_map50 = float(epoch_metrics.get("val/mAP50", 0.0))
        final_map50_95 = float(epoch_metrics.get("val/mAP50_95", 0.0))

        epochs_completed: int = getattr(trainer, "epoch", params.epochs) + 1

        return TrainingResult(
            experiment_name=params.experiment_name,
            model_variant=params.model_variant,
            mlflow_run_id=active_run.info.run_id,
            best_checkpoint_local=str(save_dir / "weights" / "best.pt"),
            best_checkpoint_s3=best_s3_uri,
            epochs_completed=epochs_completed,
            final_map50=final_map50,
            final_map50_95=final_map50_95,
        )

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

        with dest_yaml.open("w") as fh:
            yaml.dump(content, fh, default_flow_style=False, sort_keys=False)

        self._logger.debug("Wrote data.yaml to %s", dest_yaml)
        return dest_yaml

    # ------------------------------------------------------------------
    # MLflow params and tags
    # ------------------------------------------------------------------

    def _log_params_and_tags(self, params: TrainingParams) -> None:
        """Log all hyperparameters as MLflow params and set run tags."""
        aug = params.augmentation

        all_params: dict[str, Any] = {
            # Identity
            "model.variant": params.model_variant,
            "experiment.name": params.experiment_name,
            "dataset.dir": params.dataset_dir,
            "dataset.source": params.source,
            # Schedule
            "training.epochs": params.epochs,
            "training.batch_size": params.batch_size,
            "training.image_size": params.image_size,
            # LR
            "training.learning_rate": params.learning_rate,
            "training.cos_lr": params.cos_lr,
            "training.lrf": params.lrf,
            # Optimizer
            "training.optimizer": params.optimizer,
            "training.momentum": params.momentum,
            "training.weight_decay": params.weight_decay,
            # Warmup
            "training.warmup_epochs": params.warmup_epochs,
            "training.warmup_momentum": params.warmup_momentum,
            # Regularization
            "training.dropout": params.dropout,
            "training.label_smoothing": params.label_smoothing,
            # Efficiency
            "training.nbs": params.nbs,
            "training.freeze": str(params.freeze),
            "training.amp": params.amp,
            "training.close_mosaic": params.close_mosaic,
            "training.seed": params.seed,
            "training.deterministic": params.deterministic,
            # Loss gains
            "training.pose": params.pose,
            "training.kobj": params.kobj,
            "training.box": params.box,
            "training.cls": params.cls,
            "training.dfl": params.dfl,
            # Early stopping
            "training.patience": params.patience,
            # Augmentation
            "augmentation.hsv_h": aug.hsv_h,
            "augmentation.hsv_s": aug.hsv_s,
            "augmentation.hsv_v": aug.hsv_v,
            "augmentation.degrees": aug.degrees,
            "augmentation.translate": aug.translate,
            "augmentation.scale": aug.scale,
            "augmentation.shear": aug.shear,
            "augmentation.perspective": aug.perspective,
            "augmentation.flipud": aug.flipud,
            "augmentation.fliplr": aug.fliplr,
            "augmentation.mosaic": aug.mosaic,
            "augmentation.mixup": aug.mixup,
            "augmentation.copy_paste": aug.copy_paste,
            "augmentation.erasing": aug.erasing,
            "augmentation.bgr": aug.bgr,
        }

        # MLflow enforces a max of 100 params per call
        items = list(all_params.items())
        batch_size = 100
        for i in range(0, len(items), batch_size):
            batch = dict(items[i : i + batch_size])
            # Convert all values to str/int/float (mlflow requirement)
            safe_batch = {k: str(v) if not isinstance(v, (int, float)) else v for k, v in batch.items()}
            try:
                mlflow.log_params(safe_batch)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("MLflow log_params failed: %s", exc)

        try:
            mlflow.set_tags(
                {
                    "model.variant": params.model_variant,
                    "dataset.source": params.source,
                    "pipeline.step": "model_training",
                    "project": "infinite-orbits",
                    "experiment.name": params.experiment_name,
                    "training.status": "RUNNING",
                }
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("MLflow set_tags failed: %s", exc)

    # ------------------------------------------------------------------
    # train() kwargs
    # ------------------------------------------------------------------

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
            # Disable Ultralytics' own MLflow integration to avoid double-logging
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
    # Artifact logging
    # ------------------------------------------------------------------

    def _log_artifacts(self, save_dir: Path) -> None:
        """Log weights, plots, and results.csv to the active MLflow run."""
        # Weights
        for weight_name in ("best.pt", "last.pt"):
            weight_path = save_dir / "weights" / weight_name
            if weight_path.exists():
                try:
                    mlflow.log_artifact(str(weight_path), artifact_path="weights")
                    _logger.debug("Logged artifact: %s", weight_path)
                except Exception as exc:  # noqa: BLE001
                    _logger.warning("Failed to log artifact %s: %s", weight_path, exc)
            else:
                _logger.warning("Weight file not found, skipping: %s", weight_path)

        # Plots (confusion matrix, PR curve, results plots, etc.)
        for plot_path in save_dir.glob("*.png"):
            try:
                mlflow.log_artifact(str(plot_path), artifact_path="plots")
            except Exception as exc:  # noqa: BLE001
                _logger.warning("Failed to log plot %s: %s", plot_path, exc)

        # Per-epoch CSV
        results_csv = save_dir / "results.csv"
        if results_csv.exists():
            try:
                mlflow.log_artifact(str(results_csv), artifact_path="metrics")
            except Exception as exc:  # noqa: BLE001
                _logger.warning("Failed to log results.csv: %s", exc)

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
            warnings.warn(f"best.pt not found at {best_pt}; skipping S3 upload.", stacklevel=2)

        if last_pt.exists():
            self._upload_to_s3(last_pt, params.checkpoint_bucket, last_key)
            self._logger.info(
                "Uploaded last.pt to s3://%s/%s",
                params.checkpoint_bucket,
                last_key,
            )

        return best_uri

    def _upload_to_s3(self, local_path: Path, bucket: str, key: str) -> None:
        """Upload a single file to S3."""
        try:
            self._s3.upload_file(str(local_path), bucket, key)
        except Exception as exc:
            raise TrainingError(
                f"S3 upload failed for s3://{bucket}/{key}: {exc}"
            ) from exc
