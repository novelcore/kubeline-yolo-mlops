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
7. Start the background resource monitor.
8. Register Ultralytics callbacks for console output and periodic S3
   checkpoint uploads.
9. Call model.train() with the full hyperparameter set.
10. Stop the resource monitor.
11. Upload best.pt and last.pt to the S3 checkpoint path.
12. Clean up temp dirs.
"""

import json
import logging
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any

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
        # Auto-detect S3 streaming mode from manifest before validation
        params = self._apply_manifest_if_present(params)

        self._validate_params(params)

        # Defer heavy imports so that unit tests can mock them easily
        from ultralytics import YOLO  # noqa: PLC0415

        Path(params.output_dir).mkdir(parents=True, exist_ok=True)

        # Configure Ultralytics' built-in MLflow callback via env vars
        os.environ["MLFLOW_TRACKING_URI"] = self._mlflow_tracking_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = params.experiment_name

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

        monitor = ResourceMonitor(interval_sec=self._monitor_interval)

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

            # 4. Register callbacks for console output and S3 checkpoints
            epoch_metrics: dict[str, float] = {}

            def on_train_batch_end(trainer: Any) -> None:
                """Capture per-batch training losses for TrainingResult."""
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

            def on_fit_epoch_end(trainer: Any) -> None:
                """Capture per-epoch validation metrics for TrainingResult."""
                try:
                    val_metrics_raw = getattr(trainer, "metrics", {}) or {}
                    val_metrics: dict[str, float] = {
                        "val/precision": float(val_metrics_raw.get(_METRIC_PRECISION, 0.0)),
                        "val/recall": float(val_metrics_raw.get(_METRIC_RECALL, 0.0)),
                        "val/mAP50": float(val_metrics_raw.get(_METRIC_MAP50, 0.0)),
                        "val/mAP50_95": float(val_metrics_raw.get(_METRIC_MAP50_95, 0.0)),
                    }
                    epoch_metrics.update(val_metrics)

                except Exception as cb_exc:  # noqa: BLE001
                    _logger.warning("on_fit_epoch_end callback error: %s", cb_exc)

            # Periodic S3 checkpoint upload callback
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

            def on_train_end(trainer: Any) -> None:
                """Log training completion summary."""
                _logger.info(
                    "Training complete  best mAP50-95=%.4f  saved to %s",
                    float(epoch_metrics.get("val/mAP50_95", 0.0)),
                    getattr(trainer, "best", "?"),
                )

            model.add_callback("on_train_batch_end", on_train_batch_end)
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            model.add_callback("on_train_end", on_train_end)

            # 5. Start resource monitor
            monitor.start()

            # 6. Build train() kwargs — resume mode sets resume=True
            train_kwargs = self._build_train_kwargs(
                params=params,
                data_yaml_path=str(data_yaml_path),
            )

            # 6b. S3 streaming mode — inject S3PoseTrainer
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

            try:
                trainer = model.train(**train_kwargs)
            finally:
                monitor.stop()

            # 7. Determine save directory
            save_dir = Path(model.trainer.save_dir)  # type: ignore[union-attr]

            # 8. Upload final weights to S3
            best_s3_uri = self._upload_final_weights(params, save_dir)

        # Build result — get MLflow run_id from Ultralytics' run
        mlflow_run_id = ""
        try:
            import mlflow  # noqa: PLC0415

            last_run = mlflow.last_active_run()
            if last_run:
                mlflow_run_id = last_run.info.run_id
        except Exception:  # noqa: BLE001
            pass

        final_map50 = float(epoch_metrics.get("val/mAP50", 0.0))
        final_map50_95 = float(epoch_metrics.get("val/mAP50_95", 0.0))

        epochs_completed: int = getattr(trainer, "epoch", params.epochs) + 1

        return TrainingResult(
            experiment_name=params.experiment_name,
            model_variant=params.model_variant,
            mlflow_run_id=mlflow_run_id,
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
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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
