# YOLO MLflow Trainer — Agent Memory

## Project layout (model_training step)

- `app/cli.py` — Typer CLI; ~40 flags; calls `Manager().run(**kwargs)`
- `app/manager.py` — wires Config + boto3 S3 client + TrainingService; packages flat kwargs into TrainingParams/AugmentationParams
- `app/models/config.py` — BaseSettings; MLflow URI/credentials, AWS/LakeFS S3 credentials, log level
- `app/models/training.py` — TrainingParams, AugmentationParams, TrainingResult (Pydantic v2)
- `app/services/model_training.py` — TrainingService; main orchestration
- `app/services/resource_monitor.py` — ResourceMonitor.collect(); on-demand snapshot, no background thread
- `app/services/s3_pose_trainer.py` — make_s3_pose_trainer() factory; S3PoseTrainer subclasses PoseTrainer
- `app/services/s3_dataset.py` — S3YoloDataset subclasses YOLODataset; streams images from S3 with LruDiskCache
- `app/services/lru_disk_cache.py` — thread-safe bounded LRU disk cache (OrderedDict + lock)

## Key architectural decisions

- MLflow logging is handled entirely by the Ultralytics built-in MLflow callback (set via MLFLOW_TRACKING_URI / MLFLOW_EXPERIMENT_NAME env vars). The service only adds supplemental per-epoch val metrics + system resource metrics via a custom `on_fit_epoch_end` callback.
- `resource_monitor_interval_sec` was REMOVED from Config. ResourceMonitor is synchronous (no background thread); `collect()` is called from the Ultralytics epoch-end callback.
- Ultralytics and mlflow imports are deferred inside methods (not at module top) so unit tests can mock them without importing the real packages.
- `_parse_gpu_index(None)` defaults to GPU 0 when `gpu_available()` is True (single-GPU auto-select).
- `_upload_final_weights` returns `""` (not a phantom URI) when best.pt is missing.
- `mlflow.last_active_run()` logs a warning when it returns None; does not raise.
- `epochs_completed = trainer.epoch + 1` (trainer.epoch is 0-indexed); falls back to `params.epochs` when attribute is absent.

## TrainingService callback pattern

The four Ultralytics callbacks are NOT closures nested inside `_run_training`. They are factory methods that return closures:

- `_make_batch_end_callback(epoch_metrics)` — captures per-batch losses into epoch_metrics dict
- `_make_epoch_end_callback(epoch_metrics, monitor)` — logs val metrics + system resources to MLflow at each epoch
- `_make_checkpoint_callback(params)` — uploads last.pt to S3 every `checkpoint_interval` epochs (uses `self`)
- `_make_train_end_callback(epoch_metrics)` — logs completion summary

`_run_training` registers all four via `model.add_callback(event, factory_method(...))` then calls `model.train()`.

## MLflow run ID retrieval

Extracted to `_get_mlflow_run_id()` static method. Called after `model.train()` completes, outside the TemporaryDirectory context.

## Module-level constants in model_training.py

- `_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}` — used in `_validate_local_dataset` (replaces two inline sets)
- `_S3_URI_RE`, `_METRIC_*`, `_MANIFEST_FILENAME` — all module-level

## Test patterns

- All heavy deps mocked: `ultralytics.YOLO`, `mlflow`, `boto3`, `pynvml`, `psutil`
- Callbacks tested by capturing them via `model.add_callback.side_effect` and firing them synchronously from within a `_fake_train()` side effect
- `patch("app.services.resource_monitor._GPU_AVAILABLE", False)` disables GPU branch in tests
- `patch("app.services.resource_monitor.psutil")` mocks CPU/RAM reads
- Test file: `tests/test_training_service.py` (47 tests), `tests/test_resource_monitor.py` (8 tests), `tests/test_manager.py` (5 tests)

## Confirmed YOLO variant

- `yolov8n-pose.pt` used in all tests; production uses pose estimation variants

## S3 / LakeFS

- LakeFS credentials (LAKEFS_ENDPOINT, LAKEFS_ACCESS_KEY, LAKEFS_SECRET_KEY) take precedence over AWS credentials in `_build_s3_client`
- S3 streaming mode is auto-detected from `dataset_manifest.json` in dataset_dir; overrides --source/--s3-bucket/--s3-prefix CLI flags
- Intermediate checkpoints uploaded to S3 every `checkpoint_interval` epochs as `<prefix>/<experiment_name>/epoch_NNNN.pt`
- Final weights uploaded as `<prefix>/<experiment_name>/best.pt` and `last.pt`
