# Agent Memory: YOLO + MLflow Trainer — model_training step

## Project Identity
- Pipeline: Infinite Orbits spacecraft pose estimation (SPEED+ dataset)
- Task: 11-keypoint pose estimation, single class "spacecraft"
- YOLO variants in use: yolov{8|9|10|11}{n|s|m|l|x}-pose.pt (validated by config_validation)
- Label format: 1 class + 4 bbox + 11*3 keypoints = 38 tokens per line
- Default YOLO variant for experiments: yolov8n-pose.pt

## File Layout (model_training step)
- app/cli.py — Typer CLI with all hyperparameters as flags
- app/logger.py — ColorFormatter + setup_logging (identical to dataset_loading/app/logger.py)
- app/manager.py — wires Config + boto3 + TrainingService; uses setup_logging()
- app/models/config.py — Pydantic BaseSettings (env vars / .env)
- app/models/training.py — TrainingParams, AugmentationParams, TrainingResult
- app/services/model_training.py — TrainingService (core orchestrator)
- app/services/s3_dataset.py — S3YoloDataset (streaming mode)
- app/services/s3_pose_trainer.py — S3PoseTrainer + make_s3_pose_trainer factory
- app/services/resource_monitor.py — ResourceMonitor (background thread)
- app/services/lru_disk_cache.py — LruDiskCache for S3 image caching

## Key Design Decisions (confirmed with user)
- --pretrained-weights: loads weights only, epoch 0 restart
- --resume-from: full Ultralytics resume=True (restores epoch+optimizer+scheduler)
- Both accept s3:// URIs, downloaded to tempfile.TemporaryDirectory (cleaned up in finally)
- data.yaml always written to tempfile.TemporaryDirectory, deleted after training
- source auto-detected from dataset_manifest.json if present (overrides CLI --source)
- source="local" reads from dataset_dir; source="s3" uses S3YoloDataset streaming
- No ONNX export in this step; no MLflow model registry (left to model_registration)
- S3 checkpoints: s3://<checkpoint_bucket>/<checkpoint_prefix>/<experiment_name>/
- Experiment name comes from --experiment-name CLI flag (not env var)

## Manager.run() Signature Note
- source, s3_bucket, s3_prefix have defaults (= "local", = None, = None)
- disk_cache_bytes has a default (= 2GB)
- device has a default (= None, Ultralytics auto-select)
- All remaining params (epochs, batch_size, etc.) are keyword-only (after `*`)
- This was required to fix pre-existing SyntaxError: param without default after param with default

## Manifest Auto-Detection Flow
- `TrainingService.run()` calls `_apply_manifest_if_present(params)` FIRST (before _validate_params)
- Reads `{dataset_dir}/dataset_manifest.json`; if bucket+prefix found, creates new params via
  `params.model_copy(update={"source": "s3", "s3_bucket": bucket, "s3_prefix": prefix})`
- Malformed manifest or missing fields → logs warning, returns original params unchanged
- After detection, _validate_params still enforces s3_bucket/s3_prefix when source='s3'

## Local Mode Validation (_validate_local_dataset)
- Called in `_run_training` when `params.source == "local"` (before training starts)
- Checks: data.yaml exists, images/train/ and images/val/ exist with >=1 image file
- Raises TrainingError on failure; logs split counts at INFO level
- Image extensions: {.jpg, .jpeg, .png, .bmp, .webp}

## ResourceMonitor — Confirmed Design (epoch-end sampler, no thread)
- Constructor: ResourceMonitor(gpu_index=Optional[int])  — no interval_sec
- gpu_index=None → GPU metrics skipped entirely (even if pynvml available)
- gpu_index=0 → only that device is sampled (no suffix on metric keys)
- collect() → dict[str, float]: returns a snapshot of system metrics; caller logs to MLflow
- No background thread, no run_id, no step counter, no start()/stop()
- Called from on_fit_epoch_end; result merged with val_metrics and logged via mlflow.log_metrics
- All errors caught inside collect(); returns {} on unexpected failure (never raises)
- gpu_index is derived by TrainingService._parse_gpu_index(params.device):
  "0" → 0, 0 → 0, "0,1" → None, "cpu" → None, None → None
- resource_monitor_interval_sec REMOVED from Config and TrainingService.__init__

## TrainingParams.device field
- Optional[str], default None (Ultralytics auto-select)
- Passed through cli.py → Manager.run() → TrainingParams → _build_train_kwargs as "device" kwarg
- Also consumed by _parse_gpu_index to drive ResourceMonitor GPU targeting

## Dataset Stats MLflow Logging
- `_read_dataset_stats()` reads `{dataset_dir}/dataset_stats.json` → None if absent/corrupt
- `_log_params_and_tags(params, active_run=None, dataset_stats=None)` merges stats as `dataset.*`
- Logged fields: train_images, val_images, test_images, train_labels, val_labels, test_labels,
  version, source, sampled, sample_size

## MLflow Logging
- Params logged: all hyperparameters in batches of <=100 (MLflow limit)
- Per-epoch metrics via on_fit_epoch_end callback (captured + fired by test mock)
- System metrics via ResourceMonitor thread (MlflowClient.log_metric, keyed by run_id)
- Artifacts after training: weights/best.pt, weights/last.pt, plots/*.png, metrics/results.csv
- Tags: model.variant, dataset.source, pipeline.step, project, training.status

## Test Patterns (important gotchas)
- `dataset_dir` fixture MUST create images/train/ and images/val/ with >=1 .jpg file
  (local validation runs inside _run_training, not just in run())
- `_run_with_mocks` uses `mock_model.add_callback.side_effect` to capture callbacks;
  `mock_model.train.side_effect` fires `on_fit_epoch_end` so epoch_metrics is populated
- `artifact_path` in `mlflow.log_artifact` is a KEYWORD arg → `c[1]["artifact_path"]` not `c[0][1]`
- Patch make_s3_pose_trainer at source: `app.services.s3_pose_trainer.make_s3_pose_trainer`
  (it's a lazy import inside the if block, not a module-level name in model_training.py)
- ResourceMonitor tests: patch `app.services.resource_monitor.psutil` and
  `app.services.resource_monitor._GPU_AVAILABLE`; assert on the returned dict from collect()
  No MlflowClient, no _seed_run_id, no thread — tests are fully synchronous
- TrainingService run tests: patch `app.services.resource_monitor.psutil` (for collect()),
  `app.services.resource_monitor._GPU_AVAILABLE`, and `mlflow.log_metrics` (called inside
  on_fit_epoch_end closure). No longer need to patch MlflowClient or mlflow.active_run.
- Pre-existing failures: test_config_default_values, test_s3_dataset (cv2 environment issue)

## S3 Credential Pattern (same as dataset_loading)
- AWS: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, S3_ENDPOINT_URL
- LakeFS: LAKEFS_ENDPOINT, LAKEFS_ACCESS_KEY, LAKEFS_SECRET_KEY (takes precedence when set)

## See Also
- patterns.md — detailed notes on Ultralytics callback hooks and metric key names
