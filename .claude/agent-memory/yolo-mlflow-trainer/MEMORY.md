# Agent Memory: YOLO + MLflow Trainer — model_training step

## Project Identity
- Pipeline: Infinite Orbits spacecraft pose estimation (SPEED+ dataset)
- Task: 11-keypoint pose estimation, single class "spacecraft"
- YOLO variants in use: yolov{8|9|10|11}{n|s|m|l|x}-pose.pt (validated by config_validation)
- Label format: 1 class + 4 bbox + 11*3 keypoints = 38 tokens per line
- Default YOLO variant for experiments: yolov8n-pose.pt

## File Layout (model_training step)
- app/cli.py — Typer CLI with all hyperparameters as flags
- app/manager.py — wires Config + boto3 + TrainingService
- app/models/config.py — Pydantic BaseSettings (env vars / .env)
- app/models/training.py — TrainingParams, AugmentationParams, TrainingResult
- app/services/model_training.py — TrainingService (core orchestrator)
- app/services/s3_dataset.py — S3YoloDataset (streaming mode)
- app/services/resource_monitor.py — ResourceMonitor (background thread)
- No __init__.py needed in models/ or services/ subdirectories (confirmed pattern)

## Key Design Decisions (confirmed with user)
- --pretrained-weights: loads weights only, epoch 0 restart (like HF model_name_or_path)
- --resume-from: full Ultralytics resume=True (restores epoch+optimizer+scheduler)
- Both accept s3:// URIs, downloaded to tempfile.TemporaryDirectory (cleaned up in finally)
- data.yaml always written to tempfile.TemporaryDirectory, deleted after training
- source="local" reads from dataset_dir; source="s3" uses S3YoloDataset streaming
- No ONNX export in this step
- No MLflow model registry in this step (left to model_registration)
- S3 checkpoints: s3://<checkpoint_bucket>/<checkpoint_prefix>/<experiment_name>/
- Intermediate checkpoints uploaded every checkpoint_interval epochs via on_train_epoch_end callback
- Final best.pt and last.pt uploaded after training completes
- Experiment name comes from --experiment-name CLI flag (not env var)
- MLflow tracking URI from MLFLOW_TRACKING_URI env var

## MLflow Logging
- Params logged: all hyperparameters in batches of <=100 (MLflow limit)
- Per-epoch metrics via on_fit_epoch_end callback: train/box_loss, train/pose_loss,
  train/kobj_loss, train/cls_loss, train/dfl_loss, val/precision, val/recall,
  val/mAP50, val/mAP50_95
- System metrics via ResourceMonitor thread: system/ram_used_gb, system/ram_percent,
  system/cpu_percent, system/gpu_vram_used_gb, system/gpu_vram_total_gb,
  system/gpu_utilization_pct (+ _gpu{i} suffix for multi-GPU)
- Artifacts after training: weights/best.pt, weights/last.pt, plots/*.png, metrics/results.csv
- Tags: model.variant, dataset.source, pipeline.step=model_training, project=infinite-orbits,
  training.status (RUNNING -> SUCCEEDED or FAILED)

## S3 Credential Pattern (same as dataset_loading)
- AWS: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, S3_ENDPOINT_URL
- LakeFS: LAKEFS_ENDPOINT, LAKEFS_ACCESS_KEY, LAKEFS_SECRET_KEY
- LakeFS takes precedence when LAKEFS_ENDPOINT is set

## Test Patterns
- All tests mock ultralytics.YOLO, mlflow, boto3, psutil, pynvml
- S3YoloDataset tests use importlib.reload to patch sys.modules before import
- TrainingService tests build fake Ultralytics save_dir structure in tmp_path
- Manager tests mock TrainingService at app.manager.TrainingService
- ResourceMonitor tests patch app.services.resource_monitor._GPU_AVAILABLE

## Augmentation Notes for Spacecraft Pose
- fliplr=0.0 (no horizontal flip — satellites have no left-right symmetry)
- flipud=0.0 (no vertical flip — orbital geometry has defined "up")
- degrees=0.0 (rotation breaks keypoint geometry)
- perspective max 0.001 (severe keypoint displacement beyond this)

## See Also
- patterns.md — detailed notes on Ultralytics callback hooks and metric key names
