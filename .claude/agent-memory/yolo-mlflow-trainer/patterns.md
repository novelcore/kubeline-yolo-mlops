# Patterns: Ultralytics Callbacks and Metric Keys

## Ultralytics Callback Hooks Used
- on_fit_epoch_end(trainer): fired after each epoch's validation; trainer.metrics
  contains val metrics, trainer.loss_items contains train losses
- on_train_epoch_end(trainer): fired after training portion of each epoch;
  trainer.epoch is 0-indexed; trainer.last is path to last.pt

## YOLOv8-Pose loss_items index order
0: box loss
1: pose loss (keypoint regression)
2: kobj loss (keypoint objectness)
3: cls loss
4: dfl loss

## Ultralytics metric keys (trainer.metrics dict)
"metrics/precision(B)"
"metrics/recall(B)"
"metrics/mAP50(B)"
"metrics/mAP50-95(B)"

## model.train() save directory
After training: model.trainer.save_dir contains the run output.
Weights at: <save_dir>/weights/best.pt and <save_dir>/weights/last.pt
Plots at: <save_dir>/*.png
CSV at: <save_dir>/results.csv

## resume=True behavior
Ultralytics reads the full trainer state from the .pt file metadata.
Must be passed as train kwarg: model.train(resume=True, ...)
The YOLO() constructor must be given the checkpoint path (not the variant name)
when resuming.

## Ultralytics MLflow auto-integration
Ultralytics has built-in MLflow support that can fire if mlflow is installed.
Disable it by NOT setting the MLFLOW_TRACKING_URI env var inside the container
(or set mlflow.set_tracking_uri explicitly before training to override).
We manage MLflow ourselves via callbacks; do not double-enable.

## S3YoloDataset architecture note
Ultralytics calls get_img_files() to enumerate samples and load_image(i) per
sample. We override both. Label resolution uses img2label_paths() which we
also override to map synthetic s3:// URIs to local label txt paths.
The im_files list must be populated with our synthetic URIs before training.
