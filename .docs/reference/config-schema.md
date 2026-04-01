# Parameter Schema

Complete parameter reference for the Argo WorkflowTemplate. All parameters are strings (Argo passes all values as strings). Every parameter has a default — you only override what you need when submitting a workflow.

## Experiment

| Parameter | Default | Description |
| --- | --- | --- |
| `experiment-name` | `""` | Name for the MLflow experiment. All runs with the same name are grouped together. **Required.** |
| `experiment-description` | `""` | Free-text description shown in MLflow. |

## Dataset

| Parameter | Default | Constraints |
| --- | --- | --- |
| `dataset-version` | `"upload-initial"` | Used to construct S3 path when `dataset-path-override` is empty. |
| `dataset-source` | `"lakefs"` | `"s3"` or `"lakefs"` |
| `dataset-path-override` | `""` | Full `s3://` URI. Overrides auto-constructed path. |
| `dataset-sample-size` | `""` | Integer as string. Empty = full dataset. |
| `dataset-seed` | `"42"` | Random seed for reproducible sampling. |

## Model

| Parameter | Default | Constraints |
| --- | --- | --- |
| `model-config` | `"yolov8n-pose.pt"` | Must be a valid YOLO pose variant (e.g., `yolov8{n,s,m,l,x}-pose.pt`) |
| `pretrained-weights` | `""` | S3 path to a `.pt` file. Empty = use default pretrained weights. |

## Training

| Parameter | Default | Constraints |
| --- | --- | --- |
| `epochs` | `"100"` | `> 0` |
| `batch-size` | `"16"` | `> 0`; reduce if GPU OOM |
| `image-size` | `"640"` | Multiple of 32 |
| `learning-rate` | `"0.01"` | `> 0` |
| `cos-lr` | `"true"` | `"true"` or `"false"` |
| `lrf` | `"0.01"` | `(0, 1]` |
| `optimizer` | `"SGD"` | `SGD`, `Adam`, `AdamW` |
| `momentum` | `"0.937"` | `[0, 1)` |
| `weight-decay` | `"0.0005"` | `>= 0` |
| `warmup-epochs` | `"3.0"` | `>= 0` |
| `warmup-momentum` | `"0.8"` | `[0, 1)` |
| `dropout` | `"0.0"` | `[0, 1)` |
| `label-smoothing` | `"0.0"` | `[0, 1)` |
| `nbs` | `"64"` | `> 0` |
| `freeze` | `""` | Number of backbone layers to freeze. Empty = none. |
| `amp` | `"true"` | `"true"` or `"false"` |
| `close-mosaic` | `"10"` | `>= 0`; 0 = never close |
| `training-seed` | `"0"` | Integer |
| `deterministic` | `"true"` | `"true"` or `"false"` |

## Loss Gains (YOLO-Pose)

| Parameter | Default | Constraints |
| --- | --- | --- |
| `pose` | `"12.0"` | `> 0`; keypoint regression loss gain |
| `kobj` | `"2.0"` | `> 0`; keypoint objectness loss gain |
| `box` | `"7.5"` | `> 0`; bounding box loss gain |
| `cls` | `"0.5"` | `> 0`; classification loss gain |
| `dfl` | `"1.5"` | `> 0`; distribution focal loss gain |

## Early Stopping

| Parameter | Default | Description |
| --- | --- | --- |
| `early-stopping-patience` | `"50"` | Stop if mAP50 does not improve for N consecutive epochs |

## Checkpointing

| Parameter | Default | Description |
| --- | --- | --- |
| `checkpointing-interval-epochs` | `"10"` | Save checkpoint every N epochs |
| `checkpoint-resume-from` | `""` | Empty = start fresh; `"auto"` = resume from latest checkpoint in experiment directory; S3 path = resume from that specific checkpoint |

## Augmentation

| Parameter | Default | Safe Range | Notes |
| --- | --- | --- | --- |
| `aug-hsv-h` | `"0.015"` | `0.0–0.1` | Hue jitter |
| `aug-hsv-s` | `"0.7"` | `0.0–1.0` | Saturation jitter |
| `aug-hsv-v` | `"0.4"` | `0.0–1.0` | Brightness jitter |
| `aug-degrees` | `"0.0"` | **Keep at 0** | Rotation disabled for pose |
| `aug-translate` | `"0.1"` | `0.0–0.3` | Translation fraction |
| `aug-scale` | `"0.5"` | `0.0–0.9` | Scale jitter |
| `aug-shear` | `"0.0"` | `0.0–10` | Shear degrees |
| `aug-perspective` | `"0.0"` | `0.0–0.001` | Projective warp |
| `aug-flipud` | `"0.0"` | **Keep at 0** | Disabled for spacecraft |
| `aug-fliplr` | `"0.0"` | **Keep at 0** | Disabled for spacecraft |
| `aug-mosaic` | `"1.0"` | `0.0–1.0` | Probability of mosaic |
| `aug-mixup` | `"0.0"` | `0.0–1.0` | Image blending probability |
| `aug-copy-paste` | `"0.0"` | `0.0–1.0` | Segment copy-paste |
| `aug-erasing` | `"0.4"` | `0.0–0.9` | Random patch erasure |
| `aug-bgr` | `"0.0"` | `0.0–1.0` | Channel order flip |

## Model Registration

| Parameter | Default | Description |
| --- | --- | --- |
| `registered-model-name` | `"{appName}-yolo"` | Model name in MLflow Registry. Default is derived from your app name. |
| `promote-to` | `""` | Auto-promote after registration. Empty = no promotion. `"Staging"` or `"Production"`. |

## Platform Parameters (Pre-Filled)

These are set by the KAOS platform. Do not change them.

| Parameter | Description |
| --- | --- |
| `kubepool-name` | Kubernetes cluster name |
| `mlflow-endpoint` | MLflow tracking server URL |
| `lakefs-endpoint` | lakeFS server URL |
| `lakefs-repo` | lakeFS repository name |
| `lakefs-branch` | lakeFS branch |
| `s3-artifacts-bucket` | S3 bucket for MLflow artifacts |
| `checkpoint-bucket` | S3 bucket for checkpoints |
| `checkpoint-prefix` | S3 key prefix for checkpoints |
| `aws-region` | AWS region |
