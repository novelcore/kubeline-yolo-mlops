# Parameter Reference

Complete parameter-by-parameter walkthrough of the Argo workflow submission parameters.

!!! tip "Defaults are sensible"
    Every parameter has a default value. When you submit a workflow from the Argo UI, you only need to override the parameters you want to change.

---

## Experiment

Describes this run in MLflow.

| Parameter | Default | Description |
| --- | --- | --- |
| `experiment-name` | `""` | Name for the MLflow experiment. All runs submitted with the same experiment name appear together in the MLflow UI, making it easy to compare them. **Required** — set this when submitting a workflow. |
| `experiment-description` | `""` | Free-text description shown in the MLflow experiment view. |

---

## Dataset

Controls where to load training data from and how much of it to use.

| Parameter | Default | Description |
| --- | --- | --- |
| `dataset-version` | `"upload-initial"` | Dataset version tag. Used to construct the storage path automatically. |
| `dataset-source` | `"lakefs"` | Storage backend. Use `"lakefs"` when your dataset is versioned in LakeFS; use `"s3"` for a direct S3 path. |
| `dataset-path-override` | `""` | Override the auto-constructed path with a full `s3://` URI. When set, the version and source parameters are ignored. |
| `dataset-sample-size` | `""` | Use a random subset of the dataset. Set to an integer (e.g., `200`) for faster iteration during development. Leave empty for the full dataset. |
| `dataset-seed` | `"42"` | Controls which samples are selected when `dataset-sample-size` is set. Use the same seed to get the same subset across runs. |

!!! info "See also"
    [Dataset Format](datasets.md) — What format the dataset must be in for the pipeline to accept it.

**Example — quick iteration run:**

Override these parameters when submitting:

| Parameter | Value |
| --- | --- |
| `dataset-version` | `v2` |
| `dataset-sample-size` | `200` |

Everything else stays at default. This trains on 200 random images from the `v2` dataset.

---

## Model

Selects the YOLO architecture to train.

| Parameter | Default | Description |
| --- | --- | --- |
| `model-config` | `"yolov8n-pose.pt"` | YOLO pose model variant. See options below. |
| `pretrained-weights` | `""` | S3 path to a custom `.pt` file to use as starting weights instead of the default ImageNet-pretrained weights. Useful for fine-tuning a previously trained model. |

**Available variants** (ordered by size / accuracy):

| Variant | Parameters | Speed | Use When |
| --- | --- | --- | --- |
| `yolov8n-pose.pt` | ~3M | Fastest | Prototyping, limited GPU |
| `yolov8s-pose.pt` | ~11M | Fast | Balanced baseline |
| `yolov8m-pose.pt` | ~25M | Moderate | Better accuracy |
| `yolov8l-pose.pt` | ~43M | Slow | High accuracy |
| `yolov8x-pose.pt` | ~68M | Slowest | Maximum accuracy |
| `yolov11n-pose.pt` | — | Fastest | Latest architecture, nano |
| `yolov11x-pose.pt` | — | Slowest | Latest architecture, maximum |

---

## Training

All hyperparameters passed to the YOLO training process.

### Core Schedule

| Parameter | Default | Description |
| --- | --- | --- |
| `epochs` | `"100"` | Total training epochs. More epochs = more training time. Use `10–20` for fast experiments, `100–300` for full runs. |
| `batch-size` | `"16"` | Number of images per gradient update. Larger batches use more GPU memory but train faster per epoch. Reduce if you hit out-of-memory errors. |
| `image-size` | `"640"` | Input image resolution (must be a multiple of 32). Larger resolution improves accuracy for small objects but increases memory and compute cost. |

### Learning Rate

| Parameter | Default | Description |
| --- | --- | --- |
| `learning-rate` | `"0.01"` | Initial learning rate. Controls how large each gradient update step is. If training is unstable (loss spikes), lower this value. |
| `cos-lr` | `"true"` | Use cosine learning rate decay. Gradually reduces the learning rate following a cosine curve — recommended for most runs. |
| `lrf` | `"0.01"` | Final learning rate multiplier. The LR at the last epoch = `learning-rate × lrf`. Keep this small (e.g., `0.01`). |

### Optimizer

| Parameter | Options | Default | Description |
| --- | --- | --- | --- |
| `optimizer` | `SGD`, `Adam`, `AdamW` | `"SGD"` | Optimization algorithm. `SGD` is the standard for YOLO training. `Adam`/`AdamW` can converge faster but may need a lower learning rate. |
| `momentum` | float | `"0.937"` | SGD momentum (or Adam beta1). Controls how much past gradients influence the current update. Rarely needs changing. |
| `weight-decay` | float | `"0.0005"` | L2 regularization. Penalizes large weights to reduce overfitting. |

### Warmup

| Parameter | Default | Description |
| --- | --- | --- |
| `warmup-epochs` | `"3.0"` | Number of epochs to linearly ramp up the learning rate from near-zero. Prevents unstable early training. |
| `warmup-momentum` | `"0.8"` | Initial momentum during warmup, ramped up to `momentum` by the end of warmup. |

### Pose-Estimation Loss Gains

These are specific to pose estimation and control how much weight each loss component receives.

| Parameter | Default | Description |
| --- | --- | --- |
| `pose` | `"12.0"` | Keypoint regression loss gain — the primary quality lever for pose accuracy. Increase to prioritize pose accuracy over bounding box accuracy. |
| `kobj` | `"2.0"` | Keypoint objectness loss gain — controls confidence of predicted keypoints. |
| `box` | `"7.5"` | Bounding box regression loss gain. |
| `cls` | `"0.5"` | Classification loss gain. For single-class datasets (spacecraft only), this has low impact. |
| `dfl` | `"1.5"` | Distribution Focal Loss gain for bounding box distribution. |

!!! warning "Spacecraft-specific defaults"
    These defaults are tuned for spacecraft pose estimation. Do not change them without a clear reason — even small adjustments to `pose` or `kobj` can significantly affect keypoint accuracy.

### Other Training Options

| Parameter | Default | Description |
| --- | --- | --- |
| `dropout` | `"0.0"` | Dropout rate in the classifier head. Start with 0; add if you see strong overfitting. |
| `label-smoothing` | `"0.0"` | Adds small uncertainty to classification labels. Rarely needed for single-class datasets. |
| `nbs` | `"64"` | Nominal batch size for learning rate scaling. If your `batch-size` differs from `nbs`, Ultralytics adjusts the effective LR automatically. |
| `freeze` | `""` | Freeze the first N backbone layers. Useful when fine-tuning from `pretrained-weights` — freeze early layers to preserve learned features. |
| `amp` | `"true"` | Automatic Mixed Precision (FP16). Halves GPU memory usage with minimal accuracy impact. Keep `true` unless debugging numerical issues. |
| `close-mosaic` | `"10"` | Disables mosaic augmentation for the last N epochs. Helps the model converge on clean images before validation. |
| `training-seed` | `"0"` | Global random seed for reproducibility. |
| `deterministic` | `"true"` | Enables cuDNN deterministic mode. Guarantees identical results across runs with the same seed. |

---

## Early Stopping

Stops training automatically if the model stops improving.

| Parameter | Default | Description |
| --- | --- | --- |
| `early-stopping-patience` | `"50"` | Stop training if validation mAP50 does not improve for this many consecutive epochs. Saves compute on runs that have already converged. |

---

## Checkpointing

Controls how often intermediate model checkpoints are saved.

| Parameter | Default | Description |
| --- | --- | --- |
| `checkpointing-interval-epochs` | `"10"` | Save a checkpoint every N epochs. Smaller values = more storage, more resume points. |
| `checkpoint-resume-from` | `""` | Resume training from a checkpoint. Accepts three modes — see below. |

**`checkpoint-resume-from` modes:**

| Value | Behaviour |
| --- | --- |
| `""` (empty, default) | Start training from scratch. |
| `"auto"` | Automatically scan the experiment's checkpoint directory for the latest checkpoint and resume from it. |
| A specific S3 path (e.g., `s3://bucket/checkpoints/exp/last.pt`) | Resume from that exact checkpoint. |

!!! tip "Resuming after an interruption"
    If a training run is interrupted (node eviction, timeout, OOM), set `checkpoint-resume-from` to `"auto"` to automatically pick up from the latest checkpoint, or provide the S3 path of a specific checkpoint to continue from.

---

## Augmentation

Data augmentation applied to training images. The defaults are tuned for spacecraft imagery.

!!! warning "Do not enable rotation or flipping for spacecraft"
    `aug-degrees`, `aug-flipud`, and `aug-fliplr` are intentionally set to `0.0`.
    Spacecraft have orientation-dependent keypoint geometry — random rotation or flipping
    would destroy the pose labels and degrade accuracy.

| Parameter | Default | Safe Range | Description |
| --- | --- | --- | --- |
| `aug-hsv-h` | `"0.015"` | `0.0–0.1` | Hue jitter. Simulates different lighting conditions. |
| `aug-hsv-s` | `"0.7"` | `0.0–1.0` | Saturation jitter. |
| `aug-hsv-v` | `"0.4"` | `0.0–1.0` | Value (brightness) jitter. |
| `aug-degrees` | `"0.0"` | **Keep at 0** | Random rotation. Must be 0 for pose tasks. |
| `aug-translate` | `"0.1"` | `0.0–0.3` | Random translation as a fraction of image size. |
| `aug-scale` | `"0.5"` | `0.0–0.9` | Random scaling. |
| `aug-shear` | `"0.0"` | `0.0–10` | Random shear. |
| `aug-perspective` | `"0.0"` | `0.0–0.001` | Projective warp. Keep near 0 for pose. |
| `aug-flipud` | `"0.0"` | **Keep at 0** | Vertical flip probability. Disabled for spacecraft. |
| `aug-fliplr` | `"0.0"` | **Keep at 0** | Horizontal flip probability. Disabled for spacecraft. |
| `aug-mosaic` | `"1.0"` | `0.0–1.0` | Probability of mosaic augmentation (combines 4 images). |
| `aug-mixup` | `"0.0"` | `0.0–1.0` | Image blending probability. |
| `aug-copy-paste` | `"0.0"` | `0.0–1.0` | Segment copy-paste probability. |
| `aug-erasing` | `"0.4"` | `0.0–0.9` | Random rectangular patch erasing probability. |
| `aug-bgr` | `"0.0"` | `0.0–1.0` | Channel order flip probability. |

---

## Model Registration

Controls how the trained model is saved to the MLflow Model Registry after training.

| Parameter | Default | Description |
| --- | --- | --- |
| `registered-model-name` | `"{appName}-yolo"` | Name of the model in the MLflow registry. The default is derived from your app name. Set a custom name to keep models from different experiments separate. |
| `promote-to` | `""` | Automatically promote the registered model to a stage after registration. Leave empty for no promotion, or set to `"Staging"` or `"Production"`. |

!!! info "See also"
    [Promoting a Model](../results/promotion.md) — How to move models between stages after registration.

---

## Platform Parameters (Pre-Filled)

These parameters are injected by the KAOS platform when your WorkflowTemplate is created. They configure the infrastructure endpoints your pipeline connects to. **You should not change these.**

| Parameter | Description |
| --- | --- |
| `kubepool-name` | Name of the Kubernetes cluster running the pipeline |
| `mlflow-endpoint` | URL of the MLflow tracking server |
| `lakefs-endpoint` | URL of the lakeFS server |
| `lakefs-repo` | lakeFS repository for your project's data |
| `lakefs-branch` | lakeFS branch to read from |
| `s3-artifacts-bucket` | S3 bucket for MLflow artifacts |
| `checkpoint-bucket` | S3 bucket for training checkpoints |
| `checkpoint-prefix` | S3 key prefix for checkpoints |
| `aws-region` | AWS region for all S3 operations |
