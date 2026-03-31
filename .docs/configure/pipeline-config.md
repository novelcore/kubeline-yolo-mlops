# pipeline_config.yaml

Complete field-by-field walkthrough of the pipeline configuration file.

!!! tip "Start from the example"
    Copy `pipeline_config.example.yaml` from the repository and edit from there.
    All fields have comments explaining their purpose.

---

## experiment

Groups all runs from this configuration into a named experiment in MLflow.

```yaml
experiment:
  name: "spacecraft-pose-v1-yolov8n"     # Required
  description: "Baseline run on SPEED+"   # Optional — shown in MLflow UI
  tags:
    project: "infinite-orbits"            # Optional — searchable in MLflow
    phase: "1"
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `name` | string | Yes | Experiment name. Runs with the same name are grouped together in MLflow. Use a descriptive name — you cannot rename it later. |
| `description` | string | No | Free-text description shown in the MLflow experiment view. |
| `tags.*` | string | No | Any key-value pairs you want attached to the experiment for filtering. |

---

## dataset

Tells the pipeline where to find your training data.

```yaml
dataset:
  version: "v1"
  source: "lakefs"               # "s3" or "lakefs"
  lakefs_repo: "io-data"         # Required when source=lakefs
  lakefs_branch: "main"          # Required when source=lakefs
  path_override: null            # null = auto-construct path; or a full s3:// URI
  sample_size: null              # null = full dataset; integer = random subset
  seed: 42                       # Seed for reproducible sampling
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `version` | string | — | Dataset version tag. Used to construct the S3 path automatically when `path_override` is null. |
| `source` | `"s3"` or `"lakefs"` | `"lakefs"` | Storage backend. Use `"lakefs"` when your dataset is versioned in LakeFS; use `"s3"` for a direct S3 path. |
| `lakefs_repo` | string | — | LakeFS repository name. Required when `source: "lakefs"`. |
| `lakefs_branch` | string | `"main"` | LakeFS branch to read from. |
| `path_override` | string or null | `null` | Override the auto-constructed path with a full `s3://` URI. Useful for custom dataset locations. |
| `sample_size` | integer or null | `null` | Use a random subset of the dataset. Set to an integer (e.g., `500`) for faster iteration during development. Set to `null` for a full training run. |
| `seed` | integer | `42` | Controls which samples are selected when `sample_size` is set. Use the same seed to get the same subset across runs. |

!!! info "See also"
    [Dataset Format](datasets.md) — What format the dataset must be in for the pipeline to accept it.

---

## model

Selects the YOLO architecture to train.

```yaml
model:
  variant: "yolov8n-pose.pt"    # Which YOLO model to use
  pretrained_weights: null       # Optional: custom starting weights
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `variant` | string | `"yolov8n-pose.pt"` | YOLO pose model variant. See options below. |
| `pretrained_weights` | string or null | `null` | S3 path to a custom `.pt` file to use as starting weights instead of the default ImageNet-pretrained weights. Useful for fine-tuning a previously trained model. |

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

## training

All hyperparameters passed to the YOLO training process.

### Core Schedule

```yaml
training:
  epochs: 100
  batch_size: 16
  image_size: 640
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `epochs` | integer | `100` | Total training epochs. More epochs = more training time. Use `10–20` for fast experiments, `100–300` for full runs. |
| `batch_size` | integer | `16` | Number of images per gradient update. Larger batches use more GPU memory but train faster per epoch. Reduce if you hit out-of-memory errors. |
| `image_size` | integer | `640` | Input image resolution (must be a multiple of 32). Larger resolution improves accuracy for small objects but increases memory and compute cost. |

### Learning Rate

```yaml
  learning_rate: 0.01
  cos_lr: true
  lrf: 0.01
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `learning_rate` | float | `0.01` | Initial learning rate. Controls how large each gradient update step is. If training is unstable (loss spikes), lower this value. |
| `cos_lr` | boolean | `true` | Use cosine learning rate decay. Gradually reduces the learning rate following a cosine curve — recommended for most runs. |
| `lrf` | float | `0.01` | Final learning rate multiplier. The LR at the last epoch = `learning_rate × lrf`. Keep this small (e.g., `0.01`). |

### Optimizer

```yaml
  optimizer: "SGD"
  momentum: 0.937
  weight_decay: 0.0005
```

| Field | Options | Default | Description |
| --- | --- | --- | --- |
| `optimizer` | `SGD`, `Adam`, `AdamW` | `SGD` | Optimization algorithm. `SGD` is the standard for YOLO training. `Adam`/`AdamW` can converge faster but may need a lower learning rate. |
| `momentum` | float | `0.937` | SGD momentum (or Adam beta1). Controls how much past gradients influence the current update. Rarely needs changing. |
| `weight_decay` | float | `0.0005` | L2 regularization. Penalizes large weights to reduce overfitting. |

### Warmup

```yaml
  warmup_epochs: 3.0
  warmup_momentum: 0.8
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `warmup_epochs` | float | `3.0` | Number of epochs to linearly ramp up the learning rate from near-zero. Prevents unstable early training. |
| `warmup_momentum` | float | `0.8` | Initial momentum during warmup, ramped up to `momentum` by the end of warmup. |

### Pose-Estimation Loss Gains

These are specific to pose estimation and control how much weight each loss component receives.

```yaml
  pose: 12.0
  kobj: 2.0
  box: 7.5
  cls: 0.5
  dfl: 1.5
```

| Field | Default | Description |
| --- | --- | --- |
| `pose` | `12.0` | Keypoint regression loss gain — the primary quality lever for pose accuracy. Increase to prioritize pose accuracy over bounding box accuracy. |
| `kobj` | `2.0` | Keypoint objectness loss gain — controls confidence of predicted keypoints. |
| `box` | `7.5` | Bounding box regression loss gain. |
| `cls` | `0.5` | Classification loss gain. For single-class datasets (spacecraft only), this has low impact. |
| `dfl` | `1.5` | Distribution Focal Loss gain for bounding box distribution. |

!!! warning "Spacecraft-specific defaults"
    These defaults are tuned for spacecraft pose estimation. Do not change them without a clear reason — even small adjustments to `pose` or `kobj` can significantly affect keypoint accuracy.

### Other Training Options

```yaml
  dropout: 0.0
  label_smoothing: 0.0
  nbs: 64
  freeze: null
  amp: true
  close_mosaic: 10
  seed: 0
  deterministic: true
```

| Field | Default | Description |
| --- | --- | --- |
| `dropout` | `0.0` | Dropout rate in the classifier head. Start with 0; add if you see strong overfitting. |
| `label_smoothing` | `0.0` | Adds small uncertainty to classification labels. Rarely needed for single-class datasets. |
| `nbs` | `64` | Nominal batch size for learning rate scaling. If your `batch_size` differs from `nbs`, Ultralytics adjusts the effective LR automatically. |
| `freeze` | `null` | Freeze the first N backbone layers. Useful when fine-tuning from `pretrained_weights` — freeze early layers to preserve learned features. |
| `amp` | `true` | Automatic Mixed Precision (FP16). Halves GPU memory usage with minimal accuracy impact. Keep `true` unless debugging numerical issues. |
| `close_mosaic` | `10` | Disables mosaic augmentation for the last N epochs. Helps the model converge on clean images before validation. |
| `seed` | `0` | Global random seed for reproducibility. |
| `deterministic` | `true` | Enables cuDNN deterministic mode. Guarantees identical results across runs with the same seed. |

---

## checkpointing

Controls how often intermediate model checkpoints are saved to S3.

```yaml
checkpointing:
  interval_epochs: 10
  storage_path: "s3://your-bucket/checkpoints"
  resume_from: null
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `interval_epochs` | integer | `10` | Save a checkpoint every N epochs. Smaller values = more storage, more resume points. |
| `storage_path` | string | — | S3 path where checkpoints are saved. Ask your platform administrator for the correct bucket. |
| `resume_from` | `null`, `"auto"`, or S3 path | `null` | Resume training from a checkpoint. `null` = start fresh. `"auto"` = find the latest checkpoint in `storage_path`. Or provide an explicit `s3://` path. |

!!! tip "Use `resume_from: auto` after interruptions"
    If a training run is interrupted (node eviction, timeout, OOM), set `resume_from: "auto"` to continue from the last checkpoint instead of restarting from scratch.

---

## early_stopping

Stops training automatically if the model stops improving.

```yaml
early_stopping:
  patience: 50
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `patience` | integer | `50` | Stop training if validation mAP50 does not improve for this many consecutive epochs. Saves compute on runs that have already converged. |

---

## augmentation

Data augmentation applied to training images. The defaults are tuned for spacecraft imagery.

```yaml
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  translate: 0.1
  scale: 0.5
  shear: 0.0
  perspective: 0.0
  flipud: 0.0
  fliplr: 0.0
  mosaic: 1.0
  mixup: 0.0
  copy_paste: 0.0
  erasing: 0.4
  bgr: 0.0
```

!!! warning "Do not enable rotation or flipping for spacecraft"
    `degrees: 0.0`, `flipud: 0.0`, and `fliplr: 0.0` are intentionally disabled.
    Spacecraft have orientation-dependent keypoint geometry — random rotation or flipping
    would destroy the pose labels and degrade accuracy.

| Field | Default | Safe Range | Description |
| --- | --- | --- | --- |
| `hsv_h` | `0.015` | `0.0–0.1` | Hue jitter. Simulates different lighting conditions. |
| `hsv_s` | `0.7` | `0.0–1.0` | Saturation jitter. |
| `hsv_v` | `0.4` | `0.0–1.0` | Value (brightness) jitter. |
| `degrees` | `0.0` | **Keep at 0** | Random rotation. Must be 0 for pose tasks. |
| `translate` | `0.1` | `0.0–0.3` | Random translation as a fraction of image size. |
| `scale` | `0.5` | `0.0–0.9` | Random scaling. |
| `shear` | `0.0` | `0.0–10` | Random shear. |
| `perspective` | `0.0` | `0.0–0.001` | Projective warp. Keep near 0 for pose. |
| `flipud` | `0.0` | **Keep at 0** | Vertical flip probability. Disabled for spacecraft. |
| `fliplr` | `0.0` | **Keep at 0** | Horizontal flip probability. Disabled for spacecraft. |
| `mosaic` | `1.0` | `0.0–1.0` | Probability of mosaic augmentation (combines 4 images). |
| `erasing` | `0.4` | `0.0–0.9` | Random rectangular patch erasing probability. |

---

## registration

Controls how the trained model is saved to the MLflow Model Registry after training.

```yaml
registration:
  registered_model_name: null
  promote_to: null
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `registered_model_name` | string or null | `null` | Name of the model in the MLflow registry. `null` uses the value of the `REGISTERED_MODEL_NAME` environment variable (set by the platform, default: `spacecraft-pose-yolo`). Set a custom name to keep models from different experiments separate. |
| `promote_to` | `null`, `"Staging"`, or `"Production"` | `null` | Automatically promote the registered model to a stage after registration. `null` = no promotion (leaves it at `None` stage). |

!!! info "See also"
    [Promoting a Model](../results/promotion.md) — How to move models between stages after registration.
