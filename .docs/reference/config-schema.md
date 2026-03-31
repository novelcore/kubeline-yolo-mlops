# Config Schema

Complete field reference for `pipeline_config.yaml`.

## experiment

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `name` | string | Yes | — | Experiment name in MLflow. Runs with the same name are grouped. |
| `description` | string | No | `""` | Free-text description. |
| `tags.*` | string | No | — | Arbitrary key-value tags attached to the experiment. |

## dataset

| Field | Type | Required | Default | Constraints |
| --- | --- | --- | --- | --- |
| `version` | string | Yes | — | Used to construct S3 path when `path_override` is null. |
| `source` | string | Yes | `"lakefs"` | `"s3"` or `"lakefs"` |
| `lakefs_repo` | string | Conditional | — | Required when `source: "lakefs"` |
| `lakefs_branch` | string | No | `"main"` | LakeFS branch name |
| `path_override` | string or null | No | `null` | Full `s3://` URI. Overrides auto-constructed path. |
| `sample_size` | integer or null | No | `null` | Use N random images. `null` = full dataset. |
| `seed` | integer | No | `42` | Random seed for reproducible sampling. |

## model

| Field | Type | Required | Default | Constraints |
| --- | --- | --- | --- | --- |
| `variant` | string | Yes | `"yolov8n-pose.pt"` | Must be a valid YOLO pose variant (e.g., `yolov8{n,s,m,l,x}-pose.pt`) |
| `pretrained_weights` | string or null | No | `null` | S3 path to a `.pt` file |

## training

| Field | Type | Default | Constraints |
| --- | --- | --- | --- |
| `epochs` | integer | `100` | `> 0` |
| `batch_size` | integer | `16` | `> 0`; reduce if GPU OOM |
| `image_size` | integer | `640` | Multiple of 32 |
| `learning_rate` | float | `0.01` | `> 0` |
| `cos_lr` | boolean | `true` | — |
| `lrf` | float | `0.01` | `(0, 1]` |
| `optimizer` | string | `"SGD"` | `SGD`, `Adam`, `AdamW` |
| `momentum` | float | `0.937` | `[0, 1)` |
| `weight_decay` | float | `0.0005` | `≥ 0` |
| `warmup_epochs` | float | `3.0` | `≥ 0` |
| `warmup_momentum` | float | `0.8` | `[0, 1)` |
| `dropout` | float | `0.0` | `[0, 1)` |
| `label_smoothing` | float | `0.0` | `[0, 1)` |
| `nbs` | integer | `64` | `> 0` |
| `freeze` | integer or null | `null` | Number of backbone layers to freeze |
| `amp` | boolean | `true` | — |
| `close_mosaic` | integer | `10` | `≥ 0`; 0 = never close |
| `seed` | integer | `0` | — |
| `deterministic` | boolean | `true` | — |
| `pose` | float | `12.0` | `> 0`; loss gain |
| `kobj` | float | `2.0` | `> 0`; loss gain |
| `box` | float | `7.5` | `> 0`; loss gain |
| `cls` | float | `0.5` | `> 0`; loss gain |
| `dfl` | float | `1.5` | `> 0`; loss gain |

## checkpointing

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `interval_epochs` | integer | `10` | Save checkpoint every N epochs |
| `storage_path` | string | — | S3 path for checkpoint storage |
| `resume_from` | null, `"auto"`, or string | `null` | `null` = start fresh; `"auto"` = latest checkpoint in `storage_path`; string = explicit S3 path |

## early_stopping

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `patience` | integer | `50` | Stop if mAP50 does not improve for N consecutive epochs |

## augmentation

| Field | Default | Safe Range | Notes |
| --- | --- | --- | --- |
| `hsv_h` | `0.015` | `0.0–0.1` | Hue jitter |
| `hsv_s` | `0.7` | `0.0–1.0` | Saturation jitter |
| `hsv_v` | `0.4` | `0.0–1.0` | Brightness jitter |
| `degrees` | `0.0` | **Keep at 0** | Rotation disabled for pose |
| `translate` | `0.1` | `0.0–0.3` | Translation fraction |
| `scale` | `0.5` | `0.0–0.9` | Scale jitter |
| `shear` | `0.0` | `0.0–10` | Shear degrees |
| `perspective` | `0.0` | `0.0–0.001` | Projective warp |
| `flipud` | `0.0` | **Keep at 0** | Disabled for spacecraft |
| `fliplr` | `0.0` | **Keep at 0** | Disabled for spacecraft |
| `mosaic` | `1.0` | `0.0–1.0` | Probability of mosaic |
| `mixup` | `0.0` | `0.0–1.0` | Image blending probability |
| `copy_paste` | `0.0` | `0.0–1.0` | Segment copy-paste |
| `erasing` | `0.4` | `0.0–0.9` | Random patch erasure |
| `bgr` | `0.0` | `0.0–1.0` | Channel order flip |

## registration

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `registered_model_name` | string or null | `null` | Model name in MLflow Registry. `null` = use `REGISTERED_MODEL_NAME` env var. |
| `promote_to` | null, `"Staging"`, or `"Production"` | `null` | Auto-promote after registration. |

## resources (informational)

Not validated by the pipeline. Informational only — used for cost tracking dashboards.

| Field | Type | Description |
| --- | --- | --- |
| `gpu_count` | integer | Number of GPUs expected |
| `gpu_type` | string | GPU type (e.g., `"A100-40GB"`) |
| `cpu_cores` | integer | CPU cores expected |
| `memory_gb` | integer | RAM expected (GB) |
