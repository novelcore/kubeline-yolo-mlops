# Configuration Patterns and Decisions

## The training.scheduler Silent-Drop Bug (Fixed)

### Original problem
`TrainingConfig` had `model_config = ConfigDict(extra="ignore")`. The YAML contained:
```yaml
training:
  scheduler:
    cos_lr: true
    lrf: 0.01
```
These values were parsed, silently discarded by Pydantic, and never validated or passed
downstream. The orchestrator's `CFG_<section>_<key>` convention cannot handle nested
objects anyway, so these values were unreachable from any step.

### Fix applied
- Changed `TrainingConfig` to `extra="forbid"`.
- Promoted `cos_lr` (bool, default True) and `lrf` (float, (0, 1], default 0.01) to
  first-class fields on `TrainingConfig`.
- Removed the `scheduler:` block from `pipeline_config.example.yaml`.
- Updated `test_scheduler_sub_object_is_ignored` to
  `test_scheduler_sub_object_is_now_rejected` — asserts `ValidationError` is raised.
- Added `test_cos_lr_and_lrf_as_first_class_fields` to cover the happy path.

## Parameter Inclusion/Exclusion Rationale (Ultralytics YOLO)

### Included in TrainingConfig
| Parameter | Rationale |
|-----------|-----------|
| cos_lr | Promoted from scheduler; directly controls LR schedule shape |
| lrf | Promoted from scheduler; final LR ratio, key for convergence |
| momentum | Critical for SGD; also beta1 for Adam/AdamW |
| nbs | Controls gradient accumulation scaling; non-obvious effect on effective LR |
| amp | Halves GPU memory; essential on 40GB A100s with large batches |
| label_smoothing | Regularization; commonly tuned in pose tasks |
| dropout | Regularization; relevant for larger model variants |
| freeze | Enables backbone freezing for fine-tuning workflows |
| close_mosaic | Disabling mosaic in final epochs improves convergence; commonly tuned |
| pose | PRIMARY pose loss lever for spacecraft keypoint accuracy |
| kobj | Keypoint objectness; paired with pose gain |
| box | Bounding-box regression loss; affects detection anchor quality |
| cls | Classification loss; less critical for single-class spacecraft |
| dfl | Distribution Focal Loss; affects box prediction distribution |
| seed | Reproducibility; distinct scope from dataset.seed |
| deterministic | Reproducibility; enables cuDNN determinism |

### Excluded from TrainingConfig
| Parameter | Reason |
|-----------|--------|
| val | Always True in pipeline context; not a tuning parameter |
| fraction | Duplicates dataset.sample_size; two mechanisms for same thing = conflict risk |
| profile | Dev/debug tool; belongs in step-level config.py env vars |
| save | Pipeline infra concern; not a hyperparameter |
| save_period | Duplicates checkpointing.interval_epochs exactly |
| exist_ok | Pipeline infra concern |
| verbose | Pipeline infra concern; belongs in step Config |
| plots | Pipeline infra concern |
| multi_scale | Low priority; spacecraft use fixed scale |
| overlap_mask | Segmentation-only; irrelevant for pose |
| mask_ratio | Segmentation-only |
| pretrained | Covered by model.pretrained_weights |
| rect | Not relevant for spacecraft pose |
| single_cls | Not relevant for spacecraft pose |

### Included in AugmentationConfig
| Parameter | Rationale |
|-----------|-----------|
| shear | Geometric; useful for pose; complement to degrees/translate/scale |
| perspective | Projective transform; relevant for spacecraft imagery; validated [0, 0.001] |
| copy_paste | Segment copy-paste; included for completeness even if rarely tuned |
| erasing | Strong regularization augmentation; Ultralytics default 0.4 |
| bgr | BGR channel flip; included for completeness; default 0.0 for spacecraft |

### Excluded from AugmentationConfig
| Parameter | Reason |
|-----------|--------|
| auto_augment | String policy; conflicts with manual per-parameter augmentation config |
| crop_fraction | Classification-only; irrelevant for pose detection |

## Cross-Field Validators in TrainingConfig

Two blocking cross-field validations in `model_validator(mode="after")`:
1. `close_mosaic >= epochs` — mosaic would be disabled for the entire run; almost always a bug
2. `warmup_epochs >= epochs` — warmup longer than training is always a misconfiguration

These are BLOCKING (raise ValueError) not warnings, because both make training dynamics
either undefined or certainly wrong.
