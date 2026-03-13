# Pipeline Config Validator — Agent Memory

## Key Files
- Pipeline YAML example: `/home/thanos/Documents/kubeline-yolo-mlops/pipeline_config.example.yaml`
- Pydantic models: `/home/thanos/Documents/kubeline-yolo-mlops/config_validation/app/models/pipeline_config.py`
- Tests: `/home/thanos/Documents/kubeline-yolo-mlops/config_validation/tests/test_pipeline_config.py`

## Schema Decisions (confirmed in code)

### TrainingConfig uses extra="forbid"
Changed from extra="ignore" in the initial implementation. The old schema silently dropped
`training.scheduler` sub-objects. `cos_lr` and `lrf` are now first-class fields directly
on `TrainingConfig`. Any YAML with a `training.scheduler` key will now fail validation.
See: `patterns.md` for full rationale.

### Model variants
Pattern: `^yolov(8|9|10|11)[nsmlx]-pose\.pt$`
Valid versions: 8, 9, 10, 11. Valid sizes: n, s, m, l, x.

### Checkpointing resume_from
Allowed values: null | "auto" | s3:// path. lakefs:// paths are NOT allowed for resume_from
(only storage_path supports lakefs://).

### dataset.seed vs training.seed
Two separate seeds with different scopes:
- `dataset.seed`: controls reproducible dataset sampling in `dataset_loading` step
- `training.seed`: global Ultralytics RNG seed passed to `model.train(seed=...)`

## Validation Ranges (confirmed)

### AugmentationConfig
- `perspective`: [0.0, 0.001] — Ultralytics enforces this upper bound internally; beyond
  0.001 causes extreme keypoint displacement invalidating pose labels. We enforce it in Pydantic.
- `erasing`: [0.0, 0.9] — upper bound 0.9 because erasing=1.0 blanks the entire image.

### TrainingConfig
- `lrf`: (0.0, 1.0] — ratio of final LR to initial LR; 0 would zero the LR permanently.
- `momentum`: [0.0, 1.0) — strictly less than 1.0 (1.0 = no gradient update, non-convergent).
- `label_smoothing`: [0.0, 1.0) — 1.0 makes all labels uniform (meaningless).
- `dropout`: [0.0, 1.0) — 1.0 drops all activations (training would not converge).
- `close_mosaic`: must be < epochs (cross-field validated in model_validator).
- `warmup_epochs`: must be < epochs (cross-field validated in model_validator).

## Parameters Deliberately Excluded
See `patterns.md` for full exclusion rationale. Key exclusions:
- `val`, `save`, `exist_ok`, `verbose`, `plots` — pipeline/infra concerns, not hyperparameters
- `save_period` — duplicates checkpointing.interval_epochs
- `fraction` — duplicates dataset.sample_size
- `overlap_mask`, `mask_ratio` — segmentation-only, irrelevant for pose
- `pretrained` — covered by model.pretrained_weights
- `auto_augment` — conflicts with individual augmentation parameter tuning
- `crop_fraction` — classification-only

## Spacecraft-Specific Defaults
- `fliplr: 0.0` — satellites have no left-right symmetry in canonical pose
- `flipud: 0.0` — orbital geometry has a defined "up"
- `degrees: 0.0` — pose labels are rotation-sensitive
- `perspective: 0.0` — keep near 0; >0.001 invalidates keypoint geometry
- `bgr: 0.0` — space imagery doesn't benefit from BGR channel flip
