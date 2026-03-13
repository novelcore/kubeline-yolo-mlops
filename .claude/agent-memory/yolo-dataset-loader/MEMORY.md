# YOLO Dataset Loader — Agent Memory

## Dataset Format
- YOLO pose estimation for spacecraft (SpeedPlus dataset).
- Reference dataset at `/home/thanos/Documents/speedplus_yolo/`.
- 3 splits: `train`, `val`, `test`.
- Layout: `images/{split}/*.jpg`, `labels/{split}/*.txt`, `data.yaml` at root.
- Label line: 38 tokens = `class cx cy w h kp1x kp1y kp1vis ... kp11x kp11y kp11vis`.
- 11 keypoints × 3 values (x, y, visibility ∈ {0,1,2}), all bbox coords normalized [0,1].
- Class 0 = "spacecraft", kpt_shape = [11, 3], flip_idx = [].

## Storage Convention
- S3 bucket: `io-mlops`, default prefix: `datasets/speedplus_yolo/{version}/`.
- LakeFS uses S3-compatible API via boto3 with `endpoint_url=LAKEFS_ENDPOINT`.
- Always download fully before validating (no streaming mode).

## Implementation (Step 2: dataset_loading)
- Service file: `dataset_loading/app/services/dataset_loading.py`
- `DatasetLoadingService(s3_client, max_retries)` — dependency-injected boto3 client.
- `run(YoloDatasetParams) -> YoloDatasetStats`
- Full-download pipeline: `_resolve_source` → `_download` → `_write_data_yaml` → `_validate` → `_sample` (optional) → `_count_splits` + `_count_label_splits` → `_write_stats` → `_log_dataset_stats` → `_log_directory_integrity_report`
- Labels-only pipeline same but uses `_list_s3_keys` → `_download_selective` → `_validate_labels_only` → `_build_manifest` + `_write_manifest` → `_sample_labels_only` (optional) → `_count_label_splits` → `_write_stats` → `_log_dataset_stats` → `_log_directory_integrity_report`
- Output artifact: `{output_dir}/dataset_stats.json`.
- SPLITS constant exported for use in tests: `("train", "val", "test")`.

## YoloDatasetStats fields (dataset_stats.json)
- `train_images`, `val_images`, `test_images` — image counts per split (label counts in labels-only mode).
- `train_labels`, `val_labels`, `test_labels` — label file counts per split (same as *_images in labels-only mode).
- `sampled`, `sample_size`, `seed`, `version`, `source`.

## Logging methods
- `_log_dataset_stats(stats)` — emits an aligned table of images and labels per split plus total at INFO.
- `_log_directory_integrity_report(output_path, mode)` — walks the tree and logs [OK]/[SKIP]/[FAIL] per item, then a PASS/FAIL verdict. Never raises. mode="full" treats missing images/ as FAIL; mode="labels-only" treats them as SKIP.

## Type Stubs Required
- Dev deps must include `types-pyyaml` and `boto3-stubs[s3]`.
- boto3/botocore need `ignore_missing_imports = true` in `[[tool.mypy.overrides]]`.
- See `pyproject.toml` for the exact configuration pattern.

## Validation Rules
- All 3 splits must exist as directories under `images/`.
- Each split needs >= 1 image.
- Every image stem must have a matching `.txt` label file in `labels/{split}/`.
- Every label file must be non-empty.
- Spot-check: first 10 label files per split must have exactly 38 tokens per line.
- `data.yaml` must have keys: `path`, `train`, `val`, `test`, `kpt_shape`, `names`.

## Sampling
- `sample_size` + `seed` passed via CLI; seed goes through `random.Random(seed)`.
- Files deleted in-place after download+validation.
- If split has <= sample_size images: keep all, log WARNING.

## CLI Invocation (from orchestrate.sh)
```
dataset-loading run \
  --version v1 \
  --source s3 \
  --output-dir /artifacts/dataset \
  --path-override "" \
  --sample-size <N or omit> \
  --seed 42
```

## Logging Pattern (Steps 1 and 2)
- Shared `ColorFormatter` + `setup_logging()` pattern used across all steps.
- Each step has its own `app/logger.py` (copied verbatim from `config_validation/app/logger.py`).
- `Manager.__init__` calls `setup_logging(level=self._config.log_level)` before `logging.getLogger(__name__)`.
- Service `__init__` uses `logging.getLogger(__name__)` (no setup_logging call in service).
- Per-file downloads logged at DEBUG (`"Downloaded: {key}"`); ~5% progress at INFO; summary at INFO with file count, MB, and elapsed seconds.
- Manager.run() wraps service.run() in try/except and logs errors at ERROR before re-raising.

## Known Pre-existing Test Failure
- `tests/test_config.py::test_config_default_values` fails when `AWS_DEFAULT_REGION` is set in the host shell (it reads the real env var instead of the `None` default). Not a regression — unrelated to logging changes.

## Links
- Detailed patterns: `patterns.md`
