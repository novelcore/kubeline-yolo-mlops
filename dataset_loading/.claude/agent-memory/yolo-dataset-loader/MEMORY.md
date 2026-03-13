# YOLO Dataset Loader — Agent Memory

## Key Architectural Decisions

### Validation order (as of feature/01-initial-pipeline-implementation)
Three-phase validation pipeline:
1. **Pre-download S3 structure check** (`_check_s3_structure`) — runs before ANY download.
   Lists all keys once, categorises them (images/labels/other), verifies:
   - `data.yaml` key exists at prefix root
   - `images/train/` and `images/val/` have files (test is optional)
   - `labels/train/` and `labels/val/` have files
   - Every image key has a corresponding label key (stem matching)
   Returns `_S3KeyListing` so callers can reuse the key lists (no second list call).

2. **Inline label validation during download** (`_validate_label_file_inline`) —
   called for every `.txt` file under `labels/` immediately after `download_file`.
   - `data.yaml` is always downloaded FIRST (via `_download_single_key`) so
     `kpt_shape` and `names` are available for inline validation.
   - Full semantic check: empty file, token count, class ID range, bbox range,
     keypoint x/y range, visibility flags.
   - Applies to both `_run_full_download` and `_run_labels_only`.

3. **Post-download structural safety net** (`_validate`) — lightweight, no spot-check.
   Checks: data.yaml required keys, split directories exist, images non-empty,
   every image has a label file, label files non-empty.

### Error messages for test matching
- Missing images/labels on S3: `"S3 structure invalid: images/<split>/ has no files ..."`
- Missing image-label pairing on S3: `"have no corresponding label on S3"`
- No objects at all: `"No images or labels found at s3://..."`
- Empty label after download: `"Label file is empty (downloaded): ..."`
- data.yaml not on S3: `"data.yaml not found on S3 at s3://..."`

### `_S3KeyListing` internal class
- `image_keys`, `label_keys`, `other_keys`, `data_yaml_key`
- `data_yaml_key` is also included in `other_keys` for convenience

### Download flow
- `_download_single_key(bucket, key, prefix, output_path)` — single file download (data.yaml always first)
- `_download_with_inline_validation(...)` — concurrent batch download with per-label validation
  - Uses `ThreadPoolExecutor(_DOWNLOAD_WORKERS=8)` with batch-based submission
  - Files submitted in batches of 8; when error raised in batch N, batches N+1... are never submitted
  - `threading.Event` (cancel_event) lets in-flight sibling workers bail before downloading
  - Progress logged every `_PROGRESS_LOG_INTERVAL=500` files AND at 10% milestones
- `_download_and_validate_one(...)` — single-file worker called from thread pool
- `_spot_check_labels` is retained but only used by `_log_directory_integrity_report`

### Pre-download sampling flow
- `_sample_keys_by_split(image_keys, label_keys, sample_size, seed)` — samples S3 keys BEFORE any download
  - Groups image+label pairs by split, sorts pairs for reproducibility, applies `random.Random(seed).sample()`
  - Returns `(sampled_image_keys, sampled_label_keys)` — only these keys are downloaded
  - Called in both `_run_full_download` and `_run_labels_only` when `sample_size` is not None
  - Logs WARNING if a split has fewer pairs than `sample_size` (keeps all for that split)
- Old post-download `_sample()` method still exists but is NO LONGER called (pre-download sampling replaced it)
- `_sample_labels_only()` — also retained but NOT called from `_run_labels_only` anymore

### Critical: `_DOWNLOAD_WORKERS = 8` must stay <= 14
The fail-fast test (`test_inline_validation_catches_bad_label_immediately`) builds 5 train + 2 val = 14
non-data.yaml files. The bad label is at sorted index 7. With batch size 8, the bad label is in the first
batch (index 7 of 8). After the first batch fails, indices 8-13 are never submitted. `len(downloaded)` = 9 < 15.
If `_DOWNLOAD_WORKERS >= 14`, ALL files submit simultaneously and the test assertion fails.

## Project Structure

- **Step**: `dataset_loading/` — step 2 in the pipeline
- **Entry**: `app/services/dataset_loading.py` — `DatasetLoadingService`
- **Tests**: `tests/test_dataset_loading_service.py` — 48 tests, 6 classes
  - `TestHappyPath`, `TestSourceResolution`, `TestValidation`, `TestSampling`
  - `TestLabelsOnlyMode`, `TestDatasetStatsAndIntegrity`
  - `TestPreDownloadS3Check` (8 tests) — new
  - `TestInlineLabelValidation` (4 tests) — new

## Dataset Constants
- Default bucket: `temp-mlops`
- Default prefix: `datasets/speedplus_yolo/{version}/`
- kpt_shape: `[11, 3]` (11 spacecraft keypoints, 3D with visibility)
- Expected tokens per label line: 38 (1 class + 4 bbox + 11*3 keypoints)
- Required splits: `train`, `val` — `test` is optional per Ultralytics

## Mock Pattern
- `_make_mock_s3(dataset_root)` — full download mock, paginator uses all files
- `_make_mock_s3_with_listing(dataset_root)` — labels-only mock, same behaviour
- Both serve from a local `tmp_path` directory, no real S3 calls
- `get_paginator` is called once now (in `_check_s3_structure`), not in `_download`
