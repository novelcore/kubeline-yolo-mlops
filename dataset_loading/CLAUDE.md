# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Step Does

Second step in the MLOps pipeline (config_validation -> `dataset_loading` -> model_training -> model_registration). Downloads a YOLO pose dataset from S3 or LakeFS, validates its structure (splits, image-label pairing, label token format), generates/updates `data.yaml` with runtime paths, optionally sub-samples to a fixed size, and writes a `dataset_stats.json` artifact for downstream steps.

## Commands

```bash
poetry install                          # install dependencies
poetry run dataset-loading run \        # run locally
  --version v1 \
  --source s3 \
  --output-dir /tmp/dataset \
  --seed 42
  # Optional:
  # --path-override 's3://bucket/custom/prefix/'
  # --sample-size 100

poetry run pytest                                       # all tests
poetry run pytest tests/test_dataset_loading_service.py  # single file
poetry run pytest -k "test_sampling_is_reproducible"     # single test by name

poetry run black app/ tests/                            # format
poetry run isort app/ tests/                            # sort imports
poetry run mypy app/                                    # type check
```

## Architecture

Follows the standard Kubestep pattern: `cli.py` -> `Manager` -> `DatasetLoadingService`.

### Key flow

1. `cli.py` exposes a `run` command with `--version`, `--source`, `--output-dir`, and optional `--path-override`, `--sample-size`, `--seed`
2. `Manager` builds a boto3 S3 client (with LakeFS endpoint precedence if configured) and instantiates the service
3. `DatasetLoadingService.run()` resolves the S3 source, downloads files, writes `data.yaml`, validates structure, optionally samples, and writes `dataset_stats.json`

### Two config layers

- **`models/config.py`** (`Config` / `BaseSettings`) — runtime settings from env vars: log level, timeouts, retries, AWS/S3 credentials, LakeFS credentials
- **`models/dataset.py`** — domain models: `YoloDatasetParams` (CLI input) and `YoloDatasetStats` (output artifact)

### S3 client precedence

The Manager selects credentials based on what's configured:
- If `LAKEFS_ENDPOINT` is set, uses LakeFS credentials (`LAKEFS_ACCESS_KEY`, `LAKEFS_SECRET_KEY`) with the LakeFS endpoint as a custom S3 endpoint
- Otherwise uses standard AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) with optional `S3_ENDPOINT_URL` for MinIO etc.

### Source resolution

- If `--path-override` is provided, parses it as `s3://bucket/prefix/`
- Otherwise uses default bucket `temp-mlops` with prefix `datasets/speedplus_yolo/{version}/`

### Validation rules

- `data.yaml` must exist with keys: `path`, `train`, `val`, `test`, `kpt_shape`, `names`
- Each split (`train`, `val`, `test`) must have an `images/` directory with at least one image
- Every image must have a corresponding label file (`.txt` alongside in `labels/`)
- Label files must not be empty
- Label lines must have exactly 38 tokens (1 class + 4 bbox + 11 keypoints x 3 coords)

### Dataset constants

- `SPLITS = ("train", "val", "test")`
- `SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}`
- `_EXPECTED_TOKENS_PER_LINE = 38` — YOLO pose format for spacecraft (11 keypoints)
- `_DEFAULT_BUCKET = "temp-mlops"`

## Testing Patterns

- Tests mock `boto3.client` via a helper `_make_mock_s3()` that serves files from a local temp directory
- `_build_yolo_tree()` helper creates parametrized YOLO dataset structures (images per split, token counts, missing/empty files)
- Test classes are grouped: `TestHappyPath`, `TestSourceResolution`, `TestValidation`, `TestSampling`
- Manager tests mock `DatasetLoadingService` to verify wiring only
