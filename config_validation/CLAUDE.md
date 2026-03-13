# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Step Does

First step in the MLOps pipeline (`config_validation` -> dataset_loading -> model_training -> model_registration). Validates the full pipeline YAML configuration against Pydantic schemas, then optionally runs liveness checks against live infrastructure (S3 dataset paths, pretrained weights, MLflow tracking server, checkpoint resume paths). Outputs a validated JSON artifact for downstream steps.

## Commands

```bash
poetry install                          # install dependencies
poetry run config-validation run \      # run locally (all flags are required unless noted)
  --experiment-name my-exp \
  --dataset-version v1 \
  --dataset-source s3 \
  --model-variant yolov8n-pose.pt \
  --training-epochs 100 \
  --training-batch-size 16 \
  --training-image-size 640 \
  --training-learning-rate 0.01 \
  --training-optimizer SGD \
  --checkpointing-interval-epochs 10 \
  --checkpointing-storage-path s3://bucket/checkpoints \
  --early-stopping-patience 50 \
  --output-path /tmp/validated.json

poetry run pytest                               # all tests
poetry run pytest tests/test_pipeline_config.py  # single file
poetry run pytest -k "test_valid_config"         # single test by name

poetry run black app/ tests/                    # format
poetry run isort app/ tests/                    # sort imports
poetry run mypy app/                            # type check
```

Set `SKIP_LIVENESS_CHECKS=true` in `.env` or environment to bypass S3/MLflow connectivity checks during local development.

## Architecture

The CLI (`cli.py`) differs from other pipeline steps: it accepts **every pipeline config field as a Typer option** (not a `--config-path` to a YAML file). The orchestrator (`orchestrate.sh`) parses the pipeline YAML into individual `--flag value` pairs passed to this step.

### Key flow

1. `cli.py` assembles a nested dict from CLI flags and passes it to `Manager.run(config_dict=...)`
2. `Manager` instantiates `ConfigValidationService` with runtime settings from `Config` (env vars)
3. `ConfigValidationService.run()` validates the dict against `PipelineConfig` (Pydantic), runs optional liveness checks, and writes a JSON artifact

### Two config layers

- **`models/config.py`** (`Config` / `BaseSettings`) — runtime settings from env vars: log level, timeouts, `SKIP_LIVENESS_CHECKS`, `MLFLOW_TRACKING_URI`
- **`models/pipeline_config.py`** (`PipelineConfig`) — pipeline parameters from the YAML. Seven nested sub-models: `ExperimentConfig`, `DatasetConfig`, `ModelConfig`, `TrainingConfig`, `CheckpointingConfig`, `EarlyStoppingConfig`, `AugmentationConfig`

### Validation rules worth knowing

- `TrainingConfig` uses `extra="forbid"` — unknown keys raise errors (prevents silent misconfiguration)
- `PipelineConfig` uses `extra="ignore"` — top-level unknown sections (e.g. `resources`) are silently dropped
- `image_size` must be a multiple of 32
- `model.variant` must match `yolov{8|9|10|11}{n|s|m|l|x}-pose.pt`
- Cross-field: `close_mosaic < epochs`, `warmup_epochs < epochs`
- `checkpointing.storage_path` must start with `s3://` or `lakefs://`
- `checkpointing.resume_from` must be `null`, `"auto"`, or an `s3://` path

### Liveness checks (when `SKIP_LIVENESS_CHECKS=false`)

All use retry logic with `max_retries` and `timeout` from `Config`:
1. **Dataset path** — S3 `list_objects_v2` on derived or overridden path
2. **Pretrained weights** — S3 `head_object` (skipped if `null`)
3. **MLflow** — HTTP GET to `{MLFLOW_TRACKING_URI}/health`
4. **Checkpoint resume** — `"auto"` scans for `.pt` files; specific path uses `head_object`

## Testing Patterns

- Tests mock `boto3.client` and `httpx.get` to avoid real AWS/MLflow calls
- `service_no_liveness` fixture: `skip_liveness_checks=True` for pure schema tests
- `service_with_liveness` fixture: `skip_liveness_checks=False` with mocked infrastructure
- Manager tests mock `ConfigValidationService` to verify wiring only
