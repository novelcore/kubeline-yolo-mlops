# Model Registration Agent Memory

## Project location
`/home/thanos/Documents/kubeline-yolo-mlops/model_registration/`

## Step 4 implementation status
Fully implemented (2026-03-02). All stubs replaced with real MLflow calls.

## Key architectural decisions

### Input: individual CLI flags, no training summary JSON
The orchestrator (`orchestrate.sh`) passes individual flags to Step 4 — no JSON file is written by Step 3. `run_model_registration()` in `orchestrate.sh` currently only passes `--model-name`, `--checkpoint-path`, `--registry-url`. If lineage tags are needed, orchestrate.sh must be extended to pass `--mlflow-run-id`, `--best-checkpoint-path`, `--dataset-version`, etc.

### MLflow registration pattern
`mlflow.register_model(model_uri=s3_uri, name=registered_model_name)` — uses the S3 URI directly as `model_uri`. Tags are set via `MlflowClient().set_model_version_tag()` after registration.

### last.pt derivation
If `--last-checkpoint-path` is not provided, the service derives it by replacing `best.pt` with `last.pt` in the best checkpoint path. If `best.pt` is not in the path, last.pt registration is skipped with a warning.

### Retry helper pattern
`_with_retry(fn: Callable[[], Any]) -> Any` with `_RETRY_DELAYS = (1, 2, 4)` seconds.
- Uses inner named function (not lambda with defaults) inside loops to satisfy mypy.
- `Callable[[], Any]` from `collections.abc` — not `typing.Callable` — for Python 3.12.

### Model stage
Default `promote_to=None` — no automatic promotion. Stage transitions only when `--promote-to` is explicitly passed. Valid values: `Staging`, `Production`, `Archived`.

## Config env vars (required)
- `MLFLOW_TRACKING_URI` — required, no default
- `MLFLOW_EXPERIMENT_NAME` — default `"infinite-orbits"`
- `REGISTERED_MODEL_NAME` — default `"spacecraft-pose-yolo"`

## MLflow model name convention
Default registered model name: `"spacecraft-pose-yolo"` (configurable via `REGISTERED_MODEL_NAME` env var or `--registered-model-name` CLI flag).

## Lineage tags applied to each registered version
`checkpoint_type`, `training_run_id`, `dataset_version`, `dataset_sample_size`, `config_hash`, `git_commit`, `model_variant`, `best_mAP50` — all optional except `checkpoint_type`.

## mypy gotcha
Lambdas with default arguments (e.g. `lambda k=key, v=value: ...`) inside loops cause `Cannot infer type of lambda [misc]`. Use a named inner function with explicit type annotations instead.

## Test structure
- `tests/test_config.py` — 7 tests, env var reading via `monkeypatch.setenv`
- `tests/test_manager.py` — 6 tests in `TestManagerRun`, all mock `ModelRegistrationService`
- `tests/test_model_registration_service.py` — 14 tests in 4 classes, all mock `mlflow` and `MlflowClient`
- Patch targets: `app.services.model_registration.mlflow`, `app.services.model_registration.MlflowClient`, `app.services.model_registration.time.sleep`

## pyproject.toml version pins
Matches model_training step: `mlflow = ">=2.13"`, `boto3 = ">=1.34"`. python_version and black target-version both set to `3.12`.
