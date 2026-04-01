# Environment Variables

Environment variables configure the platform plumbing for each step — server URLs, credentials, and runtime behaviour.

!!! info "These are managed by KAOS"
    In production, all environment variables are injected automatically from Kubernetes Secrets by the KAOS platform. You do not need to set them manually unless running a step locally.

## config_validation

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `APP_NAME` | No | `io-config-validation` | Application name in logs |
| `LOG_LEVEL` | No | `INFO` | Log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `MAX_RETRIES` | No | `3` | Retry count for connectivity checks |
| `TIMEOUT` | No | `30` | Timeout in seconds for liveness checks |
| `STRICT_MODE` | No | `true` | Fail on unknown config fields |
| `SKIP_LIVENESS_CHECKS` | No | `false` | Skip MLflow and S3 reachability checks (useful locally) |
| `MLFLOW_TRACKING_URI` | Conditional | — | Required when `SKIP_LIVENESS_CHECKS=false` |
| `AWS_ACCESS_KEY_ID` | No | — | S3 credential (or use IAM role) |
| `AWS_SECRET_ACCESS_KEY` | No | — | S3 credential (or use IAM role) |
| `AWS_DEFAULT_REGION` | No | `eu-central-1` | AWS region |
| `S3_ENDPOINT_URL` | No | — | Custom S3 endpoint (for MinIO or non-AWS S3) |
| `LAKEFS_ENDPOINT` | Conditional | — | Required when `dataset.source: "lakefs"` |
| `LAKEFS_ACCESS_KEY` | Conditional | — | LakeFS credential |
| `LAKEFS_SECRET_KEY` | Conditional | — | LakeFS credential |

## dataset_loading

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `APP_NAME` | No | `io-dataset-loading` | Application name in logs |
| `LOG_LEVEL` | No | `INFO` | Log verbosity |
| `MAX_RETRIES` | No | `3` | S3/LakeFS request retries |
| `TIMEOUT` | No | `120` | Request timeout in seconds |
| `AWS_DEFAULT_REGION` | No | `eu-central-1` | AWS region |
| `AWS_ACCESS_KEY_ID` | No | — | S3 credential (or IAM role) |
| `AWS_SECRET_ACCESS_KEY` | No | — | S3 credential (or IAM role) |
| `S3_ENDPOINT_URL` | No | — | Custom S3 endpoint |
| `LAKEFS_ENDPOINT` | Conditional | — | Required when `dataset.source: "lakefs"` |
| `LAKEFS_ACCESS_KEY` | Conditional | — | LakeFS credential |
| `LAKEFS_SECRET_KEY` | Conditional | — | LakeFS credential |

## model_training

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `APP_NAME` | No | `io-model-training` | Application name in logs |
| `LOG_LEVEL` | No | `INFO` | Log verbosity |
| `MLFLOW_TRACKING_URI` | Yes | — | Remote MLflow server URL |
| `MLFLOW_TRACKING_USERNAME` | No | — | MLflow basic auth username |
| `MLFLOW_TRACKING_PASSWORD` | No | — | MLflow basic auth password |
| `AWS_DEFAULT_REGION` | No | `eu-central-1` | AWS region |
| `AWS_ACCESS_KEY_ID` | No | — | S3 credential (or IAM role) |
| `AWS_SECRET_ACCESS_KEY` | No | — | S3 credential (or IAM role) |
| `S3_ENDPOINT_URL` | No | — | Custom S3 endpoint |
| `LAKEFS_ENDPOINT` | No | — | LakeFS endpoint for dataset access |
| `LAKEFS_ACCESS_KEY` | No | — | LakeFS credential |
| `LAKEFS_SECRET_KEY` | No | — | LakeFS credential |

## model_registration

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `APP_NAME` | No | `io-model-registration` | Application name in logs |
| `LOG_LEVEL` | No | `INFO` | Log verbosity |
| `MAX_RETRIES` | No | `3` | MLflow API retry count |
| `TIMEOUT` | No | `60` | MLflow call timeout in seconds |
| `MLFLOW_TRACKING_URI` | Yes | — | Remote MLflow server URL |
| `MLFLOW_TRACKING_USERNAME` | No | — | MLflow basic auth username |
| `MLFLOW_TRACKING_PASSWORD` | No | — | MLflow basic auth password |
| `MLFLOW_EXPERIMENT_NAME` | No | `infinite-orbits` | Experiment name (overrides the `experiment-name` parameter if set) |
| `REGISTERED_MODEL_NAME` | No | `spacecraft-pose-yolo` | Default model name when `registered-model-name` is empty |
| `AWS_ACCESS_KEY_ID` | No | — | S3 credential (for checkpoint access) |
| `AWS_SECRET_ACCESS_KEY` | No | — | S3 credential |
| `AWS_DEFAULT_REGION` | No | `eu-central-1` | AWS region |
| `AWS_ENDPOINT_URL` | No | — | Custom S3 endpoint |
| `LAKEFS_ENDPOINT` | No | — | LakeFS endpoint |
| `LAKEFS_ACCESS_KEY` | No | — | LakeFS credential |
| `LAKEFS_SECRET_KEY` | No | — | LakeFS credential |
