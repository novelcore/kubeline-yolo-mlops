# Infinite Orbits MLOps Pipeline

An Argo Workflows-based MLOps pipeline for training YOLO pose-estimation models for spacecraft detection. Built with the **Kubestep Python Template** pattern — four independent, containerized Python steps running sequentially.

## Pipeline Overview

```
config_validation → dataset_loading → model_training → model_registration
```

| Step | Description |
|------|-------------|
| **config_validation** | Validates the full pipeline YAML (schema + liveness checks against S3, MLflow) |
| **dataset_loading** | Downloads a YOLO pose dataset from S3/LakeFS, validates structure, generates stats |
| **model_training** | Trains a YOLO pose model with Ultralytics, tracks experiments via MLflow |
| **model_registration** | Registers trained checkpoints to MLflow model registry with lineage metadata |

## Tech Stack

- **Python 3.12** + **Poetry** for dependency management
- **Pydantic v2** + **pydantic-settings** for config and domain models
- **Typer** for CLIs
- **Ultralytics** (YOLOv8–v11) for pose-estimation training
- **MLflow** for experiment tracking and model registry
- **boto3** for S3 and LakeFS integration
- **Docker** for containerization

## Project Structure

```
kubeline-yolo-mlops/
├── config_validation/         # Step 1: Validate pipeline config
├── dataset_loading/           # Step 2: Load & validate YOLO dataset
├── model_training/            # Step 3: Train YOLO pose model
├── model_registration/        # Step 4: Register model to MLflow
├── orchestrate.sh             # Pipeline orchestrator (local or Docker)
├── pipeline_config.example.yaml  # Full reference configuration
├── pipeline_config.toy.yaml   # Minimal config for quick testing
└── artifacts/                 # Shared output directory between steps
```

Each step follows an identical internal layout:

```
<step>/
├── app/
│   ├── cli.py            # Typer CLI entry point
│   ├── manager.py        # Wires Config + services, contains Manager.run()
│   ├── models/
│   │   ├── config.py     # Pydantic BaseSettings (env vars / .env)
│   │   └── <domain>.py   # Domain-specific Pydantic models
│   └── services/
│       └── <service>.py  # Core business logic
├── tests/                # pytest unit tests
├── Dockerfile
├── env.example           # Documents all env vars for the step
└── pyproject.toml        # Poetry config + script entry point
```

## Quick Start

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/)
- Docker (for containerized runs)
- Access to S3/LakeFS storage and MLflow server

### Run the Full Pipeline Locally

```bash
./orchestrate.sh --config pipeline_config.example.yaml --mode local
```

### Run a Single Step

```bash
./orchestrate.sh --config pipeline_config.yaml --step model_training
```

### Resume from a Step

```bash
./orchestrate.sh --config pipeline_config.yaml --from dataset_loading
```

### Dry Run (Print Commands Without Executing)

```bash
./orchestrate.sh --config pipeline_config.yaml --dry-run
```

## Step Details

### Step 1: Config Validation

Validates the pipeline YAML against Pydantic schemas and optionally runs liveness checks against live infrastructure (S3 bucket access, MLflow health, pretrained weights availability).

```bash
cd config_validation
poetry install
poetry run config-validation run \
  --experiment-name spacecraft-pose-v1 \
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
```

**Validation rules include:**
- `image_size` must be a multiple of 32
- `model.variant` must match `yolov{8|9|10|11}[nsmlx]-pose.pt`
- `close_mosaic < epochs` and `warmup_epochs < epochs`
- `checkpointing.storage_path` must start with `s3://` or `lakefs://`

**Output:** `artifacts/validated_config.json`

### Step 2: Dataset Loading

Downloads a YOLO pose dataset from S3 or LakeFS with concurrent downloads (8 workers). Validates dataset structure: splits, image-label pairing, and label format (38 tokens per line for spacecraft 11-keypoint pose).

```bash
cd dataset_loading
poetry install
poetry run dataset-loading run \
  --version v1 \
  --source s3 \
  --output-dir /tmp/dataset \
  --seed 42
```

**Features:**
- Pre-download S3 structure validation (fail-fast)
- `data.yaml` generation with runtime paths
- Optional sub-sampling with reproducible seed
- Labels-only and manifest-only modes for S3 streaming

**Output:** `artifacts/dataset/` (images, labels, `data.yaml`, `dataset_stats.json`)

### Step 3: Model Training

Trains a YOLO pose model using Ultralytics with full MLflow experiment tracking. Supports local datasets and S3 streaming with LRU disk caching. Monitors CPU, GPU, and memory usage throughout training.

```bash
cd model_training
poetry install
poetry run model-training run \
  --model-variant yolov8n-pose.pt \
  --experiment-name spacecraft-pose-v1-yolov8n \
  --dataset-dir /artifacts/dataset \
  --output-dir /artifacts/training \
  --source local \
  --epochs 100 \
  --batch-size 16 \
  --image-size 640 \
  --learning-rate 0.01 \
  --optimizer SGD \
  --patience 50 \
  --checkpoint-interval 10 \
  --checkpoint-bucket temp-mlops \
  --checkpoint-prefix checkpoints
```

**Dataset modes:**
- **local** — reads from local `dataset-dir`
- **s3** — streams images from S3, labels from local disk
- **s3-streaming-labels** — auto-detected when `dataset_manifest.json` has `label_keys`

**Output:** `artifacts/training/training_result.json`, checkpoints uploaded to S3

### Step 4: Model Registration

Registers trained model checkpoints to the MLflow model registry with lineage tags and optional promotion to Staging or Production.

```bash
cd model_registration
poetry install
poetry run model-registration run \
  --mlflow-run-id abc123def456 \
  --best-checkpoint-path s3://bucket/spacecraft-pose-v1-yolov8n/best.pt \
  --registered-model-name spacecraft-pose-yolo \
  --promote-to Staging \
  --dataset-version v1 \
  --model-variant yolov8n-pose.pt \
  --best-map50 0.85
```

**Lineage tags:** `dataset_version`, `sample_size`, `config_hash`, `git_commit`, `model_variant`, `best_map50`

**Output:** MLflow model registry entry with version numbers

## Docker

Build and run any step:

```bash
# Build
docker build -t io-config-validation ./config_validation
docker build -t io-dataset-loading ./dataset_loading
docker build -t io-model-training ./model_training
docker build -t io-model-registration ./model_registration

# Run (example: model training with GPU)
docker run --gpus all \
  -v $(pwd)/artifacts:/artifacts \
  io-model-training run \
  --model-variant yolov8n-pose.pt \
  --experiment-name spacecraft-pose-v1-yolov8n \
  --dataset-dir /artifacts/dataset \
  --output-dir /artifacts/training \
  --source local \
  --epochs 100
```

The orchestrator can also run steps in Docker mode:

```bash
./orchestrate.sh --config pipeline_config.yaml --mode docker
./orchestrate.sh --config pipeline_config.yaml --mode docker --skip-build
```

## Configuration

### Pipeline Config (YAML)

The pipeline YAML (`pipeline_config.example.yaml`) carries cross-step parameters:

| Section | Key Parameters |
|---------|---------------|
| `experiment` | name, description, tags |
| `dataset` | version, source (s3/lakefs), path_override, sample_size, seed |
| `model` | variant (yolov8n-pose.pt through yolov11x-pose.pt), pretrained_weights |
| `training` | epochs, batch_size, image_size, learning_rate, optimizer, warmup, regularization, augmentation, pose loss gains |
| `checkpointing` | interval_epochs, storage_path (s3:// or lakefs://), resume_from |
| `early_stopping` | patience |
| `augmentation` | HSV, rotation, translation, scaling, flipping, mosaic, mixup, erasing |
| `registration` | registered_model_name, promote_to |

### Environment Variables

Each step reads runtime settings (credentials, timeouts, log levels) from environment variables or `.env` files. See `env.example` in each step directory.

Key variables common across steps:

| Variable | Description |
|----------|-------------|
| `LOG_LEVEL` | Logging verbosity (default: INFO) |
| `AWS_ACCESS_KEY_ID` | S3/LakeFS access key |
| `AWS_SECRET_ACCESS_KEY` | S3/LakeFS secret key |
| `S3_ENDPOINT_URL` | Custom S3 endpoint (MinIO, LakeFS) |
| `MLFLOW_TRACKING_URI` | MLflow server URL |
| `MLFLOW_TRACKING_USERNAME` | MLflow auth username |
| `MLFLOW_TRACKING_PASSWORD` | MLflow auth password |
| `LAKEFS_ENDPOINT` | LakeFS server endpoint |
| `LAKEFS_ACCESS_KEY` | LakeFS access key |
| `LAKEFS_SECRET_KEY` | LakeFS secret key |

## Development

```bash
cd <step_directory>
poetry install

# Run tests
poetry run pytest

# Format
poetry run black app/ tests/
poetry run isort app/ tests/

# Type check
poetry run mypy app/
```

## Orchestrator Reference

```
./orchestrate.sh --config <path> [OPTIONS]

Options:
  --config PATH        Path to pipeline YAML (required)
  --mode local|docker  Execution mode (default: local)
  --image-prefix STR   Docker image prefix (default: io)
  --skip-build         Skip Docker image builds
  --dry-run            Print commands without executing
  --step STEP          Run a single step
  --from STEP          Start from a specific step (inclusive)
  --verbose            Enable debug logging
  -h, --help           Show help
```
