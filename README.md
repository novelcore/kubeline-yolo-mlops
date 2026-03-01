# Infinite Orbits MLOps Pipeline

An Argo Workflows-based MLOps pipeline for the Infinite Orbits project, built with the Kubestep Python Template pattern.

## Pipeline Steps

The pipeline consists of four sequential steps, each packaged as an independent containerized application:

1. **config_validation** — Validates the pipeline configuration (hyperparameters, paths, model settings) before any work begins.
2. **dataset_loading** — Loads and preprocesses the dataset from the configured source, applies splits, and persists artifacts.
3. **model_training** — Trains the model using the validated configuration and loaded dataset, tracks metrics, and saves checkpoints.
4. **model_registration** — Registers the trained model to the model registry with metadata, versioning, and promotion tags.

## Project Structure

```
infinite-orbits-mlops/
├── config_validation/      # Step 1: Validate pipeline config
│   ├── app/
│   ├── tests/
│   ├── Dockerfile
│   └── pyproject.toml
├── dataset_loading/        # Step 2: Load & preprocess data
│   ├── app/
│   ├── tests/
│   ├── Dockerfile
│   └── pyproject.toml
├── model_training/         # Step 3: Train the model
│   ├── app/
│   ├── tests/
│   ├── Dockerfile
│   └── pyproject.toml
├── model_registration/     # Step 4: Register the model
│   ├── app/
│   ├── tests/
│   ├── Dockerfile
│   └── pyproject.toml
└── README.md
```

Each step follows the same internal structure:

- `app/cli.py` — CLI entry point using Typer
- `app/manager.py` — Orchestrator that wires config, models, and services
- `app/models/config.py` — Pydantic Settings for environment-based configuration
- `app/models/` — Domain-specific Pydantic models
- `app/services/` — Business logic
- `tests/` — Unit tests

## Quick Start

Each step can be developed and tested independently:

```bash
cd config_validation  # or any step directory
poetry install
poetry run app run [OPTIONS]
poetry run pytest
```

## Docker

Build and run any step:

```bash
docker build -t io-config-validation ./config_validation
docker run io-config-validation run --config-path /data/pipeline_config.yaml
```

## Configuration

All steps use Pydantic BaseSettings and can be configured via environment variables or `.env` files. See `env.example` in each step directory.
