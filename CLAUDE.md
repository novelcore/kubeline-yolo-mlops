# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An Argo Workflows-based MLOps pipeline for the KAOS YOLO project. The pipeline consists of four sequential steps, each packaged as an independent containerized Python application following the **Kubestep Python Template** pattern.

Steps run in order: `config_validation` → `dataset_loading` → `model_training` → `model_registration`.

## Commands

All commands are run from within the step's directory (e.g., `cd config_validation`).

**Install dependencies:**
```bash
poetry install
```

**Run a step locally:**
```bash
# Each step has its own script name
poetry run config-validation run --config-path ../pipeline_config.example.yaml
poetry run dataset-loading run --source-path /data/train.parquet --output-dir /tmp/dataset --format parquet
poetry run model-training run --model-name bert-base --dataset-dir /tmp/dataset --output-dir /tmp/checkpoints
poetry run model-registration run --model-name bert-base-finetuned --checkpoint-path /tmp/checkpoints/model.pt --registry-url https://registry.example.com
```

**Run tests:**
```bash
poetry run pytest                        # all tests
poetry run pytest tests/test_manager.py  # single test file
```

**Lint and type-check:**
```bash
poetry run black app/ tests/
poetry run isort app/ tests/
poetry run mypy app/
```

**Run the full pipeline (local):**
```bash
./orchestrate.sh --config pipeline_config.example.yaml --mode local
```

**Run a single step or resume from a step via orchestrator:**
```bash
./orchestrate.sh --config pipeline_config.yaml --step model_training
./orchestrate.sh --config pipeline_config.yaml --from dataset_loading
./orchestrate.sh --config pipeline_config.yaml --dry-run
```

**Build and run a step with Docker:**
```bash
docker build -t io-config-validation ./config_validation
docker run io-config-validation run --config-path /data/pipeline_config.yaml
```

## Architecture

Every step is a self-contained Python package with an identical internal layout:

```
<step>/
├── app/
│   ├── cli.py          # Typer CLI; exposes a `run` subcommand
│   ├── manager.py      # Wires Config, services; contains Manager.run()
│   ├── models/
│   │   ├── config.py   # Pydantic BaseSettings — reads from env vars / .env
│   │   └── <domain>.py # Domain-specific Pydantic models
│   └── services/
│       └── <service>.py # Core business logic
├── tests/
├── Dockerfile          # python:3.12-alpine; Poetry installs --only=main
├── env.example         # Documents all env vars for the step
└── pyproject.toml      # Defines the poetry script entry point
```

**Key patterns:**
- `cli.py` creates a `typer.Typer` app, defines a `run` command, instantiates `Manager`, and calls `manager.run(...)`.
- `Manager.__init__` reads `Config()` (Pydantic BaseSettings) and instantiates services. `Manager.run()` delegates to services.
- `models/config.py` uses `pydantic_settings.BaseSettings` with `env_file=".env"`. Step-level runtime settings (log level, timeouts, feature flags) live here — not in the pipeline YAML.
- The pipeline YAML (`pipeline_config.yaml`) carries cross-step configuration (dataset paths, hyperparameters, registry URL). It is validated by `config_validation` using Pydantic models in `app/models/pipeline_config.py`.
- `orchestrate.sh` parses the pipeline YAML into shell variables with the `CFG_<section>_<key>` naming convention and passes the values as CLI flags to each step.
- Docker images are named `<image-prefix>-<step-name-with-dashes>` (e.g., `io-config-validation`). The orchestrator mounts the config at `/data/pipeline_config.yaml` and an `artifacts/` volume at `/artifacts`.

## kubeline.yaml — Pipeline DSL

`kubeline.yaml` (version 2) is the pipeline definition consumed by the KAOS
in-cluster `ml-ci-build` CI generator (see your-org/your-operator and
your-org/your-charts). The generator reads this file and emits an Argo
`WorkflowTemplate`.

**What the app owns:** step list, dependencies, `command`/`args` (Python shims
that parse `PIPELINE_CONFIG` and invoke the step CLIs), `inputs`/`outputs`
parameter wiring, and optional resource overrides.

**What the platform injects automatically:** image refs, nodeSelectors /
tolerations / compute classes, `serviceAccountName`, `podMetadata` cost labels,
lakeFS env (`LAKEFS_ENDPOINT`, `LAKEFS_REPOSITORY`, `LAKEFS_ACCESS_KEY_ID`,
`LAKEFS_SECRET_ACCESS_KEY`), MLflow env (`MLFLOW_TRACKING_URI`), and
`PIPELINE_CONFIG` (the raw pipeline config YAML) into every step's env.

**Constraint:** `dataset.manifest_only: true` is required. Full-download and
labels-only modes assume a shared filesystem between step pods, which the
cluster does not provide.

## Tech Stack

- **Python** 3.12, **Poetry** for dependency management
- **Pydantic v2** + **pydantic-settings** for config and domain models
- **Typer** for CLIs
- **pytest** + **pytest-cov** for testing
- **black** + **isort** for formatting, **mypy** for type checking
