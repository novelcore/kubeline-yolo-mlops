# Config Validation

Pipeline step that validates the MLOps pipeline configuration before execution begins.

## Quick Start

1. Install dependencies: `poetry install`
2. Run the application: `poetry run config-validation run --config-path ./pipeline_config.yaml`
3. Run tests: `poetry run pytest`

## Structure

- `cli.py` - Command-line interface using Typer
- `manager.py` - Main application orchestrator
- `models/` - Pydantic data models and configuration
- `services/` - Validation business logic

For complete documentation, see the root `README.md`.
