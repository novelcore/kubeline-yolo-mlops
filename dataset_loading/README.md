# Dataset Loading

Pipeline step that loads, preprocesses, and splits datasets for model training.

## Quick Start

1. Install dependencies: `poetry install`
2. Run the application: `poetry run dataset-loading run --source-path /data/raw --output-dir /data/processed --format parquet`
3. Run tests: `poetry run pytest`

## Structure

- `cli.py` - Command-line interface using Typer
- `manager.py` - Main application orchestrator
- `models/` - Pydantic data models and configuration
- `services/` - Dataset loading business logic

For complete documentation, see the root `README.md`.
