# Model Training

Pipeline step that trains the model using validated configuration and processed datasets.

## Quick Start

1. Install dependencies: `poetry install`
2. Run the application: `poetry run model-training run --model-name bert-base --dataset-dir /data/processed --output-dir /artifacts/checkpoints --epochs 10`
3. Run tests: `poetry run pytest`

## Structure

- `cli.py` - Command-line interface using Typer
- `manager.py` - Main application orchestrator
- `models/` - Pydantic data models and configuration
- `services/` - Training business logic

For complete documentation, see the root `README.md`.
