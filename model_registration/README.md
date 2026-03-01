# Model Registration

Pipeline step that registers a trained model to the model registry with metadata, versioning, and promotion tags.

## Quick Start

1. Install dependencies: `poetry install`
2. Run the application: `poetry run model-registration run --model-name bert-finetuned --checkpoint-path /artifacts/checkpoints/model.pt --registry-url https://registry.example.com`
3. Run tests: `poetry run pytest`

## Structure

- `cli.py` - Command-line interface using Typer
- `manager.py` - Main application orchestrator
- `models/` - Pydantic data models and configuration
- `services/` - Registration business logic

For complete documentation, see the root `README.md`.
