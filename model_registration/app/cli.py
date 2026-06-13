"""CLI interface for the Model Registration step."""

import json
from typing import Optional

import typer

from app.manager import Manager


def main() -> None:
    """Main CLI entry point for the Model Registration step."""
    app = typer.Typer(
        name="model-registration",
        help="Register trained YOLO models in the MLflow model registry.",
        no_args_is_help=True,
    )

    @app.command(name="run", help="Register a trained model checkpoint in MLflow.")
    def run(
        mlflow_run_id: str = typer.Option(
            ...,
            help="MLflow run ID from the training step.",
        ),
        best_checkpoint_path: str = typer.Option(
            ...,
            help="S3 URI of the best.pt checkpoint produced by training.",
        ),
        last_checkpoint_path: Optional[str] = typer.Option(
            None,
            help=(
                "S3 URI of the last.pt checkpoint. "
                "Derived from --best-checkpoint-path if omitted."
            ),
        ),
        registered_model_name: Optional[str] = typer.Option(
            None,
            help="Override the registered model name (defaults to REGISTERED_MODEL_NAME env var).",
        ),
        promote_to: Optional[str] = typer.Option(
            None,
            help="Assign this registry alias to the best.pt version after registration (champion or challenger).",
        ),
        dataset_version: Optional[str] = typer.Option(
            None,
            help="Lineage tag: dataset version used for training.",
        ),
        dataset_sample_size: Optional[int] = typer.Option(
            None,
            help="Lineage tag: number of samples used (null = full dataset).",
        ),
        config_hash: Optional[str] = typer.Option(
            None,
            help="Lineage tag: SHA-256 hash of the experiment YAML.",
        ),
        git_commit: Optional[str] = typer.Option(
            None,
            help="Lineage tag: git commit hash at the time of registration.",
        ),
        model_variant: Optional[str] = typer.Option(
            None,
            help="Lineage tag: YOLO model variant, e.g. yolov8n-pose.pt.",
        ),
        best_map50: Optional[float] = typer.Option(
            None,
            help="Lineage tag: best validation mAP50 achieved during training.",
        ),
        exported_models: Optional[str] = typer.Option(
            None,
            help=(
                "JSON dict mapping export labels to S3 URIs, "
                "e.g. '{\"engine_fp16\": \"s3://bucket/model.engine\"}'."
            ),
        ),
    ) -> None:
        try:
            manager = Manager()
            parsed_exports: Optional[dict[str, str]] = None
            if exported_models:
                parsed_exports = json.loads(exported_models)

            result = manager.run(
                mlflow_run_id=mlflow_run_id,
                best_checkpoint_path=best_checkpoint_path,
                last_checkpoint_path=last_checkpoint_path,
                registered_model_name=registered_model_name,
                promote_to=promote_to,
                dataset_version=dataset_version,
                dataset_sample_size=dataset_sample_size,
                config_hash=config_hash,
                git_commit=git_commit,
                model_variant=model_variant,
                best_map50=best_map50,
                exported_models=parsed_exports,
            )
            typer.echo(result.model_dump_json(indent=2))
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

    app()
