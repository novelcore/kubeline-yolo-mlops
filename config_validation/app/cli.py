"""CLI interface for the Config Validation step."""

from typing import Optional

import typer

from app.manager import Manager


def main() -> None:
    """Main CLI entry point for the Config Validation step."""
    app = typer.Typer(
        name="config-validation",
        help="Validate the Infinite Orbits MLOps pipeline configuration.",
        no_args_is_help=True,
    )

    @app.command(name="run", help="Validate a pipeline configuration.")
    def run(
        # Experiment
        experiment_name: str = typer.Option(..., help="Experiment name (alphanumeric and hyphens)."),
        experiment_description: Optional[str] = typer.Option(default=None, help="Optional experiment description."),
        # Dataset
        dataset_version: str = typer.Option(..., help="Dataset version string."),
        dataset_source: str = typer.Option(..., help="Dataset source: 's3' or 'lakefs'."),
        dataset_path_override: Optional[str] = typer.Option(default=None, help="Optional S3/LakeFS path override for the dataset."),
        dataset_sample_size: Optional[int] = typer.Option(default=None, help="Optional sample size (must be > 0)."),
        dataset_seed: int = typer.Option(default=42, help="Random seed for dataset splitting."),
        # Model
        model_variant: str = typer.Option(..., help="YOLO Pose model variant (e.g. yolov8n-pose.pt)."),
        model_pretrained_weights: Optional[str] = typer.Option(default=None, help="Optional S3 path to pretrained weights."),
        # Training
        training_epochs: int = typer.Option(..., help="Number of training epochs (> 0)."),
        training_batch_size: int = typer.Option(..., help="Training batch size (> 0)."),
        training_image_size: int = typer.Option(..., help="Input image size in pixels (multiple of 32)."),
        training_learning_rate: float = typer.Option(..., help="Learning rate (> 0)."),
        training_optimizer: str = typer.Option(..., help="Optimizer: 'SGD', 'Adam', or 'AdamW'."),
        training_warmup_epochs: float = typer.Option(default=3.0, help="Warmup epochs."),
        training_warmup_momentum: float = typer.Option(default=0.8, help="Warmup momentum."),
        training_weight_decay: float = typer.Option(default=0.0005, help="Weight decay."),
        # Checkpointing
        checkpointing_interval_epochs: int = typer.Option(..., help="Save a checkpoint every N epochs (> 0)."),
        checkpointing_storage_path: str = typer.Option(..., help="S3 or LakeFS URI for checkpoint storage."),
        checkpointing_resume_from: Optional[str] = typer.Option(default=None, help="Checkpoint to resume from: null, 'auto', or an S3 path."),
        # Early stopping
        early_stopping_patience: int = typer.Option(..., help="Early stopping patience in epochs (> 0)."),
        # Output
        output_path: Optional[str] = typer.Option(
            default=None,
            help="Path to write the validated config JSON artifact. If not set, no file is written.",
        ),
    ) -> None:
        try:
            config_dict = {
                "experiment": {
                    "name": experiment_name,
                    "description": experiment_description,
                },
                "dataset": {
                    "version": dataset_version,
                    "source": dataset_source,
                    "path_override": dataset_path_override,
                    "sample_size": dataset_sample_size,
                    "seed": dataset_seed,
                },
                "model": {
                    "variant": model_variant,
                    "pretrained_weights": model_pretrained_weights,
                },
                "training": {
                    "epochs": training_epochs,
                    "batch_size": training_batch_size,
                    "image_size": training_image_size,
                    "learning_rate": training_learning_rate,
                    "optimizer": training_optimizer,
                    "warmup_epochs": training_warmup_epochs,
                    "warmup_momentum": training_warmup_momentum,
                    "weight_decay": training_weight_decay,
                },
                "checkpointing": {
                    "interval_epochs": checkpointing_interval_epochs,
                    "storage_path": checkpointing_storage_path,
                    "resume_from": checkpointing_resume_from,
                },
                "early_stopping": {
                    "patience": early_stopping_patience,
                },
            }

            manager = Manager()
            manager.run(config_dict=config_dict, output_path=output_path)
        except Exception as e:
            print(f"Error: {e}")
            raise typer.Exit(code=1)

    app()