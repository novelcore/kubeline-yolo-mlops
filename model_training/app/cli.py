"""CLI interface for the Model Training step."""

import typer

from app.manager import Manager


def main() -> None:
    """Main CLI entry point for the Model Training step."""
    app = typer.Typer(
        name="model-training",
        help="Train models for the Infinite Orbits MLOps pipeline.",
        no_args_is_help=True,
    )

    @app.command(name="run", help="Train a model.")
    def run(
        model_name: str = typer.Option(..., help="Name or path of the model to train."),
        dataset_dir: str = typer.Option(..., help="Path to the processed dataset directory."),
        output_dir: str = typer.Option(..., help="Directory to save checkpoints."),
        epochs: int = typer.Option(10, help="Number of training epochs."),
        batch_size: int = typer.Option(32, help="Training batch size."),
        learning_rate: float = typer.Option(1e-4, help="Learning rate."),
        optimizer: str = typer.Option("adamw", help="Optimizer name."),
    ) -> None:
        try:
            manager = Manager()
            manager.run(
                model_name=model_name,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                optimizer=optimizer,
            )
        except Exception as e:
            print(f"Error: {e}")
            raise typer.Exit(code=1)

    app()
