"""CLI interface for the Dataset Loading step."""

import typer

from app.manager import Manager


def main() -> None:
    """Main CLI entry point for the Dataset Loading step."""
    app = typer.Typer(
        name="dataset-loading",
        help="Load and preprocess datasets for the Infinite Orbits MLOps pipeline.",
        no_args_is_help=True,
    )

    @app.command(name="run", help="Load and split a dataset.")
    def run(
        source_path: str = typer.Option(..., help="Path to the raw dataset source."),
        output_dir: str = typer.Option(..., help="Directory to write processed splits."),
        format: str = typer.Option(..., help="Dataset format: csv, parquet, jsonl, or hf."),
        train_split: float = typer.Option(0.8, help="Fraction of data for training."),
        val_split: float = typer.Option(0.1, help="Fraction of data for validation."),
        test_split: float = typer.Option(0.1, help="Fraction of data for testing."),
    ) -> None:
        try:
            manager = Manager()
            manager.run(
                source_path=source_path,
                output_dir=output_dir,
                format=format,
                train_split=train_split,
                val_split=val_split,
                test_split=test_split,
            )
        except Exception as e:
            print(f"Error: {e}")
            raise typer.Exit(code=1)

    app()
