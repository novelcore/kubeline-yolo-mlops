"""CLI interface for the Model Registration step."""

from typing import Optional

import typer

from app.manager import Manager


def main() -> None:
    """Main CLI entry point for the Model Registration step."""
    app = typer.Typer(
        name="model-registration",
        help="Register trained models for the Infinite Orbits MLOps pipeline.",
        no_args_is_help=True,
    )

    @app.command(name="run", help="Register a trained model to the registry.")
    def run(
        model_name: str = typer.Option(..., help="Name of the model to register."),
        checkpoint_path: str = typer.Option(..., help="Path to the model checkpoint."),
        registry_url: Optional[str] = typer.Option(None, help="Model registry URL (overrides env)."),
        version: Optional[str] = typer.Option(None, help="Version string (auto-generated if omitted)."),
        tags: Optional[list[str]] = typer.Option(None, help="Tags to apply to the model."),
        promote_to: Optional[str] = typer.Option(None, help="Promote to stage: staging or production."),
    ) -> None:
        try:
            manager = Manager()
            manager.run(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                registry_url=registry_url,
                version=version,
                tags=tags or [],
                promote_to=promote_to,
            )
        except Exception as e:
            print(f"Error: {e}")
            raise typer.Exit(code=1)

    app()
