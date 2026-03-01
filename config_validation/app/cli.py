"""CLI interface for the Config Validation step."""

import typer

from app.manager import Manager


def main() -> None:
    """Main CLI entry point for the Config Validation step."""
    app = typer.Typer(
        name="config-validation",
        help="Validate the Infinite Orbits MLOps pipeline configuration.",
        no_args_is_help=True,
    )

    @app.command(name="run", help="Validate a pipeline configuration file.")
    def run(
        config_path: str = typer.Option(..., help="Path to the pipeline config YAML file."),
    ) -> None:
        try:
            manager = Manager()
            manager.run(config_path=config_path)
        except Exception as e:
            print(f"Error: {e}")
            raise typer.Exit(code=1)

    app()
