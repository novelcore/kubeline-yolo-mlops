"""CLI interface for the Dataset Loading step."""

from typing import Optional

import typer

from app.manager import Manager


def main() -> None:
    """Main CLI entry point for the Dataset Loading step."""
    app = typer.Typer(
        name="dataset-loading",
        help="Download and validate a YOLO pose-estimation dataset from S3 or LakeFS.",
        no_args_is_help=True,
    )

    @app.command(
        name="run", help="Download, validate, and (optionally) sample a YOLO dataset."
    )
    def run(
        version: str = typer.Option(
            ...,
            help="Dataset version tag (e.g. 'v1'). Used to resolve the S3 path.",
        ),
        source: str = typer.Option(
            ...,
            help="Storage backend to use: 's3' or 'lakefs'.",
        ),
        lakefs_repo: Optional[str] = typer.Option(
            None,
            help="LakeFS repository name (used when source='lakefs').",
        ),
        lakefs_branch: Optional[str] = typer.Option(
            None,
            help="LakeFS branch name (used when source='lakefs').",
        ),
        output_dir: str = typer.Option(
            ...,
            help="Local directory where the YOLO dataset tree will be written.",
        ),
        path_override: str = typer.Option(
            "",
            help=(
                "Full S3 URI override (e.g. 's3://my-bucket/my/prefix/'). "
                "When set, the conventional bucket/prefix is ignored. "
                "Pass an empty string or omit to use the default location."
            ),
        ),
        labels_only: bool = typer.Option(
            False,
            help=(
                "Download only label files and data.yaml (no images). "
                "Used when images will be streamed from S3 during training."
            ),
        ),
        manifest_only: bool = typer.Option(
            False,
            help=(
                "Download only data.yaml and list S3 keys (no images, no labels). "
                "Both images and labels are streamed from S3 during training."
            ),
        ),
        sample_size: Optional[int] = typer.Option(
            None,
            help=(
                "If set, keep only this many image+label pairs per split after "
                "downloading. Excess files are deleted in-place."
            ),
        ),
        seed: int = typer.Option(
            42,
            help="Random seed for reproducible sampling.",
        ),
    ) -> None:
        try:
            manager = Manager()
            manager.run(
                version=version,
                source=source,
                output_dir=output_dir,
                lakefs_repo=lakefs_repo,
                lakefs_branch=lakefs_branch,
                path_override=path_override if path_override else None,
                labels_only=labels_only,
                manifest_only=manifest_only,
                sample_size=sample_size,
                seed=seed,
            )
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

    app()
