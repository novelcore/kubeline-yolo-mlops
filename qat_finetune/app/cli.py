"""CLI interface for the QAT fine-tune step."""

import json
import os
from typing import Optional

import typer

from app.manager import Manager


def main() -> None:
    """Main CLI entry point for the QAT fine-tune step."""
    app = typer.Typer(
        name="qat-finetune",
        help="Run PT2E QAT fine-tuning on a YOLO FP32 checkpoint and export INT8 TFLite.",
        no_args_is_help=True,
    )

    @app.command(name="run", help="Run the QAT fine-tune pipeline.")
    def run(
        fp32_checkpoint_path: str = typer.Option(
            ...,
            help="Local path or s3:// URI to the FP32 .pt checkpoint from model-training.",
        ),
        source_mlflow_run_id: str = typer.Option(
            ...,
            help="MLflow run ID of the model-training run that produced the checkpoint.",
        ),
        dataset_dir: str = typer.Option(
            ...,
            help="Local path to the YOLO dataset directory (images/ and labels/ subdirs).",
        ),
        output_dir: str = typer.Option(
            ...,
            help="Local directory for intermediate artifacts.",
        ),
        output_bucket: str = typer.Option(
            ...,
            help="S3 bucket for TFLite artifact upload.",
        ),
        output_prefix: str = typer.Option(
            ...,
            help="S3 key prefix for TFLite artifact (e.g. 'qat/exp-001').",
        ),
        experiment_name: str = typer.Option(
            ...,
            help="MLflow experiment name for this QAT run.",
        ),
        image_size: int = typer.Option(640, help="Input image size (square)."),
        device: Optional[str] = typer.Option(
            None,
            help="Compute device: 'cuda', 'cpu', '0'. Defaults to cuda if available.",
        ),
        qat_epochs: int = typer.Option(10, help="Number of QAT fine-tune epochs."),
        qat_lr: float = typer.Option(1e-4, help="Learning rate for QAT fine-tuning."),
        calibration_frames: int = typer.Option(
            512, help="Number of calibration frames sampled from the training set."
        ),
        calibration_seed: int = typer.Option(
            42, help="RNG seed for deterministic calibration frame sampling."
        ),
        parity_frames: int = typer.Option(
            100, help="Number of frames for the INT8 vs FP32 parity check."
        ),
        parity_max_abs_error: float = typer.Option(
            0.05, help="Maximum allowed max-abs-error in the parity check."
        ),
    ) -> None:
        try:
            manager = Manager()
            result = manager.run(
                fp32_checkpoint_path=fp32_checkpoint_path,
                source_mlflow_run_id=source_mlflow_run_id,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                output_bucket=output_bucket,
                output_prefix=output_prefix,
                experiment_name=experiment_name,
                image_size=image_size,
                device=device,
                qat_epochs=qat_epochs,
                qat_lr=qat_lr,
                calibration_frames=calibration_frames,
                calibration_seed=calibration_seed,
                parity_frames=parity_frames,
                parity_max_abs_error=parity_max_abs_error,
            )
            result_path = os.path.join(output_dir, "qat_result.json")
            with open(result_path, "w") as f:
                f.write(result.model_dump_json(indent=2))
            typer.echo(result.model_dump_json(indent=2))
            typer.echo(f"QAT result written to: {result_path}")
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

    app()
