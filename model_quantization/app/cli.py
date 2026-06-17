"""CLI interface for the model-quantization step."""

import os
from typing import Optional

import typer

from app.manager import Manager


def main() -> None:
    app = typer.Typer(
        name="model-quantization",
        help="Run PTQ (Ultralytics INT8) or receive QAT TFLite and log to MLflow.",
        no_args_is_help=True,
    )

    @app.command(name="run")
    def run(
        mode: str = typer.Option(
            ...,
            help="Quantization mode: 'ptq' or 'qat'. ('none' is handled by DAG gating.)",
        ),
        source_mlflow_run_id: str = typer.Option(
            ..., help="MLflow run ID of the model-training step."
        ),
        dataset_dir: str = typer.Option(
            ..., help="Local path to YOLO dataset directory (must contain data.yaml)."
        ),
        output_dir: str = typer.Option(
            ..., help="Local directory for intermediate artifacts."
        ),
        output_bucket: str = typer.Option(..., help="S3 bucket for TFLite upload."),
        output_prefix: str = typer.Option(..., help="S3 key prefix for TFLite artifact."),
        experiment_name: str = typer.Option(..., help="MLflow experiment name."),
        fp32_checkpoint_path: Optional[str] = typer.Option(
            None, help="Local path or s3:// URI to FP32 checkpoint. Required for PTQ."
        ),
        tflite_s3_uri: Optional[str] = typer.Option(
            None, help="S3 URI of TFLite from qat-finetune. Required for QAT."
        ),
        qat_run_id: Optional[str] = typer.Option(
            None, help="MLflow run ID from qat-finetune (QAT lineage tag)."
        ),
        image_size: int = typer.Option(640, help="Input image size."),
        calibration_frames: int = typer.Option(512, help="PTQ calibration frames."),
        calibration_seed: int = typer.Option(42, help="Calibration sampling seed."),
        parity_frames: int = typer.Option(100, help="Frames for parity check."),
        parity_max_abs_error: float = typer.Option(
            0.05, help="Max allowed abs error in parity check."
        ),
    ) -> None:
        try:
            manager = Manager()
            result = manager.run(
                mode=mode,
                source_mlflow_run_id=source_mlflow_run_id,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                output_bucket=output_bucket,
                output_prefix=output_prefix,
                experiment_name=experiment_name,
                fp32_checkpoint_path=fp32_checkpoint_path,
                tflite_s3_uri=tflite_s3_uri,
                qat_run_id=qat_run_id,
                image_size=image_size,
                calibration_frames=calibration_frames,
                calibration_seed=calibration_seed,
                parity_frames=parity_frames,
                parity_max_abs_error=parity_max_abs_error,
            )
            result_path = os.path.join(output_dir, "quantization_result.json")
            with open(result_path, "w") as f:
                f.write(result.model_dump_json(indent=2))
            typer.echo(result.model_dump_json(indent=2))
            typer.echo(f"Quantization result written to: {result_path}")
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

    app()
