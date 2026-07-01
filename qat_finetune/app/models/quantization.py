"""Domain models for the QAT fine-tune step."""

from typing import Optional

from pydantic import BaseModel, Field


class QATParams(BaseModel):
    """Parameters for a PT2E QAT fine-tune run."""

    # ── Source ────────────────────────────────────────────────────────────────
    fp32_checkpoint_path: str = Field(
        ...,
        description="Local path or s3:// URI to the FP32 .pt checkpoint from model-training.",
    )
    source_mlflow_run_id: str = Field(
        ...,
        description="MLflow run ID of the model-training run that produced the checkpoint.",
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_dir: str = Field(
        ...,
        description="Local path to the YOLO dataset directory (images/ and labels/ subdirs).",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = Field(
        ...,
        description="Local directory for intermediate artifacts (downloaded checkpoint, TFLite).",
    )
    output_bucket: str = Field(
        ...,
        description="S3 bucket for TFLite artifact upload.",
    )
    output_prefix: str = Field(
        ...,
        description="S3 key prefix for TFLite artifact (e.g. 'qat/exp-001').",
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    experiment_name: str = Field(
        ...,
        description="MLflow experiment name for this QAT run.",
    )

    # ── Model / training ──────────────────────────────────────────────────────
    image_size: int = Field(
        default=640,
        gt=0,
        description="Input image size (square). Must match the training image_size.",
    )
    device: Optional[str] = Field(
        default=None,
        description="Compute device: 'cuda', 'cpu', '0'. Defaults to cuda if available.",
    )

    # ── QAT hyperparameters (from pipeline_config.yaml quantization: section) ─
    qat_epochs: int = Field(
        default=10,
        gt=0,
        description="Number of QAT fine-tune epochs.",
    )
    qat_lr: float = Field(
        default=1e-4,
        gt=0.0,
        description="Learning rate for QAT fine-tuning optimizer.",
    )
    calibration_frames: int = Field(
        default=512,
        ge=100,
        le=10000,
        description="Number of calibration frames sampled from the training set.",
    )
    calibration_seed: int = Field(
        default=42,
        description="RNG seed for calibration frame sampling (determinism, FR-M-04).",
    )
    parity_frames: int = Field(
        default=100,
        ge=1,
        description="Number of frames used in the parity check (FR-M-03).",
    )
    parity_max_abs_error: float = Field(
        default=0.05,
        gt=0.0,
        description="Maximum allowed max-abs-error between INT8 and FP32 outputs.",
    )


class QATResult(BaseModel):
    """Result of a PT2E QAT fine-tune run."""

    mlflow_run_id: str = Field(
        ...,
        description="MLflow run ID created for this QAT step.",
    )
    source_run_id: str = Field(
        ...,
        description="MLflow run ID of the model-training step (lineage).",
    )
    tflite_s3_uri: str = Field(
        ...,
        description="S3 URI of the produced INT8 TFLite artifact.",
    )
    parity_passed: bool = Field(
        ...,
        description="True if the parity check passed within parity_max_abs_error.",
    )
    parity_max_abs_error: float = Field(
        ...,
        description="Observed max-abs-error between INT8 TFLite and FP32 outputs.",
    )
