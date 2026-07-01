"""Domain models for the model-quantization step."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class QuantizationParams(BaseModel):
    """Parameters for a model-quantization run."""

    mode: Literal["ptq", "qat"] = Field(
        ...,
        description="Quantization mode. 'none' is handled by DAG gating — never passed here.",
    )

    # ── PTQ inputs (required when mode=ptq) ───────────────────────────────────
    fp32_checkpoint_path: Optional[str] = Field(
        default=None,
        description="Local path or s3:// URI to the FP32 .pt checkpoint. Required for PTQ.",
    )

    # ── QAT inputs (required when mode=qat) ───────────────────────────────────
    tflite_s3_uri: Optional[str] = Field(
        default=None,
        description="S3 URI of the INT8 TFLite produced by qat-finetune. Required for QAT.",
    )
    qat_run_id: Optional[str] = Field(
        default=None,
        description="MLflow run ID from the qat-finetune step (lineage tag).",
    )

    # ── Common ────────────────────────────────────────────────────────────────
    source_mlflow_run_id: str = Field(
        ...,
        description="MLflow run ID of the model-training step (FP32 source lineage).",
    )
    dataset_dir: str = Field(
        ...,
        description="Local path to the YOLO dataset directory. Used for PTQ calibration.",
    )
    output_dir: str = Field(..., description="Local directory for intermediate artifacts.")
    output_bucket: str = Field(..., description="S3 bucket for TFLite artifact upload.")
    output_prefix: str = Field(..., description="S3 key prefix for TFLite artifact.")
    experiment_name: str = Field(..., description="MLflow experiment name.")

    image_size: int = Field(default=640, gt=0)
    calibration_frames: int = Field(default=512, ge=100, le=10000)
    calibration_seed: int = Field(default=42)
    parity_frames: int = Field(default=100, ge=1)
    parity_max_abs_error: float = Field(default=0.05, gt=0.0)

    @model_validator(mode="after")
    def _check_mode_inputs(self) -> "QuantizationParams":
        if self.mode == "ptq" and not self.fp32_checkpoint_path:
            raise ValueError("fp32_checkpoint_path is required when mode='ptq'")
        if self.mode == "qat" and not self.tflite_s3_uri:
            raise ValueError("tflite_s3_uri is required when mode='qat'")
        return self


class ParityReport(BaseModel):
    """Result of the FP32 vs INT8 parity check (FR-M-03)."""

    parity_passed: bool
    max_abs_error: float
    threshold: float
    frames_tested: int


class QuantizationResult(BaseModel):
    """Result of a model-quantization run."""

    mlflow_run_id: str
    source_run_id: str
    mode: str
    tflite_s3_uri: str
    parity_passed: bool = Field(
        default=True,
        description="Parity check result.",
    )
    parity_max_abs_error: float = Field(
        default=0.0,
        description="Observed max-abs-error from parity check.",
    )
