from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class RegistrationParams(BaseModel):
    """Parameters for the MLflow model registration operation."""

    mlflow_run_id: str = Field(
        description="MLflow run ID produced by the training step"
    )
    best_checkpoint_path: str = Field(description="S3 URI of the best.pt checkpoint")
    last_checkpoint_path: Optional[str] = Field(
        default=None,
        description="S3 URI of the last.pt checkpoint; derived from best_checkpoint_path if omitted",
    )
    registered_model_name: str = Field(
        description="Name under which to register the model in the MLflow model registry"
    )
    promote_to: Optional[Literal["champion", "challenger"]] = Field(
        default=None,
        description=(
            "Registry alias to assign to the best.pt version after registration "
            "(champion or challenger); None/unset assigns no alias"
        ),
    )

    # Lineage tags — all optional; passed through to MLflow model version tags
    dataset_version: Optional[str] = Field(default=None)
    dataset_sample_size: Optional[int] = Field(default=None)
    config_hash: Optional[str] = Field(
        default=None, description="SHA-256 of the experiment YAML"
    )
    git_commit: Optional[str] = Field(
        default=None, description="Git commit hash at registration time"
    )
    model_variant: Optional[str] = Field(
        default=None, description="YOLO model variant, e.g. yolov8n-pose.pt"
    )
    best_map50: Optional[float] = Field(
        default=None, description="Best validation mAP50 from training"
    )
    exported_models: Optional[dict[str, str]] = Field(
        default=None,
        description=(
            "Map of export labels to S3 URIs from the training step, "
            "e.g. {'engine_fp16': 's3://...', 'onnx_fp16': 's3://...'}."
        ),
    )


class RegistrationResult(BaseModel):
    """Outcome of a completed model registration."""

    registered_model_name: str
    best_version: int = Field(description="MLflow model version number for best.pt")
    last_version: Optional[int] = Field(
        default=None,
        description="MLflow model version number for last.pt; None when last.pt was skipped",
    )
    registered_at: datetime
    promoted_to: Optional[str] = Field(
        default=None,
        description="Registry alias assigned to the best.pt version, if any",
    )
