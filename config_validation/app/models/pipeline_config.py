import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Valid YOLO Pose model variants (v8–v11, sizes n/s/m/l/x)
_YOLO_POSE_VARIANT_PATTERN = re.compile(r"^yolov(8|9|10|11)[nsmlx]-pose\.pt$")


class ExperimentConfig(BaseModel):
    name: str
    description: Optional[str] = None
    tags: Optional[dict[str, str]] = None

    @field_validator("name", mode="after")
    @classmethod
    def name_must_be_valid(cls, v: str) -> str:
        if not v:
            raise ValueError("experiment.name must not be empty")
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9\-]*$", v):
            raise ValueError(
                f"experiment.name must contain only alphanumeric characters and hyphens "
                f"and must not start with a hyphen, got: {v!r}"
            )
        return v


class DatasetConfig(BaseModel):
    version: str
    source: str
    path_override: Optional[str] = None
    sample_size: Optional[int] = None
    seed: int = 42

    @field_validator("version", mode="after")
    @classmethod
    def version_must_be_non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("dataset.version must not be empty")
        return v

    @field_validator("source", mode="after")
    @classmethod
    def source_must_be_valid(cls, v: str) -> str:
        if v not in ("s3", "lakefs"):
            raise ValueError(
                f"dataset.source must be 's3' or 'lakefs', got: {v!r}"
            )
        return v

    @field_validator("sample_size", mode="after")
    @classmethod
    def sample_size_must_be_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError(
                f"dataset.sample_size must be > 0 when set, got: {v}"
            )
        return v


class ModelConfig(BaseModel):
    variant: str
    pretrained_weights: Optional[str] = None

    @field_validator("variant", mode="after")
    @classmethod
    def variant_must_be_valid_yolo_pose(cls, v: str) -> str:
        if not _YOLO_POSE_VARIANT_PATTERN.match(v):
            raise ValueError(
                f"model.variant {v!r} is not a valid YOLO Pose variant. "
                f"Expected format: yolov{{8|9|10|11}}{{n|s|m|l|x}}-pose.pt "
                f"(e.g. yolov8n-pose.pt, yolov11x-pose.pt)"
            )
        return v


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")  # silently drops training.scheduler

    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    image_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    optimizer: str
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    weight_decay: float = 0.0005

    @field_validator("image_size", mode="after")
    @classmethod
    def image_size_must_be_multiple_of_32(cls, v: int) -> int:
        if v % 32 != 0:
            raise ValueError(
                f"training.image_size must be a multiple of 32, got: {v}"
            )
        return v

    @field_validator("optimizer", mode="after")
    @classmethod
    def optimizer_must_be_valid(cls, v: str) -> str:
        if v not in ("SGD", "Adam", "AdamW"):
            raise ValueError(
                f"training.optimizer must be one of 'SGD', 'Adam', 'AdamW', got: {v!r}"
            )
        return v


class CheckpointingConfig(BaseModel):
    interval_epochs: int = Field(gt=0)
    storage_path: str
    resume_from: Optional[str] = None

    @field_validator("storage_path", mode="after")
    @classmethod
    def storage_path_must_be_cloud_uri(cls, v: str) -> str:
        if not (v.startswith("s3://") or v.startswith("lakefs://")):
            raise ValueError(
                f"checkpointing.storage_path must start with 's3://' or 'lakefs://', got: {v!r}"
            )
        return v

    @field_validator("resume_from", mode="after")
    @classmethod
    def resume_from_must_be_valid(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "auto":
            return v
        if v.startswith("s3://"):
            return v
        raise ValueError(
            f"checkpointing.resume_from must be null, 'auto', or an S3 path starting with "
            f"'s3://', got: {v!r}"
        )


class EarlyStoppingConfig(BaseModel):
    patience: int = Field(gt=0)


class AugmentationConfig(BaseModel):
    hsv_h: float = Field(default=0.015, ge=0.0, le=1.0)
    hsv_s: float = Field(default=0.7, ge=0.0, le=1.0)
    hsv_v: float = Field(default=0.4, ge=0.0, le=1.0)
    degrees: float = Field(default=0.0, ge=0.0)
    translate: float = Field(default=0.1, ge=0.0)
    scale: float = Field(default=0.5, ge=0.0)
    flipud: float = Field(default=0.0, ge=0.0, le=1.0)
    fliplr: float = Field(default=0.0, ge=0.0, le=1.0)
    mosaic: float = Field(default=1.0, ge=0.0, le=1.0)
    mixup: float = Field(default=0.0, ge=0.0, le=1.0)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")  # silently drops resources section

    experiment: ExperimentConfig
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    checkpointing: CheckpointingConfig
    early_stopping: EarlyStoppingConfig
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)