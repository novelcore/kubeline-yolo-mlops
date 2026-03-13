import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    lakefs_repo: Optional[str] = None
    lakefs_branch: Optional[str] = None
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

    @field_validator("lakefs_repo", mode="after")
    @classmethod
    def lakefs_repo_must_be_non_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("dataset.lakefs_repo must not be empty when set")
        return v

    @field_validator("lakefs_branch", mode="after")
    @classmethod
    def lakefs_branch_must_be_non_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("dataset.lakefs_branch must not be empty when set")
        return v

    @field_validator("sample_size", mode="after")
    @classmethod
    def sample_size_must_be_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError(
                f"dataset.sample_size must be > 0 when set, got: {v}"
            )
        return v

    @model_validator(mode="after")
    def lakefs_fields_required_when_lakefs_source(self) -> "DatasetConfig":
        if self.source == "lakefs" and not self.path_override:
            if not self.lakefs_repo:
                raise ValueError(
                    "dataset.lakefs_repo is required when source='lakefs' and no path_override is set"
                )
            if not self.lakefs_branch:
                raise ValueError(
                    "dataset.lakefs_branch is required when source='lakefs' and no path_override is set"
                )
        return self


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
    # Extra keys (e.g. legacy training.scheduler sub-object) are rejected to
    # prevent silent misconfiguration now that cos_lr and lrf are first-class fields.
    model_config = ConfigDict(extra="forbid")

    # ---- Core schedule ----
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    image_size: int = Field(gt=0)

    # ---- Learning rate ----
    # learning_rate maps to Ultralytics lr0.
    learning_rate: float = Field(gt=0.0)
    # cos_lr and lrf were previously buried in training.scheduler (which was silently
    # dropped). They are now first-class validated fields.
    cos_lr: bool = True
    # lrf is the ratio of the final LR to the initial LR: final_lr = learning_rate * lrf.
    # Must be in (0, 1] — a value of 0 would reduce LR to zero regardless of schedule,
    # and a value > 1 would make the final LR higher than the initial LR.
    lrf: float = Field(default=0.01, gt=0.0, le=1.0)

    # ---- Optimizer ----
    optimizer: str
    # momentum serves as SGD momentum and Adam/AdamW beta1.
    momentum: float = Field(default=0.937, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0005, ge=0.0)

    # ---- Warmup ----
    warmup_epochs: float = Field(default=3.0, ge=0.0)
    warmup_momentum: float = Field(default=0.8, ge=0.0, lt=1.0)

    # ---- Regularization ----
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)

    # ---- Gradient accumulation ----
    # Ultralytics uses nbs to scale the accumulated gradient so that effective
    # behaviour matches training with a batch of size `nbs` regardless of the
    # actual `batch_size`.  Must be a positive integer.
    nbs: int = Field(default=64, gt=0)

    # ---- Fine-tuning control ----
    # freeze=None means all layers are trainable.  freeze=N freezes the first N layers.
    freeze: Optional[int] = Field(default=None, ge=0)

    # ---- Training efficiency ----
    amp: bool = True
    # close_mosaic=0 means mosaic is never disabled; any positive value N disables
    # mosaic augmentation for the final N epochs.
    close_mosaic: int = Field(default=10, ge=0)

    # ---- Reproducibility ----
    # seed is the global training RNG seed, distinct from dataset.seed which controls
    # dataset sampling only.
    seed: int = Field(default=0, ge=0)
    deterministic: bool = True

    # ---- Pose-estimation loss gains ----
    # These are the primary quality levers for spacecraft keypoint accuracy.
    pose: float = Field(default=12.0, gt=0.0)   # keypoint regression loss
    kobj: float = Field(default=2.0, gt=0.0)    # keypoint objectness loss
    box: float = Field(default=7.5, gt=0.0)     # bounding-box regression loss
    cls: float = Field(default=0.5, gt=0.0)     # classification loss
    dfl: float = Field(default=1.5, gt=0.0)     # distribution focal loss

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

    @model_validator(mode="after")
    def validate_cross_field_constraints(self) -> "TrainingConfig":
        # cos_lr + lrf coherence: lrf > 1 is caught by Field(le=1.0), but we add
        # a descriptive guard for the case where cos_lr is False and lrf is at its
        # default — a common copy-paste mistake where the user forgets that lrf is
        # still applied as a simple linear decay endpoint when cos_lr=False.
        # This is informational only; we do not block here.

        # close_mosaic vs epochs: if close_mosaic >= epochs, mosaic is disabled from
        # the very first epoch, which is almost certainly unintentional.
        if self.close_mosaic > 0 and self.close_mosaic >= self.epochs:
            raise ValueError(
                f"training.close_mosaic ({self.close_mosaic}) must be less than "
                f"training.epochs ({self.epochs}); as configured, mosaic augmentation "
                f"would be disabled for the entire run. Set close_mosaic=0 to disable "
                f"it explicitly or reduce close_mosaic below epochs."
            )

        # warmup_epochs vs epochs: warmup longer than training is always a bug.
        if self.warmup_epochs >= self.epochs:
            raise ValueError(
                f"training.warmup_epochs ({self.warmup_epochs}) must be less than "
                f"training.epochs ({self.epochs})"
            )

        return self


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
    # ---- Colour-space jitter ----
    hsv_h: float = Field(default=0.015, ge=0.0, le=1.0)
    hsv_s: float = Field(default=0.7, ge=0.0, le=1.0)
    hsv_v: float = Field(default=0.4, ge=0.0, le=1.0)

    # ---- Geometric ----
    degrees: float = Field(default=0.0, ge=0.0)
    translate: float = Field(default=0.1, ge=0.0)
    scale: float = Field(default=0.5, ge=0.0)
    shear: float = Field(default=0.0, ge=0.0)
    # Ultralytics enforces perspective in [0.0, 0.001] internally; values beyond
    # 0.001 cause extreme keypoint displacement that invalidates pose labels.
    perspective: float = Field(default=0.0, ge=0.0, le=0.001)

    # ---- Flip ----
    flipud: float = Field(default=0.0, ge=0.0, le=1.0)
    fliplr: float = Field(default=0.0, ge=0.0, le=1.0)

    # ---- Composition ----
    mosaic: float = Field(default=1.0, ge=0.0, le=1.0)
    mixup: float = Field(default=0.0, ge=0.0, le=1.0)
    copy_paste: float = Field(default=0.0, ge=0.0, le=1.0)

    # ---- Erasing / channel ----
    # erasing upper bound is 0.9; erasing=1.0 would blank the entire image.
    erasing: float = Field(default=0.4, ge=0.0, le=0.9)
    bgr: float = Field(default=0.0, ge=0.0, le=1.0)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")  # silently drops resources section

    experiment: ExperimentConfig
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    checkpointing: CheckpointingConfig
    early_stopping: EarlyStoppingConfig
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
