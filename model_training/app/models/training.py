"""Domain models for the YOLO model training step."""

from typing import Optional

from pydantic import BaseModel, Field


class AugmentationParams(BaseModel):
    """Ultralytics augmentation hyperparameters."""

    hsv_h: float = Field(default=0.015, ge=0.0, le=1.0)
    hsv_s: float = Field(default=0.7, ge=0.0, le=1.0)
    hsv_v: float = Field(default=0.4, ge=0.0, le=1.0)
    degrees: float = Field(default=0.0, ge=0.0)
    translate: float = Field(default=0.1, ge=0.0)
    scale: float = Field(default=0.5, ge=0.0)
    shear: float = Field(default=0.0, ge=0.0)
    perspective: float = Field(default=0.0, ge=0.0, le=0.001)
    flipud: float = Field(default=0.0, ge=0.0, le=1.0)
    fliplr: float = Field(default=0.0, ge=0.0, le=1.0)
    mosaic: float = Field(default=1.0, ge=0.0, le=1.0)
    mixup: float = Field(default=0.0, ge=0.0, le=1.0)
    copy_paste: float = Field(default=0.0, ge=0.0, le=1.0)
    erasing: float = Field(default=0.4, ge=0.0, le=0.9)
    bgr: float = Field(default=0.0, ge=0.0, le=1.0)


class TrainingParams(BaseModel):
    """All parameters required to execute a YOLO training run."""

    # ---- Identity ----
    model_variant: str = Field(
        description="YOLO model variant file, e.g. 'yolov8n-pose.pt'."
    )
    experiment_name: str = Field(
        description="MLflow experiment name and S3 checkpoint prefix."
    )
    dataset_dir: str = Field(
        description="Local directory containing the downloaded YOLO dataset."
    )
    output_dir: str = Field(
        description="Local directory for Ultralytics runs/ output."
    )

    # ---- Dataset source ----
    source: str = Field(
        pattern=r"^(local|s3)$",
        description=(
            "Dataset read mode: 'local' reads from dataset_dir, "
            "'s3' streams images directly from S3."
        ),
    )
    s3_bucket: Optional[str] = Field(
        default=None,
        description="S3 bucket containing images (required when source='s3').",
    )
    s3_prefix: Optional[str] = Field(
        default=None,
        description="S3 key prefix for images (required when source='s3').",
    )

    # ---- Weight initialisation / resume ----
    pretrained_weights: Optional[str] = Field(
        default=None,
        description=(
            "Local path or s3:// URI to a custom .pt file. "
            "Loads weights only; training starts from epoch 0."
        ),
    )
    resume_from: Optional[str] = Field(
        default=None,
        description=(
            "'auto', local path, or s3:// URI for full Ultralytics resume "
            "(restores epoch counter, optimizer, and scheduler state)."
        ),
    )

    # ---- Core schedule ----
    epochs: int = Field(default=100, gt=0)
    batch_size: int = Field(default=16, gt=0)
    image_size: int = Field(default=640, gt=0)

    # ---- Learning rate ----
    learning_rate: float = Field(default=0.01, gt=0.0)
    cos_lr: bool = Field(default=True)
    lrf: float = Field(default=0.01, gt=0.0, le=1.0)

    # ---- Optimizer ----
    optimizer: str = Field(default="SGD", pattern=r"^(SGD|Adam|AdamW)$")
    momentum: float = Field(default=0.937, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0005, ge=0.0)

    # ---- Warmup ----
    warmup_epochs: float = Field(default=3.0, ge=0.0)
    warmup_momentum: float = Field(default=0.8, ge=0.0, lt=1.0)

    # ---- Regularization ----
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)

    # ---- Gradient accumulation ----
    nbs: int = Field(default=64, gt=0)

    # ---- Fine-tuning control ----
    freeze: Optional[int] = Field(default=None, ge=0)

    # ---- Training efficiency ----
    amp: bool = Field(default=True)
    close_mosaic: int = Field(default=10, ge=0)
    seed: int = Field(default=0, ge=0)
    deterministic: bool = Field(default=True)

    # ---- Pose-estimation loss gains ----
    pose: float = Field(default=12.0, gt=0.0)
    kobj: float = Field(default=2.0, gt=0.0)
    box: float = Field(default=7.5, gt=0.0)
    cls: float = Field(default=0.5, gt=0.0)
    dfl: float = Field(default=1.5, gt=0.0)

    # ---- Early stopping ----
    patience: int = Field(default=50, gt=0)

    # ---- S3 streaming cache ----
    disk_cache_bytes: int = Field(
        default=2 * 1024**3,
        gt=0,
        description="Maximum disk budget (bytes) for the S3 image streaming cache.",
    )

    # ---- Checkpointing ----
    checkpoint_interval: int = Field(default=10, gt=0)
    checkpoint_bucket: str = Field(default="temp-mlops")
    checkpoint_prefix: str = Field(default="checkpoints")

    # ---- Augmentation ----
    augmentation: AugmentationParams = Field(default_factory=AugmentationParams)


class TrainingResult(BaseModel):
    """Outcome of a completed training run."""

    experiment_name: str
    model_variant: str
    mlflow_run_id: str
    best_checkpoint_local: str = Field(description="Local path to best.pt.")
    best_checkpoint_s3: str = Field(description="S3 URI where best.pt was uploaded.")
    epochs_completed: int
    final_map50: float
    final_map50_95: float
