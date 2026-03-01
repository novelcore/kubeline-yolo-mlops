from pydantic import BaseModel, Field
from typing import Optional


class DatasetConfig(BaseModel):
    """Dataset configuration block."""
    source_path: str
    format: str = Field(pattern=r"^(csv|parquet|jsonl|hf)$")
    train_split: float = Field(ge=0.0, le=1.0, default=0.8)
    val_split: float = Field(ge=0.0, le=1.0, default=0.1)
    test_split: float = Field(ge=0.0, le=1.0, default=0.1)


class TrainingConfig(BaseModel):
    """Training configuration block."""
    model_name: str
    epochs: int = Field(ge=1, le=1000, default=10)
    batch_size: int = Field(ge=1, default=32)
    learning_rate: float = Field(gt=0.0, default=1e-4)
    optimizer: str = Field(default="adamw")
    seed: int = Field(default=42)
    output_dir: str = Field(default="/artifacts/checkpoints")


class RegistrationConfig(BaseModel):
    """Model registration configuration block."""
    registry_url: str
    model_name: str
    tags: list[str] = Field(default_factory=list)
    promote_to: Optional[str] = Field(default=None, pattern=r"^(staging|production)$")


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration validated by this step."""
    pipeline_name: str
    version: str
    dataset: DatasetConfig
    training: TrainingConfig
    registration: RegistrationConfig
