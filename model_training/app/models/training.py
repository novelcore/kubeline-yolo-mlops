from pydantic import BaseModel, Field
from typing import Optional


class TrainingParams(BaseModel):
    """Parameters for model training."""
    model_name: str
    dataset_dir: str
    output_dir: str
    epochs: int = Field(ge=1, le=1000, default=10)
    batch_size: int = Field(ge=1, default=32)
    learning_rate: float = Field(gt=0.0, default=1e-4)
    optimizer: str = Field(default="adamw")
    seed: int = Field(default=42)


class TrainingMetrics(BaseModel):
    """Metrics produced by a training run."""
    model_name: str
    epochs_completed: int
    final_train_loss: float
    final_val_loss: Optional[float] = None
    checkpoint_path: str
