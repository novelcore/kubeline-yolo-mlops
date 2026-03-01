from pydantic import BaseModel, Field


class DatasetParams(BaseModel):
    """Parameters for dataset loading and splitting."""
    source_path: str
    output_dir: str
    format: str = Field(pattern=r"^(csv|parquet|jsonl|hf)$")
    train_split: float = Field(ge=0.0, le=1.0, default=0.8)
    val_split: float = Field(ge=0.0, le=1.0, default=0.1)
    test_split: float = Field(ge=0.0, le=1.0, default=0.1)


class DatasetSplitResult(BaseModel):
    """Result of a dataset loading and splitting operation."""
    total_records: int
    train_records: int
    val_records: int
    test_records: int
    output_dir: str
