"""Domain models for the YOLO dataset loading step."""

from typing import Optional

from pydantic import BaseModel, Field


class YoloDatasetParams(BaseModel):
    """Parameters passed into the dataset loading step from the CLI / orchestrator."""

    version: str = Field(description="Dataset version tag (e.g. 'v1').")
    source: str = Field(
        pattern=r"^(s3|lakefs)$",
        description="Storage backend: 's3' or 'lakefs'.",
    )
    path_override: Optional[str] = Field(
        default=None,
        description=(
            "Full S3 URI override (e.g. 's3://my-bucket/path/'). "
            "When set, the conventional bucket/prefix is ignored."
        ),
    )
    sample_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="If set, keep only this many image+label pairs per split.",
    )
    seed: int = Field(
        default=42,
        description="Random seed used for reproducible sampling.",
    )
    output_dir: str = Field(description="Local directory where the dataset is written.")


class YoloDatasetStats(BaseModel):
    """Statistics artifact written to {output_dir}/dataset_stats.json after a run."""

    version: str = Field(description="Dataset version tag.")
    source: str = Field(description="Storage backend used: 's3' or 'lakefs'.")
    train_images: int = Field(ge=0, description="Number of images in the train split.")
    val_images: int = Field(ge=0, description="Number of images in the val split.")
    test_images: int = Field(ge=0, description="Number of images in the test split.")
    sampled: bool = Field(description="True when the dataset was sub-sampled.")
    sample_size: Optional[int] = Field(
        default=None,
        description="The requested sample size (None when not sampled).",
    )
    seed: int = Field(description="Random seed used during sampling.")
