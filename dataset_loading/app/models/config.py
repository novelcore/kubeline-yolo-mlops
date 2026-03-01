from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Config(BaseSettings):
    """Application configuration using environment variables."""

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Application settings
    app_name: str = Field(default="io-dataset-loading", description="Application name")
    log_level: str = Field(default="INFO", description="Logging level")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    timeout: int = Field(default=120, description="Timeout in seconds")

    # Dataset settings
    shuffle_seed: int = Field(default=42, description="Random seed for shuffling")
