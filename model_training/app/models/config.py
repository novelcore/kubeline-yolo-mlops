from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Config(BaseSettings):
    """Application configuration using environment variables."""

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Application settings
    app_name: str = Field(default="io-model-training", description="Application name")
    log_level: str = Field(default="INFO", description="Logging level")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    timeout: int = Field(default=3600, description="Timeout in seconds")

    # Training settings
    default_seed: int = Field(default=42, description="Default random seed")
    device: str = Field(default="cpu", description="Training device (cpu, cuda, mps)")
