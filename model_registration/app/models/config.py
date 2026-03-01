from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Config(BaseSettings):
    """Application configuration using environment variables."""

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Application settings
    app_name: str = Field(default="io-model-registration", description="Application name")
    log_level: str = Field(default="INFO", description="Logging level")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    timeout: int = Field(default=60, description="Timeout in seconds")

    # Registry settings
    registry_url: str = Field(default="https://registry.example.com", description="Model registry URL")
    registry_token: str = Field(default="", description="Authentication token for the registry")
