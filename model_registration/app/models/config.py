from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Application configuration for the model registration step."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = Field(
        default="io-model-registration", description="Application name"
    )
    log_level: str = Field(default="INFO", description="Logging level")
    max_retries: int = Field(
        default=3, description="Maximum retry attempts for MLflow calls"
    )
    timeout: int = Field(default=60, description="Timeout in seconds for remote calls")

    mlflow_tracking_uri: str = Field(
        description="URI of the MLflow tracking server",
    )
    mlflow_experiment_name: str = Field(
        default="infinite-orbits",
        description="MLflow experiment name",
    )
    registered_model_name: str = Field(
        default="spacecraft-pose-yolo",
        description="Name under which the model is registered in the MLflow model registry",
    )
