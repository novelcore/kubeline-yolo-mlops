from typing import Optional

from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Application configuration using environment variables."""

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Application settings
    app_name: str = Field(default="io-config-validation", description="Application name")
    log_level: str = Field(default="INFO", description="Logging level")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    timeout: int = Field(default=30, description="Timeout in seconds")

    # Validation settings
    strict_mode: bool = Field(default=True, description="Enable strict validation mode")

    # Liveness check settings
    skip_liveness_checks: bool = Field(
        default=False,
        description="Skip S3, weights, and MLflow liveness checks. Use for local runs or CI.",
    )
    mlflow_tracking_uri: Optional[str] = Field(
        default=None,
        description="MLflow tracking server URI. Required when liveness checks are enabled.",
    )

    # AWS / S3 connection settings
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key ID")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS secret access key")
    aws_default_region: Optional[str] = Field(default=None, description="AWS region")
    s3_endpoint_url: Optional[str] = Field(default=None, description="Custom S3 endpoint URL")

    # LakeFS connection settings (S3-compatible API)
    lakefs_endpoint: Optional[str] = Field(default=None, description="LakeFS S3-compatible endpoint URL")
    lakefs_access_key: Optional[str] = Field(default=None, description="LakeFS access key ID")
    lakefs_secret_key: Optional[str] = Field(default=None, description="LakeFS secret access key")
