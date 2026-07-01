"""Application configuration for the model quantization step."""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = Field(default="io-model-quantization")
    log_level: str = Field(default="INFO")

    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_tracking_username: Optional[str] = Field(default=None)
    mlflow_tracking_password: Optional[str] = Field(default=None)

    aws_default_region: Optional[str] = Field(default=None)
    aws_access_key_id: Optional[str] = Field(default=None)
    aws_secret_access_key: Optional[str] = Field(default=None)
    s3_endpoint_url: Optional[str] = Field(default=None)

    lakefs_endpoint: Optional[str] = Field(default=None)
    lakefs_access_key: Optional[str] = Field(default=None)
    lakefs_secret_key: Optional[str] = Field(default=None)
