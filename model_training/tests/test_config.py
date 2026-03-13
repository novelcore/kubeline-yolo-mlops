"""Tests for the model_training Config (Pydantic BaseSettings)."""

import pytest

from app.models.config import Config


def test_config_default_values() -> None:
    """Config should provide sensible defaults for all fields."""
    config = Config()
    assert config.app_name == "io-model-training"
    assert config.log_level == "INFO"
    assert config.mlflow_tracking_uri == "http://localhost:5000"
    # Optional S3/LakeFS fields default to None
    assert config.aws_access_key_id is None
    assert config.aws_secret_access_key is None
    assert config.aws_default_region is None
    assert config.s3_endpoint_url is None
    assert config.lakefs_endpoint is None
    assert config.lakefs_access_key is None
    assert config.lakefs_secret_key is None


def test_config_reads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config should pick up all supported environment variables."""
    monkeypatch.setenv("APP_NAME", "test-trainer")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.test:5000")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://minio.test")
    config = Config()

    assert config.app_name == "test-trainer"
    assert config.log_level == "DEBUG"
    assert config.mlflow_tracking_uri == "http://mlflow.test:5000"
    assert config.aws_default_region == "eu-central-1"
    assert config.aws_access_key_id == "test-key"
    assert config.aws_secret_access_key == "test-secret"
    assert config.s3_endpoint_url == "https://minio.test"


def test_config_lakefs_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """LakeFS credentials are read from env vars."""
    monkeypatch.setenv("LAKEFS_ENDPOINT", "https://lakefs.test")
    monkeypatch.setenv("LAKEFS_ACCESS_KEY", "lf-key")
    monkeypatch.setenv("LAKEFS_SECRET_KEY", "lf-secret")

    config = Config()

    assert config.lakefs_endpoint == "https://lakefs.test"
    assert config.lakefs_access_key == "lf-key"
    assert config.lakefs_secret_key == "lf-secret"
