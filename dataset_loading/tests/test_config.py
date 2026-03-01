"""Tests for the Config model."""

import pytest

from app.models.config import Config


def test_config_default_values() -> None:
    """Config has the correct default values when no env vars are set."""
    config = Config()
    assert config.app_name == "io-dataset-loading"
    assert config.log_level == "INFO"
    assert config.max_retries == 3
    assert config.timeout == 120
    # Storage fields default to None
    assert config.aws_default_region is None
    assert config.aws_access_key_id is None
    assert config.aws_secret_access_key is None
    assert config.s3_endpoint_url is None
    assert config.lakefs_endpoint is None
    assert config.lakefs_access_key is None
    assert config.lakefs_secret_key is None


def test_config_reads_app_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config picks up APP_NAME and LOG_LEVEL from environment variables."""
    monkeypatch.setenv("APP_NAME", "test-dataset-loading")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MAX_RETRIES", "5")
    monkeypatch.setenv("TIMEOUT", "60")

    config = Config()
    assert config.app_name == "test-dataset-loading"
    assert config.log_level == "DEBUG"
    assert config.max_retries == 5
    assert config.timeout == 60


def test_config_reads_aws_credentials_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config reads AWS credential fields from environment variables."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
    monkeypatch.setenv(
        "AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    )
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://minio.example.com")

    config = Config()
    assert config.aws_default_region == "us-east-1"
    assert config.aws_access_key_id == "AKIAIOSFODNN7EXAMPLE"
    assert config.aws_secret_access_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    assert config.s3_endpoint_url == "https://minio.example.com"


def test_config_reads_lakefs_credentials_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config reads LakeFS credential fields from environment variables."""
    monkeypatch.setenv("LAKEFS_ENDPOINT", "https://lakefs.example.com")
    monkeypatch.setenv("LAKEFS_ACCESS_KEY", "my-lakefs-access-key")
    monkeypatch.setenv("LAKEFS_SECRET_KEY", "my-lakefs-secret-key")

    config = Config()
    assert config.lakefs_endpoint == "https://lakefs.example.com"
    assert config.lakefs_access_key == "my-lakefs-access-key"
    assert config.lakefs_secret_key == "my-lakefs-secret-key"
