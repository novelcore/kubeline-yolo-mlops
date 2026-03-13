import pytest
from app.models.config import Config


def test_config_default_values():
    """Test that Config has correct default values."""
    config = Config()
    assert config.app_name == "io-config-validation"
    assert config.log_level == "INFO"
    assert config.max_retries == 3
    assert config.timeout == 30
    assert config.strict_mode is True


def test_config_with_environment_variables(monkeypatch):
    """Test that Config can be configured via environment variables."""
    monkeypatch.setenv("APP_NAME", "test-app")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("STRICT_MODE", "false")

    config = Config()
    assert config.app_name == "test-app"
    assert config.log_level == "DEBUG"
    assert config.strict_mode is False


def test_config_skip_liveness_checks_default():
    """Test that skip_liveness_checks defaults to False."""
    config = Config()
    assert config.skip_liveness_checks is False


def test_config_skip_liveness_checks_env_override(monkeypatch):
    """Test that SKIP_LIVENESS_CHECKS env var overrides the default."""
    monkeypatch.setenv("SKIP_LIVENESS_CHECKS", "true")
    config = Config()
    assert config.skip_liveness_checks is True


def test_config_mlflow_tracking_uri_default():
    """Test that mlflow_tracking_uri defaults to None."""
    config = Config()
    assert config.mlflow_tracking_uri is None


def test_config_mlflow_tracking_uri_env_override(monkeypatch):
    """Test that MLFLOW_TRACKING_URI env var sets the tracking URI."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.test")
    config = Config()
    assert config.mlflow_tracking_uri == "http://mlflow.test"
