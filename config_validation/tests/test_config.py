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
