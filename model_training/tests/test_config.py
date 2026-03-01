import pytest
from app.models.config import Config


def test_config_default_values():
    """Test that Config has correct default values."""
    config = Config()
    assert config.app_name == "io-model-training"
    assert config.log_level == "INFO"
    assert config.max_retries == 3
    assert config.timeout == 3600
    assert config.default_seed == 42
    assert config.device == "cpu"


def test_config_with_environment_variables(monkeypatch):
    """Test that Config can be configured via environment variables."""
    monkeypatch.setenv("APP_NAME", "test-trainer")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DEVICE", "cuda")

    config = Config()
    assert config.app_name == "test-trainer"
    assert config.log_level == "DEBUG"
    assert config.device == "cuda"
