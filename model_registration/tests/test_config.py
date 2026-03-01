import pytest
from app.models.config import Config


def test_config_default_values():
    """Test that Config has correct default values."""
    config = Config()
    assert config.app_name == "io-model-registration"
    assert config.log_level == "INFO"
    assert config.max_retries == 3
    assert config.timeout == 60
    assert config.registry_url == "https://registry.example.com"
    assert config.registry_token == ""


def test_config_with_environment_variables(monkeypatch):
    """Test that Config can be configured via environment variables."""
    monkeypatch.setenv("APP_NAME", "test-registry")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("REGISTRY_URL", "https://custom-registry.io")
    monkeypatch.setenv("REGISTRY_TOKEN", "secret-token")

    config = Config()
    assert config.app_name == "test-registry"
    assert config.log_level == "DEBUG"
    assert config.registry_url == "https://custom-registry.io"
    assert config.registry_token == "secret-token"
