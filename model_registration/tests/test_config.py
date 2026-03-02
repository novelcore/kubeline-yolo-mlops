"""Tests for the Config pydantic-settings model."""

import pytest

from app.models.config import Config


def test_config_default_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    config = Config()

    assert config.app_name == "io-model-registration"
    assert config.log_level == "INFO"
    assert config.max_retries == 3
    assert config.timeout == 60
    assert config.mlflow_experiment_name == "infinite-orbits"
    assert config.registered_model_name == "spacecraft-pose-yolo"


def test_config_reads_mlflow_tracking_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.prod.example.com:5000")

    config = Config()

    assert config.mlflow_tracking_uri == "http://mlflow.prod.example.com:5000"


def test_config_reads_mlflow_experiment_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "spacecraft-pose-v2")

    config = Config()

    assert config.mlflow_experiment_name == "spacecraft-pose-v2"


def test_config_reads_registered_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("REGISTERED_MODEL_NAME", "custom-yolo-model")

    config = Config()

    assert config.registered_model_name == "custom-yolo-model"


def test_config_reads_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    config = Config()

    assert config.log_level == "DEBUG"


def test_config_reads_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MAX_RETRIES", "5")

    config = Config()

    assert config.max_retries == 5


def test_config_missing_mlflow_tracking_uri_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with pytest.raises(Exception):
        Config()
