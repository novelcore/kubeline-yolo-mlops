import pytest
import yaml
from pathlib import Path

from app.services.config_validation import ConfigValidationService, ConfigValidationError


VALID_CONFIG = {
    "pipeline_name": "test-pipeline",
    "version": "1.0.0",
    "dataset": {
        "source_path": "/data/train.parquet",
        "format": "parquet",
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
    },
    "training": {
        "model_name": "bert-base",
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "optimizer": "adamw",
        "seed": 42,
        "output_dir": "/artifacts/checkpoints",
    },
    "registration": {
        "registry_url": "https://registry.example.com",
        "model_name": "bert-base-finetuned",
        "tags": ["v1", "experiment"],
        "promote_to": "staging",
    },
}


@pytest.fixture
def valid_config_file(tmp_path: Path) -> Path:
    config_path = tmp_path / "pipeline_config.yaml"
    config_path.write_text(yaml.dump(VALID_CONFIG))
    return config_path


@pytest.fixture
def service() -> ConfigValidationService:
    return ConfigValidationService()


def test_valid_config_passes(service, valid_config_file, capfd):
    """Test that a valid config passes validation."""
    result = service.run(str(valid_config_file))
    out, _ = capfd.readouterr()
    assert "Config validation passed" in out
    assert result.pipeline_name == "test-pipeline"


def test_missing_file_raises_error(service):
    """Test that a missing config file raises an error."""
    with pytest.raises(ConfigValidationError, match="Config file not found"):
        service.run("/nonexistent/config.yaml")


def test_invalid_format_raises_error(service, tmp_path):
    """Test that a non-YAML file raises an error."""
    bad_file = tmp_path / "config.txt"
    bad_file.write_text("not yaml")
    with pytest.raises(ConfigValidationError, match="Config file must be YAML"):
        service.run(str(bad_file))


def test_invalid_schema_raises_error(service, tmp_path):
    """Test that an invalid schema raises an error."""
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(yaml.dump({"pipeline_name": "test"}))
    with pytest.raises(ConfigValidationError, match="Schema validation failed"):
        service.run(str(bad_config))


def test_bad_splits_strict_raises_error(service, tmp_path):
    """Test that splits not summing to 1.0 raises error in strict mode."""
    config = VALID_CONFIG.copy()
    config["dataset"] = {
        **VALID_CONFIG["dataset"],
        "train_split": 0.5,
        "val_split": 0.1,
        "test_split": 0.1,
    }
    config_path = tmp_path / "bad_splits.yaml"
    config_path.write_text(yaml.dump(config))

    with pytest.raises(ConfigValidationError, match="splits must sum to 1.0"):
        service.run(str(config_path), strict=True)
