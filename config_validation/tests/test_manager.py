import pytest
import yaml
from pathlib import Path

from app.manager import Manager


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
        "tags": ["v1"],
        "promote_to": "staging",
    },
}


def test_manager_runs_config_validation(tmp_path, capfd):
    """Test that Manager.run validates a config and prints success."""
    config_path = tmp_path / "pipeline_config.yaml"
    config_path.write_text(yaml.dump(VALID_CONFIG))

    manager = Manager()
    manager.run(config_path=str(config_path))
    out, _ = capfd.readouterr()
    assert "Config validation passed" in out
