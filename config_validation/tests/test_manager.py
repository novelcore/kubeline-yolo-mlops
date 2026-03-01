import pytest

from app.manager import Manager
from app.services.config_validation import ConfigValidationError

VALID_CONFIG_DICT: dict = {
    "experiment": {"name": "spacecraft-pose-v1-yolov8n", "description": "Baseline run"},
    "dataset": {"version": "v1", "source": "s3", "sample_size": None, "seed": 42},
    "model": {"variant": "yolov8n-pose.pt", "pretrained_weights": None},
    "training": {
        "epochs": 100,
        "batch_size": 16,
        "image_size": 640,
        "learning_rate": 0.01,
        "optimizer": "SGD",
    },
    "checkpointing": {
        "interval_epochs": 10,
        "storage_path": "s3://io-mlops/checkpoints",
        "resume_from": None,
    },
    "early_stopping": {"patience": 50},
}


def test_manager_run_calls_service_with_correct_args(mocker):
    """Manager.run forwards config_dict and output_path to the service."""
    mock_service_cls = mocker.patch("app.manager.ConfigValidationService")
    mock_service = mock_service_cls.return_value

    manager = Manager()
    manager.run(config_dict=VALID_CONFIG_DICT, output_path="/fake/out.json")

    mock_service.run.assert_called_once_with(
        config_dict=VALID_CONFIG_DICT,
        output_path="/fake/out.json",
    )


def test_manager_passes_config_fields_to_service_constructor(mocker):
    """Manager passes the correct Config fields to ConfigValidationService."""
    mock_service_cls = mocker.patch("app.manager.ConfigValidationService")

    Manager()

    mock_service_cls.assert_called_once_with(
        skip_liveness_checks=False,
        max_retries=3,
        timeout=30,
        mlflow_tracking_uri=None,
    )


def test_manager_run_raises_on_service_failure(mocker):
    """Exceptions from the service propagate out of Manager.run."""
    mock_service_cls = mocker.patch("app.manager.ConfigValidationService")
    mock_service_cls.return_value.run.side_effect = ConfigValidationError("bad config")

    manager = Manager()
    with pytest.raises(ConfigValidationError, match="bad config"):
        manager.run(config_dict=VALID_CONFIG_DICT)


def test_manager_output_path_defaults_to_none(mocker):
    """Calling manager.run without output_path passes None to the service."""
    mock_service_cls = mocker.patch("app.manager.ConfigValidationService")
    mock_service = mock_service_cls.return_value

    manager = Manager()
    manager.run(config_dict=VALID_CONFIG_DICT)

    mock_service.run.assert_called_once_with(
        config_dict=VALID_CONFIG_DICT,
        output_path=None,
    )
