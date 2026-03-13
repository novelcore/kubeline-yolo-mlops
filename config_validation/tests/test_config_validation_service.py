import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.services.config_validation import ConfigValidationError, ConfigValidationService
from app.models.pipeline_config import PipelineConfig


VALID_CONFIG: dict = {
    "experiment": {
        "name": "spacecraft-pose-v1-yolov8n",
        "description": "Baseline run",
    },
    "dataset": {
        "version": "v1",
        "source": "s3",
        "sample_size": None,
        "seed": 42,
    },
    "model": {
        "variant": "yolov8n-pose.pt",
        "pretrained_weights": None,
    },
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
    "early_stopping": {
        "patience": 50,
    },
}


@pytest.fixture
def mock_s3():
    return MagicMock()


@pytest.fixture
def service_no_liveness(mock_s3) -> ConfigValidationService:
    return ConfigValidationService(
        skip_liveness_checks=True,
        max_retries=3,
        timeout=10,
        mlflow_tracking_uri=None,
        s3_client=mock_s3,
    )


@pytest.fixture
def service_with_liveness(mock_s3) -> ConfigValidationService:
    return ConfigValidationService(
        skip_liveness_checks=False,
        max_retries=1,
        timeout=5,
        mlflow_tracking_uri="http://mlflow.test",
        s3_client=mock_s3,
    )


# ---------------------------------------------------------------------------
# Schema tests (liveness skipped)
# ---------------------------------------------------------------------------

def test_valid_config_returns_pipeline_config(service_no_liveness):
    result = service_no_liveness.run(VALID_CONFIG)
    assert isinstance(result, PipelineConfig)
    assert result.experiment.name == "spacecraft-pose-v1-yolov8n"


def test_valid_config_prints_success(service_no_liveness, capfd):
    service_no_liveness.run(VALID_CONFIG)
    out, _ = capfd.readouterr()
    assert "Config validation passed" in out


def test_invalid_schema_raises_config_validation_error(service_no_liveness):
    bad_config = {"experiment": {"name": ""}}
    with pytest.raises(ConfigValidationError, match="Schema validation failed"):
        service_no_liveness.run(bad_config)


def test_output_written_to_file(service_no_liveness, tmp_path):
    out = tmp_path / "out.json"
    service_no_liveness.run(VALID_CONFIG, output_path=str(out))
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["experiment"]["name"] == "spacecraft-pose-v1-yolov8n"


def test_output_path_parent_dirs_created(service_no_liveness, tmp_path):
    out = tmp_path / "nested" / "dir" / "out.json"
    service_no_liveness.run(VALID_CONFIG, output_path=str(out))
    assert out.exists()


def test_no_output_file_when_output_path_is_none(service_no_liveness, tmp_path):
    service_no_liveness.run(VALID_CONFIG, output_path=None)
    # No file should have been written
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Liveness check tests
# ---------------------------------------------------------------------------

def test_liveness_checks_not_called_when_skipped(service_no_liveness, mock_s3):
    service_no_liveness.run(VALID_CONFIG)
    mock_s3.list_objects_v2.assert_not_called()


def test_dataset_path_check_calls_list_objects(service_with_liveness, mock_s3, mocker):
    mock_s3.list_objects_v2.return_value = {"KeyCount": 1}
    mock_s3.head_object.return_value = {}
    mocker.patch("app.services.config_validation.httpx.get").return_value = MagicMock(is_success=True)

    service_with_liveness.run(VALID_CONFIG)

    calls = mock_s3.list_objects_v2.call_args_list
    assert any(
        "io-mlops" in str(call) and "dataset" in str(call)
        for call in calls
    )


def test_dataset_path_not_found_raises_error(service_with_liveness, mock_s3, mocker):
    mock_s3.list_objects_v2.return_value = {"KeyCount": 0}
    mocker.patch("app.services.config_validation.httpx.get").return_value = MagicMock(is_success=True)

    with pytest.raises(ConfigValidationError, match="not found or empty in S3"):
        service_with_liveness.run(VALID_CONFIG)


def test_mlflow_uri_check_calls_health_endpoint(service_with_liveness, mock_s3, mocker):
    mock_s3.list_objects_v2.return_value = {"KeyCount": 1}
    mock_get = mocker.patch("app.services.config_validation.httpx.get")
    mock_get.return_value = MagicMock(is_success=True)

    service_with_liveness.run(VALID_CONFIG)

    mock_get.assert_called_once()
    called_url = mock_get.call_args[0][0]
    assert called_url.endswith("/health")


def test_mlflow_uri_unreachable_raises_error(service_with_liveness, mock_s3, mocker):
    import httpx

    mock_s3.list_objects_v2.return_value = {"KeyCount": 1}
    mocker.patch(
        "app.services.config_validation.httpx.get",
        side_effect=httpx.RequestError("connection refused"),
    )

    with pytest.raises(ConfigValidationError, match="unreachable"):
        service_with_liveness.run(VALID_CONFIG)


def test_mlflow_uri_not_set_raises_error():
    mock_s3 = MagicMock()
    mock_s3.list_objects_v2.return_value = {"KeyCount": 1}
    svc = ConfigValidationService(
        skip_liveness_checks=False,
        max_retries=1,
        timeout=5,
        mlflow_tracking_uri=None,
        s3_client=mock_s3,
    )

    with pytest.raises(ConfigValidationError, match="MLFLOW_TRACKING_URI"):
        svc.run(VALID_CONFIG)


def test_resume_from_auto_found_passes(mocker):
    from datetime import datetime, timezone

    config = dict(VALID_CONFIG)
    config["checkpointing"] = {**VALID_CONFIG["checkpointing"], "resume_from": "auto"}

    mock_s3 = MagicMock()
    mock_s3.list_objects_v2.return_value = {
        "KeyCount": 1,
        "Contents": [
            {"Key": "spacecraft-pose-v1-yolov8n/last.pt", "LastModified": datetime.now(timezone.utc)}
        ],
    }
    mocker.patch("app.services.config_validation.httpx.get").return_value = MagicMock(is_success=True)

    svc = ConfigValidationService(
        skip_liveness_checks=False, max_retries=1, timeout=5,
        mlflow_tracking_uri="http://mlflow.test", s3_client=mock_s3,
    )
    svc.run(config)  # must not raise


def test_resume_from_auto_not_found_raises_error(mocker):
    config = dict(VALID_CONFIG)
    config["checkpointing"] = {**VALID_CONFIG["checkpointing"], "resume_from": "auto"}

    mock_s3 = MagicMock()
    # First call = dataset path check (passes), second call = checkpoint scan (empty)
    mock_s3.list_objects_v2.side_effect = [
        {"KeyCount": 1},
        {"KeyCount": 0, "Contents": []},
    ]
    mocker.patch("app.services.config_validation.httpx.get").return_value = MagicMock(is_success=True)

    svc = ConfigValidationService(
        skip_liveness_checks=False, max_retries=1, timeout=5,
        mlflow_tracking_uri="http://mlflow.test", s3_client=mock_s3,
    )
    with pytest.raises(ConfigValidationError, match="no .pt checkpoint found"):
        svc.run(config)


def test_resume_from_specific_path_found_passes(mocker):
    config = dict(VALID_CONFIG)
    config["checkpointing"] = {
        **VALID_CONFIG["checkpointing"],
        "resume_from": "s3://io-mlops/checkpoints/exp/last.pt",
    }

    mock_s3 = MagicMock()
    mock_s3.list_objects_v2.return_value = {"KeyCount": 1}
    mock_s3.head_object.return_value = {}
    mocker.patch("app.services.config_validation.httpx.get").return_value = MagicMock(is_success=True)

    svc = ConfigValidationService(
        skip_liveness_checks=False, max_retries=1, timeout=5,
        mlflow_tracking_uri="http://mlflow.test", s3_client=mock_s3,
    )
    svc.run(config)  # must not raise


def test_resume_from_specific_path_not_found_raises_error(mocker):
    import botocore.exceptions

    config = dict(VALID_CONFIG)
    config["checkpointing"] = {
        **VALID_CONFIG["checkpointing"],
        "resume_from": "s3://io-mlops/checkpoints/exp/last.pt",
    }

    mock_s3 = MagicMock()
    mock_s3.list_objects_v2.return_value = {"KeyCount": 1}
    mock_s3.head_object.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
    )
    mocker.patch("app.services.config_validation.httpx.get").return_value = MagicMock(is_success=True)

    svc = ConfigValidationService(
        skip_liveness_checks=False, max_retries=1, timeout=5,
        mlflow_tracking_uri="http://mlflow.test", s3_client=mock_s3,
    )
    with pytest.raises(ConfigValidationError, match="Checkpoint file not found"):
        svc.run(config)


def test_pretrained_weights_none_skips_s3_check(service_with_liveness, mock_s3, mocker):
    mock_s3.list_objects_v2.return_value = {"KeyCount": 1}
    mocker.patch("app.services.config_validation.httpx.get").return_value = MagicMock(is_success=True)

    service_with_liveness.run(VALID_CONFIG)

    # head_object should NOT have been called for weights (only list_objects_v2 for dataset)
    for call in mock_s3.head_object.call_args_list:
        assert "weights" not in str(call).lower()


def test_pretrained_weights_s3_path_checked(mocker):
    config = dict(VALID_CONFIG)
    config["model"] = {**VALID_CONFIG["model"], "pretrained_weights": "s3://io-mlops/weights/custom.pt"}

    mock_s3 = MagicMock()
    mock_s3.list_objects_v2.return_value = {"KeyCount": 1}
    mock_s3.head_object.return_value = {}
    mocker.patch("app.services.config_validation.httpx.get").return_value = MagicMock(is_success=True)

    svc = ConfigValidationService(
        skip_liveness_checks=False, max_retries=1, timeout=5,
        mlflow_tracking_uri="http://mlflow.test", s3_client=mock_s3,
    )
    svc.run(config)

    mock_s3.head_object.assert_called_once_with(Bucket="io-mlops", Key="weights/custom.pt")
