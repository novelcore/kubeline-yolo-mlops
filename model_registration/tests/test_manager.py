"""Tests for the Manager class."""

from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from app.manager import Manager
from app.models.config import Config
from app.models.registration import RegistrationResult

TRACKING_URI = "http://mlflow.test.local:5000"
MODEL_NAME = "spacecraft-pose-yolo"
RUN_ID = "run123"
BEST_S3 = "s3://io-mlops/checkpoints/exp/best.pt"
LAST_S3 = "s3://io-mlops/checkpoints/exp/last.pt"


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", TRACKING_URI)
    return Config()


def _make_result(
    best_version: int = 1,
    last_version: Optional[int] = None,
    promoted_to: Optional[str] = None,
) -> RegistrationResult:
    return RegistrationResult(
        registered_model_name=MODEL_NAME,
        best_version=best_version,
        last_version=last_version,
        registered_at=datetime.now(timezone.utc),
        promoted_to=promoted_to,
    )


class TestManagerRun:
    def test_delegates_to_service(self, config: Config) -> None:
        with patch("app.manager.ModelRegistrationService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.run.return_value = _make_result(best_version=1)

            manager = Manager(config=config)
            result = manager.run(
                mlflow_run_id=RUN_ID,
                best_checkpoint_path=BEST_S3,
            )

        mock_service.run.assert_called_once()
        assert result.best_version == 1

    def test_passes_all_lineage_params(self, config: Config) -> None:
        with patch("app.manager.ModelRegistrationService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.run.return_value = _make_result(best_version=2, last_version=3)

            manager = Manager(config=config)
            manager.run(
                mlflow_run_id=RUN_ID,
                best_checkpoint_path=BEST_S3,
                last_checkpoint_path=LAST_S3,
                dataset_version="v1",
                dataset_sample_size=5000,
                config_hash="abc123",
                git_commit="def456",
                model_variant="yolov8n-pose.pt",
                best_map50=0.85,
            )

        called_params = mock_service.run.call_args.kwargs["params"]
        assert called_params.dataset_version == "v1"
        assert called_params.dataset_sample_size == 5000
        assert called_params.config_hash == "abc123"
        assert called_params.git_commit == "def456"
        assert called_params.model_variant == "yolov8n-pose.pt"
        assert called_params.best_map50 == pytest.approx(0.85)

    def test_uses_config_registered_model_name_by_default(self, config: Config) -> None:
        with patch("app.manager.ModelRegistrationService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.run.return_value = _make_result()

            manager = Manager(config=config)
            manager.run(
                mlflow_run_id=RUN_ID,
                best_checkpoint_path=BEST_S3,
            )

        called_params = mock_service.run.call_args.kwargs["params"]
        assert called_params.registered_model_name == config.registered_model_name

    def test_cli_override_of_registered_model_name(self, config: Config) -> None:
        with patch("app.manager.ModelRegistrationService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.run.return_value = _make_result()

            manager = Manager(config=config)
            manager.run(
                mlflow_run_id=RUN_ID,
                best_checkpoint_path=BEST_S3,
                registered_model_name="override-model",
            )

        called_params = mock_service.run.call_args.kwargs["params"]
        assert called_params.registered_model_name == "override-model"

    def test_passes_promote_to(self, config: Config) -> None:
        with patch("app.manager.ModelRegistrationService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.run.return_value = _make_result(promoted_to="Staging")

            manager = Manager(config=config)
            result = manager.run(
                mlflow_run_id=RUN_ID,
                best_checkpoint_path=BEST_S3,
                promote_to="Staging",
            )

        called_params = mock_service.run.call_args.kwargs["params"]
        assert called_params.promote_to == "Staging"
        assert result.promoted_to == "Staging"

    def test_constructs_service_with_tracking_uri_from_config(
        self, config: Config
    ) -> None:
        with patch("app.manager.ModelRegistrationService") as mock_service_cls:
            mock_service_cls.return_value.run.return_value = _make_result()

            Manager(config=config)

        mock_service_cls.assert_called_once_with(
            mlflow_tracking_uri=TRACKING_URI,
            max_retries=config.max_retries,
        )
