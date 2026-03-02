"""Tests for ModelRegistrationService with mocked MLflow."""

from unittest.mock import MagicMock, call, patch

import pytest

from app.models.registration import RegistrationParams, RegistrationResult
from app.services.model_registration import (
    ModelRegistrationError,
    ModelRegistrationService,
)

TRACKING_URI = "http://mlflow.example.com"
MODEL_NAME = "spacecraft-pose-yolo"
RUN_ID = "abc123def456"
BEST_S3 = "s3://io-mlops/checkpoints/exp-001/best.pt"
LAST_S3 = "s3://io-mlops/checkpoints/exp-001/last.pt"


@pytest.fixture
def service() -> ModelRegistrationService:
    return ModelRegistrationService(mlflow_tracking_uri=TRACKING_URI, max_retries=3)


@pytest.fixture
def minimal_params() -> RegistrationParams:
    return RegistrationParams(
        mlflow_run_id=RUN_ID,
        best_checkpoint_path=BEST_S3,
        registered_model_name=MODEL_NAME,
    )


@pytest.fixture
def full_params() -> RegistrationParams:
    return RegistrationParams(
        mlflow_run_id=RUN_ID,
        best_checkpoint_path=BEST_S3,
        last_checkpoint_path=LAST_S3,
        registered_model_name=MODEL_NAME,
        dataset_version="v1",
        dataset_sample_size=5000,
        config_hash="deadbeef" * 8,
        git_commit="1a2b3c4d",
        model_variant="yolov8n-pose.pt",
        best_map50=0.85,
    )


def _make_mv(version: str = "1") -> MagicMock:
    mv = MagicMock()
    mv.version = version
    return mv


class TestRegisterBestOnly:
    """Registration of best.pt only, with last.pt derived from the path."""

    def test_registers_best_and_returns_result(
        self, service: ModelRegistrationService, minimal_params: RegistrationParams
    ) -> None:
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient") as mock_client_cls,
        ):
            mock_mlflow.register_model.return_value = _make_mv("1")
            mock_client = mock_client_cls.return_value

            result = service.run(minimal_params)

        assert isinstance(result, RegistrationResult)
        assert result.registered_model_name == MODEL_NAME
        assert result.best_version == 1
        assert result.promoted_to is None

    def test_derives_last_pt_path_from_best(
        self, service: ModelRegistrationService, minimal_params: RegistrationParams
    ) -> None:
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient"),
        ):
            mock_mlflow.register_model.side_effect = [_make_mv("1"), _make_mv("2")]

            result = service.run(minimal_params)

        assert result.last_version == 2
        calls = mock_mlflow.register_model.call_args_list
        assert calls[1] == call(model_uri=LAST_S3, name=MODEL_NAME)

    def test_skips_last_pt_when_best_path_has_no_best_pt_substring(
        self, service: ModelRegistrationService
    ) -> None:
        params = RegistrationParams(
            mlflow_run_id=RUN_ID,
            best_checkpoint_path="s3://io-mlops/checkpoints/model.pt",
            registered_model_name=MODEL_NAME,
        )
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient"),
        ):
            mock_mlflow.register_model.return_value = _make_mv("1")

            result = service.run(params)

        assert result.last_version is None
        assert mock_mlflow.register_model.call_count == 1


class TestRegisterBestAndLast:
    """Registration of both best.pt and explicitly provided last.pt."""

    def test_registers_both_checkpoints(
        self, service: ModelRegistrationService, full_params: RegistrationParams
    ) -> None:
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient"),
        ):
            mock_mlflow.register_model.side_effect = [_make_mv("3"), _make_mv("4")]

            result = service.run(full_params)

        assert result.best_version == 3
        assert result.last_version == 4
        assert mock_mlflow.register_model.call_count == 2

    def test_registers_best_with_correct_uri(
        self, service: ModelRegistrationService, full_params: RegistrationParams
    ) -> None:
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient"),
        ):
            mock_mlflow.register_model.side_effect = [_make_mv("1"), _make_mv("2")]
            service.run(full_params)

        first_call = mock_mlflow.register_model.call_args_list[0]
        assert first_call == call(model_uri=BEST_S3, name=MODEL_NAME)

    def test_registers_last_with_correct_uri(
        self, service: ModelRegistrationService, full_params: RegistrationParams
    ) -> None:
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient"),
        ):
            mock_mlflow.register_model.side_effect = [_make_mv("1"), _make_mv("2")]
            service.run(full_params)

        second_call = mock_mlflow.register_model.call_args_list[1]
        assert second_call == call(model_uri=LAST_S3, name=MODEL_NAME)


class TestLineageTags:
    """Verify that all lineage tags are set on the registered versions."""

    def test_best_checkpoint_type_tag_is_set(
        self, service: ModelRegistrationService, minimal_params: RegistrationParams
    ) -> None:
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient") as mock_client_cls,
        ):
            mock_mlflow.register_model.side_effect = [_make_mv("1"), _make_mv("2")]
            mock_client = mock_client_cls.return_value

            service.run(minimal_params)

        tag_calls = mock_client.set_model_version_tag.call_args_list
        tag_keys = [c.kwargs.get("key") or c.args[2] for c in tag_calls]
        tag_values = [c.kwargs.get("value") or c.args[3] for c in tag_calls]
        best_idx = tag_keys.index("checkpoint_type")
        assert tag_values[best_idx] == "best"

    def test_last_checkpoint_type_tag_is_set(
        self, service: ModelRegistrationService, full_params: RegistrationParams
    ) -> None:
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient") as mock_client_cls,
        ):
            mock_mlflow.register_model.side_effect = [_make_mv("1"), _make_mv("2")]
            mock_client = mock_client_cls.return_value

            service.run(full_params)

        tag_calls = mock_client.set_model_version_tag.call_args_list
        checkpoint_type_calls = [
            c
            for c in tag_calls
            if (c.kwargs.get("key") or c.args[2]) == "checkpoint_type"
        ]
        assert len(checkpoint_type_calls) == 2
        values = [(c.kwargs.get("value") or c.args[3]) for c in checkpoint_type_calls]
        assert "best" in values
        assert "last" in values

    def test_all_lineage_tags_are_applied(
        self, service: ModelRegistrationService, full_params: RegistrationParams
    ) -> None:
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient") as mock_client_cls,
        ):
            mock_mlflow.register_model.side_effect = [_make_mv("1"), _make_mv("2")]
            mock_client = mock_client_cls.return_value

            service.run(full_params)

        all_keys = {
            (c.kwargs.get("key") or c.args[2])
            for c in mock_client.set_model_version_tag.call_args_list
        }
        expected_keys = {
            "checkpoint_type",
            "training_run_id",
            "dataset_version",
            "dataset_sample_size",
            "config_hash",
            "git_commit",
            "model_variant",
            "best_mAP50",
        }
        assert expected_keys.issubset(all_keys)

    def test_optional_tags_omitted_when_not_provided(
        self, service: ModelRegistrationService, minimal_params: RegistrationParams
    ) -> None:
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient") as mock_client_cls,
        ):
            mock_mlflow.register_model.side_effect = [_make_mv("1"), _make_mv("2")]
            mock_client = mock_client_cls.return_value

            service.run(minimal_params)

        all_keys = {
            (c.kwargs.get("key") or c.args[2])
            for c in mock_client.set_model_version_tag.call_args_list
        }
        assert "dataset_version" not in all_keys
        assert "config_hash" not in all_keys


class TestRetryBehavior:
    """Exponential backoff on transient MLflow failures."""

    def test_retries_on_failure_then_succeeds(
        self, service: ModelRegistrationService, minimal_params: RegistrationParams
    ) -> None:
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient"),
            patch("app.services.model_registration.time.sleep") as mock_sleep,
        ):
            mock_mlflow.register_model.side_effect = [
                Exception("connection refused"),
                _make_mv("1"),
                _make_mv("2"),
            ]

            result = service.run(minimal_params)

        assert result.best_version == 1
        mock_sleep.assert_called_once_with(1)

    def test_raises_after_max_retries_exhausted(
        self, service: ModelRegistrationService, minimal_params: RegistrationParams
    ) -> None:
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient"),
            patch("app.services.model_registration.time.sleep"),
        ):
            mock_mlflow.register_model.side_effect = Exception("MLflow is down")

            with pytest.raises(ModelRegistrationError, match="3 attempts"):
                service.run(minimal_params)

    def test_sleep_delays_follow_backoff_schedule(
        self, service: ModelRegistrationService, minimal_params: RegistrationParams
    ) -> None:
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient"),
            patch("app.services.model_registration.time.sleep") as mock_sleep,
        ):
            mock_mlflow.register_model.side_effect = [
                Exception("fail"),
                Exception("fail"),
                _make_mv("1"),
                _make_mv("2"),
            ]

            service.run(minimal_params)

        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_calls == [1, 2]


class TestPromotion:
    """Model version stage transition."""

    def test_promotes_best_version_when_requested(
        self, service: ModelRegistrationService, minimal_params: RegistrationParams
    ) -> None:
        minimal_params.promote_to = "Staging"
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient") as mock_client_cls,
        ):
            mock_mlflow.register_model.side_effect = [_make_mv("7"), _make_mv("8")]
            mock_client = mock_client_cls.return_value

            result = service.run(minimal_params)

        assert result.promoted_to == "Staging"
        mock_client.transition_model_version_stage.assert_called_once_with(
            name=MODEL_NAME,
            version="7",
            stage="Staging",
        )

    def test_no_promotion_when_promote_to_is_none(
        self, service: ModelRegistrationService, minimal_params: RegistrationParams
    ) -> None:
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient") as mock_client_cls,
        ):
            mock_mlflow.register_model.side_effect = [_make_mv("1"), _make_mv("2")]
            mock_client = mock_client_cls.return_value

            result = service.run(minimal_params)

        assert result.promoted_to is None
        mock_client.transition_model_version_stage.assert_not_called()
