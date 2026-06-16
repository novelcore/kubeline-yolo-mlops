"""Tests for ModelRegistrationService with mocked MLflow."""

from unittest.mock import MagicMock, call, patch

import pytest
from pydantic import ValidationError

from app.models.registration import RegistrationParams, RegistrationResult
from app.services.model_registration import (
    ModelRegistrationError,
    ModelRegistrationService,
)

TRACKING_URI = "http://mlflow.example.com"
MODEL_NAME = "object-pose-yolo"
RUN_ID = "abc123def456"
BEST_S3 = "s3://mlops-artifacts/checkpoints/exp-001/best.pt"
LAST_S3 = "s3://mlops-artifacts/checkpoints/exp-001/last.pt"


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
            best_checkpoint_path="s3://mlops-artifacts/checkpoints/model.pt",
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
    """Model version registry alias assignment."""

    def test_promotes_best_version_when_requested(
        self, service: ModelRegistrationService, minimal_params: RegistrationParams
    ) -> None:
        minimal_params.promote_to = "champion"
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient") as mock_client_cls,
        ):
            mock_mlflow.register_model.side_effect = [_make_mv("7"), _make_mv("8")]
            mock_client = mock_client_cls.return_value

            result = service.run(minimal_params)

        assert result.promoted_to == "champion"
        mock_client.set_registered_model_alias.assert_called_once_with(
            name=MODEL_NAME,
            alias="champion",
            version="7",
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
        mock_client.set_registered_model_alias.assert_not_called()


class TestRegistrationParamsValidation:
    """Pydantic validation rules for RegistrationParams.promote_to (F-06 fix)."""

    def _base(self, **kwargs) -> dict:
        return {
            "mlflow_run_id": RUN_ID,
            "best_checkpoint_path": BEST_S3,
            "registered_model_name": MODEL_NAME,
            **kwargs,
        }

    def test_promote_to_none_python_is_accepted(self) -> None:
        """AC-04: Python None is a valid promote_to value."""
        params = RegistrationParams(**self._base(promote_to=None))
        assert params.promote_to is None

    def test_promote_to_omitted_defaults_to_none(self) -> None:
        """promote_to field is optional and defaults to None."""
        params = RegistrationParams(**self._base())
        assert params.promote_to is None

    def test_promote_to_champion_is_accepted(self) -> None:
        params = RegistrationParams(**self._base(promote_to="champion"))
        assert params.promote_to == "champion"

    def test_promote_to_challenger_is_accepted(self) -> None:
        params = RegistrationParams(**self._base(promote_to="challenger"))
        assert params.promote_to == "challenger"

    def test_promote_to_string_none_is_rejected(self) -> None:
        """AC-03: The string 'None' must be rejected — it is not the same as Python None."""
        with pytest.raises(ValidationError, match="champion"):
            RegistrationParams(**self._base(promote_to="None"))

    def test_promote_to_invalid_alias_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RegistrationParams(**self._base(promote_to="staging"))


EXPORTED_MODELS = {
    "engine_fp16": "s3://mlops-artifacts/checkpoints/exp-001/best.engine",
    "onnx_fp16": "s3://mlops-artifacts/checkpoints/exp-001/best.onnx",
}


class TestRegisterExportedVariants:
    """F-06: exported model variants registered as separate MLflow model versions."""

    def _params_with_exports(self, exported_models=None) -> RegistrationParams:
        return RegistrationParams(
            mlflow_run_id=RUN_ID,
            best_checkpoint_path=BEST_S3,
            registered_model_name=MODEL_NAME,
            exported_models=exported_models or EXPORTED_MODELS,
        )

    def test_registers_each_variant_under_derived_name(
        self, service: ModelRegistrationService
    ) -> None:
        """AC-04: each label is registered under {model_name}-{label}."""
        params = self._params_with_exports()
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient") as mock_client_cls,
        ):
            # best.pt + last.pt + 2 exports = 4 calls
            mock_mlflow.register_model.side_effect = [
                _make_mv("1"), _make_mv("2"), _make_mv("3"), _make_mv("4")
            ]
            mock_client_cls.return_value

            service.run(params)

        names_used = [
            c.kwargs.get("name") or c.args[1]
            for c in mock_mlflow.register_model.call_args_list
        ]
        assert f"{MODEL_NAME}-engine_fp16" in names_used
        assert f"{MODEL_NAME}-onnx_fp16" in names_used

    def test_exported_versions_returned_in_result(
        self, service: ModelRegistrationService
    ) -> None:
        """AC-05: RegistrationResult.exported_versions maps labels to version numbers."""
        params = self._params_with_exports()
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient"),
        ):
            mock_mlflow.register_model.side_effect = [
                _make_mv("1"), _make_mv("2"), _make_mv("3"), _make_mv("4")
            ]

            result = service.run(params)

        assert "engine_fp16" in result.exported_versions
        assert "onnx_fp16" in result.exported_versions
        assert result.exported_versions["engine_fp16"] == 3
        assert result.exported_versions["onnx_fp16"] == 4

    def test_lineage_tags_set_on_exported_version(
        self, service: ModelRegistrationService
    ) -> None:
        """AC-04: source_model_name, source_model_version, source_run_id tags are set."""
        params = RegistrationParams(
            mlflow_run_id=RUN_ID,
            best_checkpoint_path=BEST_S3,
            registered_model_name=MODEL_NAME,
            exported_models={"engine_fp16": "s3://bucket/best.engine"},
        )
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient") as mock_client_cls,
        ):
            mock_mlflow.register_model.side_effect = [
                _make_mv("1"), _make_mv("2"), _make_mv("5")
            ]
            mock_client = mock_client_cls.return_value

            service.run(params)

        tag_calls = mock_client.set_model_version_tag.call_args_list
        # Extract tags for the exported version (version "5")
        export_tags = {
            (c.kwargs.get("key") or c.args[2]): (c.kwargs.get("value") or c.args[3])
            for c in tag_calls
            if (c.kwargs.get("version") or c.args[1]) == "5"
        }
        assert export_tags.get("source_model_name") == MODEL_NAME
        assert export_tags.get("source_model_version") == "1"  # best_version
        assert export_tags.get("source_run_id") == RUN_ID
        assert export_tags.get("export_label") == "engine_fp16"
        assert export_tags.get("checkpoint_type") == "exported"

    def test_label_sanitisation_replaces_invalid_chars(
        self, service: ModelRegistrationService
    ) -> None:
        """CON-04: dots and spaces in labels are replaced with hyphens."""
        params = RegistrationParams(
            mlflow_run_id=RUN_ID,
            best_checkpoint_path=BEST_S3,
            registered_model_name=MODEL_NAME,
            exported_models={"engine.fp16": "s3://bucket/best.engine"},
        )
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient"),
        ):
            mock_mlflow.register_model.side_effect = [
                _make_mv("1"), _make_mv("2"), _make_mv("3")
            ]

            service.run(params)

        names_used = [
            c.kwargs.get("name") or c.args[1]
            for c in mock_mlflow.register_model.call_args_list
        ]
        assert f"{MODEL_NAME}-engine-fp16" in names_used
        assert f"{MODEL_NAME}-engine.fp16" not in names_used

    def test_single_variant_failure_is_non_fatal(
        self, service: ModelRegistrationService
    ) -> None:
        """CON-01: a failed export registration does not abort remaining variants."""
        params = self._params_with_exports()
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient"),
            patch("app.services.model_registration.time.sleep"),
        ):
            # best.pt, last.pt succeed; first export always fails; second export succeeds
            mock_mlflow.register_model.side_effect = [
                _make_mv("1"),   # best.pt
                _make_mv("2"),   # last.pt
                Exception("TRT engine upload failed"),  # attempt 1 for export 1
                Exception("TRT engine upload failed"),  # attempt 2 for export 1
                Exception("TRT engine upload failed"),  # attempt 3 for export 1
                _make_mv("4"),   # second export succeeds
            ]

            result = service.run(params)

        # One variant failed, one succeeded — result must not be empty
        assert len(result.exported_versions) == 1

    def test_no_exported_variants_when_exported_models_is_none(
        self, service: ModelRegistrationService
    ) -> None:
        """CON-02: exported_versions defaults to {} when exported_models is not provided."""
        params = RegistrationParams(
            mlflow_run_id=RUN_ID,
            best_checkpoint_path=BEST_S3,
            registered_model_name=MODEL_NAME,
        )
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient"),
        ):
            mock_mlflow.register_model.side_effect = [_make_mv("1"), _make_mv("2")]

            result = service.run(params)

        assert result.exported_versions == {}

    def test_no_exported_variants_when_exported_models_is_empty(
        self, service: ModelRegistrationService
    ) -> None:
        """Empty dict is a valid input — no extra register_model calls."""
        params = RegistrationParams(
            mlflow_run_id=RUN_ID,
            best_checkpoint_path=BEST_S3,
            registered_model_name=MODEL_NAME,
            exported_models={},
        )
        with (
            patch("app.services.model_registration.mlflow") as mock_mlflow,
            patch("app.services.model_registration.MlflowClient"),
        ):
            mock_mlflow.register_model.side_effect = [_make_mv("1"), _make_mv("2")]

            result = service.run(params)

        assert result.exported_versions == {}
        assert mock_mlflow.register_model.call_count == 2  # only best.pt + last.pt


class TestRegistrationResultExportedVersions:
    """F-07: RegistrationResult.exported_versions field (CON-02 backward compat)."""

    def test_exported_versions_defaults_to_empty_dict(self) -> None:
        """CON-02: existing callers that don't pass exported_versions still work."""
        result = RegistrationResult(
            registered_model_name=MODEL_NAME,
            best_version=1,
            registered_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        )
        assert result.exported_versions == {}

    def test_exported_versions_can_be_set(self) -> None:
        result = RegistrationResult(
            registered_model_name=MODEL_NAME,
            best_version=1,
            registered_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            exported_versions={"engine_fp16": 3, "onnx_fp16": 4},
        )
        assert result.exported_versions["engine_fp16"] == 3
        assert result.exported_versions["onnx_fp16"] == 4
