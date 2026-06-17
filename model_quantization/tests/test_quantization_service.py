"""Tests for QuantizationService.

Covers:
- PTQ: YOLO.export called with format='tflite', int8=True, correct data/imgsz
- QAT passthrough: TFLite URI forwarded; MLflow logs qat_run_id
- Mode dispatch: ptq → _run_ptq, qat → _run_qat_passthrough
- Checkpoint resolution: s3:// triggers download, local path unchanged
- Data YAML discovery: finds data.yaml / dataset.yaml, raises on missing
- S3 upload: correct bucket, key, URI format
- MLflow logging: all required params present, errors swallowed
- QuantizationParams validation: mode-specific field requirements
"""

import os
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest
from pydantic import ValidationError

from app.models.quantization import QuantizationParams, QuantizationResult
from app.services.quantization_service import QuantizationError, QuantizationService

MLFLOW_URI = "http://mlflow.example.com"
SOURCE_RUN_ID = "train-run-abc123"
QAT_RUN_ID = "qat-run-xyz789"
OUTPUT_BUCKET = "mlops-artifacts"
OUTPUT_PREFIX = "quant/exp-001"
DATASET_DIR = "/data/dataset"
OUTPUT_DIR = "/tmp/quant_out"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def s3() -> MagicMock:
    return MagicMock(name="s3_client")


@pytest.fixture
def service(s3: MagicMock) -> QuantizationService:
    return QuantizationService(s3_client=s3, mlflow_tracking_uri=MLFLOW_URI)


@pytest.fixture
def ptq_params() -> QuantizationParams:
    return QuantizationParams(
        mode="ptq",
        fp32_checkpoint_path="/local/best.pt",
        source_mlflow_run_id=SOURCE_RUN_ID,
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        output_bucket=OUTPUT_BUCKET,
        output_prefix=OUTPUT_PREFIX,
        experiment_name="quant-exp-v1",
    )


@pytest.fixture
def qat_params() -> QuantizationParams:
    return QuantizationParams(
        mode="qat",
        tflite_s3_uri=f"s3://{OUTPUT_BUCKET}/qat/exp-001/model_int8.tflite",
        qat_run_id=QAT_RUN_ID,
        source_mlflow_run_id=SOURCE_RUN_ID,
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        output_bucket=OUTPUT_BUCKET,
        output_prefix=OUTPUT_PREFIX,
        experiment_name="quant-exp-v1",
    )


def _mock_mlflow_ctx(run_id: str = "quant-run-001") -> MagicMock:
    mock = MagicMock()
    active_run = MagicMock()
    active_run.info.run_id = run_id
    mock.start_run.return_value.__enter__ = MagicMock(return_value=active_run)
    mock.start_run.return_value.__exit__ = MagicMock(return_value=False)
    return mock


# ── QuantizationParams validation ─────────────────────────────────────────────

class TestQuantizationParamsValidation:
    def _base(self, **kwargs) -> dict:
        return {
            "source_mlflow_run_id": SOURCE_RUN_ID,
            "dataset_dir": DATASET_DIR,
            "output_dir": OUTPUT_DIR,
            "output_bucket": OUTPUT_BUCKET,
            "output_prefix": OUTPUT_PREFIX,
            "experiment_name": "exp",
            **kwargs,
        }

    def test_ptq_without_checkpoint_raises(self) -> None:
        with pytest.raises(ValidationError, match="fp32_checkpoint_path"):
            QuantizationParams(**self._base(mode="ptq"))

    def test_qat_without_tflite_uri_raises(self) -> None:
        with pytest.raises(ValidationError, match="tflite_s3_uri"):
            QuantizationParams(**self._base(mode="qat"))

    def test_ptq_with_checkpoint_accepted(self) -> None:
        p = QuantizationParams(**self._base(mode="ptq", fp32_checkpoint_path="/local/best.pt"))
        assert p.mode == "ptq"

    def test_qat_with_tflite_uri_accepted(self) -> None:
        p = QuantizationParams(**self._base(mode="qat", tflite_s3_uri="s3://b/k.tflite"))
        assert p.mode == "qat"

    def test_mode_none_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QuantizationParams(**self._base(mode="none", fp32_checkpoint_path="/x"))  # type: ignore[arg-type]


# ── CheckpointResolution ──────────────────────────────────────────────────────

class TestCheckpointResolution:
    def test_local_path_returned_unchanged(self, service: QuantizationService, s3: MagicMock) -> None:
        result = service._resolve_checkpoint("/data/best.pt", "/tmp/out")
        assert result == "/data/best.pt"
        s3.download_file.assert_not_called()

    def test_s3_path_triggers_download(self, service: QuantizationService, s3: MagicMock) -> None:
        service._resolve_checkpoint("s3://my-bucket/checkpoints/best.pt", "/tmp/out")
        s3.download_file.assert_called_once_with("my-bucket", "checkpoints/best.pt", ANY)

    def test_s3_path_returns_local_path(self, service: QuantizationService, s3: MagicMock) -> None:
        local = service._resolve_checkpoint("s3://bucket/a/b/model.pt", "/tmp/out")
        assert local.startswith("/tmp/out")
        assert local.endswith("model.pt")


# ── DataYamlDiscovery ─────────────────────────────────────────────────────────

class TestDataYamlDiscovery:
    def test_finds_data_yaml(self, service: QuantizationService, tmp_path: Path) -> None:
        yaml = tmp_path / "data.yaml"
        yaml.write_text("nc: 1")
        result = service._find_data_yaml(str(tmp_path))
        assert result == str(yaml)

    def test_finds_dataset_yaml(self, service: QuantizationService, tmp_path: Path) -> None:
        yaml = tmp_path / "dataset.yaml"
        yaml.write_text("nc: 1")
        result = service._find_data_yaml(str(tmp_path))
        assert result == str(yaml)

    def test_finds_config_yaml(self, service: QuantizationService, tmp_path: Path) -> None:
        yaml = tmp_path / "config.yaml"
        yaml.write_text("nc: 1")
        result = service._find_data_yaml(str(tmp_path))
        assert result == str(yaml)

    def test_raises_when_no_yaml_found(self, service: QuantizationService, tmp_path: Path) -> None:
        with pytest.raises(QuantizationError, match="No YOLO data YAML"):
            service._find_data_yaml(str(tmp_path))

    def test_data_yaml_takes_priority_over_dataset_yaml(
        self, service: QuantizationService, tmp_path: Path
    ) -> None:
        (tmp_path / "data.yaml").write_text("nc: 1")
        (tmp_path / "dataset.yaml").write_text("nc: 1")
        result = service._find_data_yaml(str(tmp_path))
        assert result.endswith("data.yaml")


# ── PTQ Export ────────────────────────────────────────────────────────────────

class TestPTQExport:
    def test_yolo_export_called_with_tflite_format(
        self, service: QuantizationService, tmp_path: Path
    ) -> None:
        data_yaml = str(tmp_path / "data.yaml")
        (tmp_path / "data.yaml").write_text("nc: 1")
        mock_model = MagicMock()
        mock_model.export.return_value = str(tmp_path / "model_int8.tflite")

        with patch("app.services.quantization_service.YOLO", return_value=mock_model):
            service._export_ptq("/local/best.pt", data_yaml, MagicMock(image_size=640))

        mock_model.export.assert_called_once_with(
            format="tflite",
            int8=True,
            data=data_yaml,
            imgsz=640,
        )

    def test_export_returns_tflite_path(
        self, service: QuantizationService, tmp_path: Path
    ) -> None:
        expected = str(tmp_path / "model_int8.tflite")
        mock_model = MagicMock()
        mock_model.export.return_value = expected

        with patch("app.services.quantization_service.YOLO", return_value=mock_model):
            result = service._export_ptq("/local/best.pt", "/data/data.yaml", MagicMock(image_size=320))

        assert result == expected

    def test_yolo_loaded_with_checkpoint_path(
        self, service: QuantizationService
    ) -> None:
        mock_model = MagicMock()
        mock_model.export.return_value = "/out/model_int8.tflite"

        with patch("app.services.quantization_service.YOLO", return_value=mock_model) as mock_yolo:
            service._export_ptq("/local/best.pt", "/data.yaml", MagicMock(image_size=640))

        mock_yolo.assert_called_once_with("/local/best.pt")


# ── S3 Upload ─────────────────────────────────────────────────────────────────

class TestS3Upload:
    def test_upload_called_with_correct_bucket(
        self, service: QuantizationService, s3: MagicMock, ptq_params: QuantizationParams
    ) -> None:
        service._upload_tflite("/tmp/model_int8.tflite", ptq_params)
        _, pos_args, _ = s3.upload_file.mock_calls[0]
        assert pos_args[1] == OUTPUT_BUCKET

    def test_upload_called_with_correct_key(
        self, service: QuantizationService, s3: MagicMock, ptq_params: QuantizationParams
    ) -> None:
        service._upload_tflite("/tmp/model_int8.tflite", ptq_params)
        _, pos_args, _ = s3.upload_file.mock_calls[0]
        assert pos_args[2] == f"{OUTPUT_PREFIX}/model_int8.tflite"

    def test_s3_uri_format(
        self, service: QuantizationService, s3: MagicMock, ptq_params: QuantizationParams
    ) -> None:
        uri = service._upload_tflite("/tmp/model_int8.tflite", ptq_params)
        assert uri == f"s3://{OUTPUT_BUCKET}/{OUTPUT_PREFIX}/model_int8.tflite"


# ── MLflow Logging — PTQ ──────────────────────────────────────────────────────

class TestMLflowLoggingPTQ:
    def test_quantization_mode_ptq_logged(
        self, service: QuantizationService, ptq_params: QuantizationParams
    ) -> None:
        with patch("app.services.quantization_service.MlflowClient") as mock_cls:
            service._log_ptq_run("run-001", ptq_params, "s3://b/k.tflite")
        param_map = {c.args[1]: c.args[2] for c in mock_cls.return_value.log_param.call_args_list}
        assert param_map["quantization_mode"] == "ptq"

    def test_quantization_scheme_logged(
        self, service: QuantizationService, ptq_params: QuantizationParams
    ) -> None:
        with patch("app.services.quantization_service.MlflowClient") as mock_cls:
            service._log_ptq_run("run-001", ptq_params, "s3://b/k.tflite")
        param_map = {c.args[1]: c.args[2] for c in mock_cls.return_value.log_param.call_args_list}
        assert param_map["quantization_scheme"] == "per_tensor_int8"

    def test_source_run_id_logged(
        self, service: QuantizationService, ptq_params: QuantizationParams
    ) -> None:
        with patch("app.services.quantization_service.MlflowClient") as mock_cls:
            service._log_ptq_run("run-001", ptq_params, "s3://b/k.tflite")
        param_map = {c.args[1]: c.args[2] for c in mock_cls.return_value.log_param.call_args_list}
        assert param_map["source_run_id"] == SOURCE_RUN_ID

    def test_tflite_uri_logged(
        self, service: QuantizationService, ptq_params: QuantizationParams
    ) -> None:
        uri = "s3://bucket/quant/model_int8.tflite"
        with patch("app.services.quantization_service.MlflowClient") as mock_cls:
            service._log_ptq_run("run-001", ptq_params, uri)
        param_map = {c.args[1]: c.args[2] for c in mock_cls.return_value.log_param.call_args_list}
        assert param_map["tflite_s3_uri"] == uri

    def test_mlflow_error_swallowed(
        self, service: QuantizationService, ptq_params: QuantizationParams
    ) -> None:
        with patch("app.services.quantization_service.MlflowClient") as mock_cls:
            mock_cls.return_value.log_param.side_effect = Exception("MLflow down")
            service._log_ptq_run("run-001", ptq_params, "s3://b/k.tflite")  # must not raise


# ── MLflow Logging — QAT passthrough ─────────────────────────────────────────

class TestMLflowLoggingQAT:
    def test_quantization_mode_qat_logged(
        self, service: QuantizationService, qat_params: QuantizationParams
    ) -> None:
        with patch("app.services.quantization_service.MlflowClient") as mock_cls:
            service._log_qat_passthrough_run("run-002", qat_params)
        param_map = {c.args[1]: c.args[2] for c in mock_cls.return_value.log_param.call_args_list}
        assert param_map["quantization_mode"] == "qat"

    def test_qat_run_id_logged_when_present(
        self, service: QuantizationService, qat_params: QuantizationParams
    ) -> None:
        with patch("app.services.quantization_service.MlflowClient") as mock_cls:
            service._log_qat_passthrough_run("run-002", qat_params)
        param_map = {c.args[1]: c.args[2] for c in mock_cls.return_value.log_param.call_args_list}
        assert param_map.get("qat_run_id") == QAT_RUN_ID

    def test_qat_run_id_omitted_when_none(
        self, service: QuantizationService, qat_params: QuantizationParams
    ) -> None:
        qat_params.qat_run_id = None
        with patch("app.services.quantization_service.MlflowClient") as mock_cls:
            service._log_qat_passthrough_run("run-002", qat_params)
        param_map = {c.args[1]: c.args[2] for c in mock_cls.return_value.log_param.call_args_list}
        assert "qat_run_id" not in param_map


# ── Mode dispatch and run() integration ───────────────────────────────────────

class TestModeDispatch:
    def test_ptq_mode_dispatches_to_run_ptq(
        self, service: QuantizationService, ptq_params: QuantizationParams
    ) -> None:
        with patch.object(service, "_run_ptq", return_value=MagicMock(spec=QuantizationResult)) as mock_ptq, \
             patch("app.services.quantization_service.mlflow") as mock_mlflow:
            mock_mlflow.set_tracking_uri = MagicMock()
            mock_mlflow.set_experiment = MagicMock()
            service.run(ptq_params)
        mock_ptq.assert_called_once_with(ptq_params)

    def test_qat_mode_dispatches_to_run_qat_passthrough(
        self, service: QuantizationService, qat_params: QuantizationParams
    ) -> None:
        with patch.object(service, "_run_qat_passthrough", return_value=MagicMock(spec=QuantizationResult)) as mock_qat, \
             patch("app.services.quantization_service.mlflow") as mock_mlflow:
            mock_mlflow.set_tracking_uri = MagicMock()
            mock_mlflow.set_experiment = MagicMock()
            service.run(qat_params)
        mock_qat.assert_called_once_with(qat_params)


class TestRunPTQOrchestration:
    def _run_ptq_mocked(
        self,
        service: QuantizationService,
        params: QuantizationParams,
        run_id: str = "quant-run-001",
    ) -> QuantizationResult:
        tflite_path = os.path.join(params.output_dir, "model_int8.tflite")
        s3_uri = f"s3://{OUTPUT_BUCKET}/{OUTPUT_PREFIX}/model_int8.tflite"

        with (
            patch.object(service, "_resolve_checkpoint", return_value="/local/best.pt"),
            patch.object(service, "_find_data_yaml", return_value="/data/data.yaml"),
            patch.object(service, "_export_ptq", return_value=tflite_path),
            patch.object(service, "_upload_tflite", return_value=s3_uri),
            patch.object(service, "_log_ptq_run"),
            patch("app.services.quantization_service.mlflow") as mock_mlflow,
        ):
            active_run = MagicMock()
            active_run.info.run_id = run_id
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=active_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            return service._run_ptq(params)

    def test_returns_quantization_result(
        self, service: QuantizationService, ptq_params: QuantizationParams
    ) -> None:
        result = self._run_ptq_mocked(service, ptq_params)
        assert isinstance(result, QuantizationResult)

    def test_result_mode_is_ptq(
        self, service: QuantizationService, ptq_params: QuantizationParams
    ) -> None:
        result = self._run_ptq_mocked(service, ptq_params)
        assert result.mode == "ptq"

    def test_result_source_run_id(
        self, service: QuantizationService, ptq_params: QuantizationParams
    ) -> None:
        result = self._run_ptq_mocked(service, ptq_params)
        assert result.source_run_id == SOURCE_RUN_ID

    def test_result_tflite_s3_uri(
        self, service: QuantizationService, ptq_params: QuantizationParams
    ) -> None:
        result = self._run_ptq_mocked(service, ptq_params)
        assert result.tflite_s3_uri.startswith("s3://")

    def test_mlflow_start_run_tags_source_run_id(
        self, service: QuantizationService, ptq_params: QuantizationParams
    ) -> None:
        with (
            patch.object(service, "_resolve_checkpoint", return_value="/local/best.pt"),
            patch.object(service, "_find_data_yaml", return_value="/data/data.yaml"),
            patch.object(service, "_export_ptq", return_value="/tmp/m.tflite"),
            patch.object(service, "_upload_tflite", return_value="s3://b/k"),
            patch.object(service, "_log_ptq_run"),
            patch("app.services.quantization_service.mlflow") as mock_mlflow,
        ):
            active_run = MagicMock()
            active_run.info.run_id = "r"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=active_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            service._run_ptq(ptq_params)

        mock_mlflow.start_run.assert_called_once_with(
            tags={"source_run_id": SOURCE_RUN_ID}
        )


class TestRunQATPassthroughOrchestration:
    def _run_qat_mocked(
        self,
        service: QuantizationService,
        params: QuantizationParams,
        run_id: str = "quant-run-002",
    ) -> QuantizationResult:
        with (
            patch.object(service, "_log_qat_passthrough_run"),
            patch("app.services.quantization_service.mlflow") as mock_mlflow,
        ):
            active_run = MagicMock()
            active_run.info.run_id = run_id
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=active_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            return service._run_qat_passthrough(params)

    def test_tflite_uri_forwarded_in_result(
        self, service: QuantizationService, qat_params: QuantizationParams
    ) -> None:
        result = self._run_qat_mocked(service, qat_params)
        assert result.tflite_s3_uri == qat_params.tflite_s3_uri

    def test_mode_is_qat(
        self, service: QuantizationService, qat_params: QuantizationParams
    ) -> None:
        result = self._run_qat_mocked(service, qat_params)
        assert result.mode == "qat"

    def test_source_run_id_in_result(
        self, service: QuantizationService, qat_params: QuantizationParams
    ) -> None:
        result = self._run_qat_mocked(service, qat_params)
        assert result.source_run_id == SOURCE_RUN_ID

    def test_mlflow_tags_include_qat_run_id(
        self, service: QuantizationService, qat_params: QuantizationParams
    ) -> None:
        with (
            patch.object(service, "_log_qat_passthrough_run"),
            patch("app.services.quantization_service.mlflow") as mock_mlflow,
        ):
            active_run = MagicMock()
            active_run.info.run_id = "r"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=active_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            service._run_qat_passthrough(qat_params)

        _, kwargs = mock_mlflow.start_run.call_args
        assert kwargs["tags"]["qat_run_id"] == QAT_RUN_ID
        assert kwargs["tags"]["source_run_id"] == SOURCE_RUN_ID

    def test_no_upload_called_for_qat_passthrough(
        self, service: QuantizationService, s3: MagicMock, qat_params: QuantizationParams
    ) -> None:
        """QAT TFLite is already on S3 from qat-finetune — no re-upload."""
        with (
            patch.object(service, "_log_qat_passthrough_run"),
            patch("app.services.quantization_service.mlflow") as mock_mlflow,
        ):
            active_run = MagicMock()
            active_run.info.run_id = "r"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=active_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            service._run_qat_passthrough(qat_params)

        s3.upload_file.assert_not_called()
