"""Tests for QATService.

Tests verify:
- CON-02: per-tensor INT8 (is_per_channel=False) enforced
- CON-03: head exclusion recipe applied before graph capture
- torch.export called with strict=False; .module() extracted
- convert_pt2e called with fold_quantize=False (CON-01)
- litert_torch.convert used for TFLite export
- MLflow source_run_id in tags; all params logged; errors swallowed
- S3 upload called with correct bucket/key; URI format correct
- Checkpoint downloaded from S3 when path starts with s3://
- run() returns QATResult with correct field values
"""

import sys
from unittest.mock import ANY, MagicMock, call, patch

import pytest

from app.models.quantization import QATParams, QATResult
from app.services.qat_service import QATService

# Convenience references to the stubs installed by conftest.py
_torchao_pt2e = sys.modules["torchao.quantization.pt2e"]
_litert_pt2e = sys.modules["litert_torch.quantization.pt2e"]
_litert = sys.modules["litert_torch"]


# ── Fixtures ──────────────────────────────────────────────────────────────────

MLFLOW_URI = "http://mlflow.example.com"
SOURCE_RUN_ID = "train-run-abc123"
OUTPUT_BUCKET = "mlops-artifacts"
OUTPUT_PREFIX = "qat/exp-001"


@pytest.fixture
def s3() -> MagicMock:
    return MagicMock(name="s3_client")


@pytest.fixture
def service(s3: MagicMock) -> QATService:
    return QATService(s3_client=s3, mlflow_tracking_uri=MLFLOW_URI)


@pytest.fixture
def params() -> QATParams:
    return QATParams(
        fp32_checkpoint_path="/local/best.pt",
        source_mlflow_run_id=SOURCE_RUN_ID,
        dataset_dir="/data/dataset",
        output_dir="/tmp/qat_out",
        output_bucket=OUTPUT_BUCKET,
        output_prefix=OUTPUT_PREFIX,
        experiment_name="qat-exp-v1",
        image_size=640,
        qat_epochs=5,
        qat_lr=1e-4,
        calibration_frames=200,
        calibration_seed=42,
    )


def _make_mlflow_run(run_id: str = "qat-run-xyz") -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    return run


def _mock_mlflow_ctx(run_id: str = "qat-run-xyz") -> MagicMock:
    mock_mlflow = MagicMock()
    active_run = _make_mlflow_run(run_id)
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=active_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
    return mock_mlflow


# ── CheckpointResolution ──────────────────────────────────────────────────────


class TestCheckpointResolution:
    def test_local_path_returned_unchanged(self, service: QATService, s3: MagicMock) -> None:
        result = service._resolve_checkpoint("/data/best.pt", "/tmp/out")
        assert result == "/data/best.pt"
        s3.download_file.assert_not_called()

    def test_s3_path_triggers_download(self, service: QATService, s3: MagicMock) -> None:
        service._resolve_checkpoint("s3://my-bucket/checkpoints/best.pt", "/tmp/out")
        s3.download_file.assert_called_once_with(
            "my-bucket", "checkpoints/best.pt", ANY
        )

    def test_s3_path_returns_local_path_in_output_dir(
        self, service: QATService, s3: MagicMock
    ) -> None:
        local = service._resolve_checkpoint(
            "s3://my-bucket/checkpoints/best.pt", "/tmp/out"
        )
        assert local.startswith("/tmp/out")
        assert local.endswith("best.pt")

    def test_s3_path_with_nested_key(self, service: QATService, s3: MagicMock) -> None:
        service._resolve_checkpoint(
            "s3://bucket/a/b/c/model.pt", "/output"
        )
        s3.download_file.assert_called_once_with("bucket", "a/b/c/model.pt", ANY)


# ── HeadExclusion (CON-03) ────────────────────────────────────────────────────


class TestHeadExclusion:
    """Verify CON-03: head exclusion recipe applied before graph capture."""

    def _make_yolo_mock(self) -> MagicMock:
        yolo = MagicMock(name="YOLO")
        module = MagicMock(name="yolo.model")
        module.to.return_value = module
        yolo.model = module
        return yolo

    def test_module_moved_to_device(self, service: QATService) -> None:
        yolo = self._make_yolo_mock()
        with patch("app.services.qat_service.YOLO", return_value=yolo):
            service._load_headless_module("/local/best.pt", "cuda")
        yolo.model.to.assert_called_once_with("cuda")

    def test_eval_called_on_module(self, service: QATService) -> None:
        yolo = self._make_yolo_mock()
        with patch("app.services.qat_service.YOLO", return_value=yolo):
            service._load_headless_module("/local/best.pt", "cpu")
        yolo.model.eval.assert_called_once()

    def test_detect_head_set_to_training_mode(self, service: QATService) -> None:
        """CON-03: model.model[-1].training = True excludes head from graph."""
        yolo = self._make_yolo_mock()
        head_mock = MagicMock(name="detect_head")
        yolo.model.model.__getitem__ = MagicMock(return_value=head_mock)

        with patch("app.services.qat_service.YOLO", return_value=yolo):
            service._load_headless_module("/local/best.pt", "cpu")

        # training attribute must be set to True on the last layer
        assert head_mock.training is True or (
            hasattr(head_mock, "training") and yolo.model.model[-1].training is True
        )


# ── GraphCapture ──────────────────────────────────────────────────────────────


class TestGraphCapture:
    def test_torch_export_called_with_strict_false(self, service: QATService) -> None:
        module_mock = MagicMock()
        sample = (MagicMock(),)

        with patch("app.services.qat_service.torch") as mock_torch:
            exported_program = MagicMock()
            mock_torch.export.export.return_value = exported_program

            service._capture_graph(module_mock, sample)

        mock_torch.export.export.assert_called_once_with(
            module_mock, sample, strict=False
        )

    def test_module_extracted_via_dot_module(self, service: QATService) -> None:
        """Must call .module() — NOT export_for_training()."""
        module_mock = MagicMock()
        sample = (MagicMock(),)

        with patch("app.services.qat_service.torch") as mock_torch:
            exported_program = MagicMock()
            extracted = MagicMock(name="extracted_module")
            exported_program.module.return_value = extracted
            mock_torch.export.export.return_value = exported_program

            result = service._capture_graph(module_mock, sample)

        exported_program.module.assert_called_once_with()
        assert result is extracted


# ── QuantizerSetup (CON-02) ───────────────────────────────────────────────────


class TestQuantizerSetup:
    """Verify CON-02: litert_torch PT2EQuantizer with is_per_channel=False."""

    def test_litert_pt2e_quantizer_is_used(self, service: QATService) -> None:
        """Must use litert_torch's PT2EQuantizer — NOT torch.ao's."""
        mock_quantizer_cls = MagicMock(name="PT2EQuantizer")
        mock_quantizer = MagicMock()
        mock_quantizer_cls.return_value = mock_quantizer
        mock_quantizer.set_global.return_value = mock_quantizer

        with patch.dict(
            sys.modules,
            {
                "litert_torch.quantization.pt2e": MagicMock(
                    PT2EQuantizer=mock_quantizer_cls,
                    get_symmetric_quantization_config=MagicMock(return_value=MagicMock()),
                )
            },
        ):
            with patch("app.services.qat_service.prepare_qat_pt2e", return_value=MagicMock()):
                service._prepare_qat(MagicMock())

        mock_quantizer_cls.assert_called_once_with()

    def test_per_tensor_config_enforced(self, service: QATService) -> None:
        """CON-02: get_symmetric_quantization_config must receive is_per_channel=False."""
        mock_get_config = MagicMock(return_value=MagicMock())
        mock_quantizer = MagicMock()
        mock_quantizer.set_global.return_value = mock_quantizer

        with patch.dict(
            sys.modules,
            {
                "litert_torch.quantization.pt2e": MagicMock(
                    PT2EQuantizer=MagicMock(return_value=mock_quantizer),
                    get_symmetric_quantization_config=mock_get_config,
                )
            },
        ):
            with patch("app.services.qat_service.prepare_qat_pt2e", return_value=MagicMock()):
                service._prepare_qat(MagicMock())

        mock_get_config.assert_called_once_with(is_per_channel=False)

    def test_prepare_qat_pt2e_called_with_module_and_quantizer(
        self, service: QATService
    ) -> None:
        input_module = MagicMock(name="exported_module")
        mock_quantizer = MagicMock()
        mock_quantizer.set_global.return_value = mock_quantizer
        prepared = MagicMock(name="prepared")

        with patch.dict(
            sys.modules,
            {
                "litert_torch.quantization.pt2e": MagicMock(
                    PT2EQuantizer=MagicMock(return_value=mock_quantizer),
                    get_symmetric_quantization_config=MagicMock(return_value=MagicMock()),
                )
            },
        ):
            with patch(
                "app.services.qat_service.prepare_qat_pt2e", return_value=prepared
            ) as mock_prepare:
                result = service._prepare_qat(input_module)

        mock_prepare.assert_called_once_with(input_module, mock_quantizer)
        assert result is prepared


# ── Conversion (CON-01: fold_quantize=False) ──────────────────────────────────


class TestConversion:
    def test_convert_pt2e_called_with_fold_quantize_false(
        self, service: QATService
    ) -> None:
        """CON-01: fold_quantize=False is the only representation litert_torch accepts."""
        prepared = MagicMock(name="prepared_module")
        quantized = MagicMock(name="quantized_module")

        with patch(
            "app.services.qat_service.convert_pt2e", return_value=quantized
        ) as mock_convert:
            result = service._convert(prepared)

        mock_convert.assert_called_once_with(prepared, fold_quantize=False)
        assert result is quantized

    def test_fold_quantize_true_not_used(self, service: QATService) -> None:
        """Sanity check: fold_quantize=True must never appear in the call."""
        with patch("app.services.qat_service.convert_pt2e") as mock_convert:
            mock_convert.return_value = MagicMock()
            service._convert(MagicMock())

        _, kwargs = mock_convert.call_args
        assert kwargs.get("fold_quantize") is False


# ── TFLite Export ─────────────────────────────────────────────────────────────


class TestTFLiteExport:
    def test_litert_torch_convert_called(
        self, service: QATService, tmp_path: "pytest.TempPathFactory"
    ) -> None:
        quantized = MagicMock()
        sample = (MagicMock(),)
        edge_model = MagicMock()
        mock_litert = MagicMock(convert=MagicMock(return_value=edge_model))

        with patch.dict(sys.modules, {"litert_torch": mock_litert}):
            service._export_tflite(quantized, sample, str(tmp_path))
            # Assert inside the block — sys.modules["litert_torch"] is restored on exit
            mock_litert.convert.assert_called_once_with(quantized, sample)

    def test_edge_model_export_called_with_output_path(
        self, service: QATService, tmp_path: "pytest.TempPathFactory"
    ) -> None:
        edge_model = MagicMock()

        with patch.dict(sys.modules, {"litert_torch": MagicMock(convert=MagicMock(return_value=edge_model))}):
            result = service._export_tflite(MagicMock(), (MagicMock(),), str(tmp_path))

        edge_model.export.assert_called_once_with(result)
        assert result.endswith("model_int8.tflite")

    def test_tflite_path_in_output_dir(
        self, service: QATService, tmp_path: "pytest.TempPathFactory"
    ) -> None:
        edge_model = MagicMock()

        with patch.dict(sys.modules, {"litert_torch": MagicMock(convert=MagicMock(return_value=edge_model))}):
            result = service._export_tflite(MagicMock(), (MagicMock(),), str(tmp_path))

        assert result.startswith(str(tmp_path))


# ── S3 Upload ─────────────────────────────────────────────────────────────────


class TestS3Upload:
    def test_upload_file_called_with_correct_bucket(
        self, service: QATService, s3: MagicMock, params: QATParams
    ) -> None:
        service._upload_tflite("/tmp/model_int8.tflite", params)
        s3.upload_file.assert_called_once()
        _, pos_args, _ = s3.upload_file.mock_calls[0]
        assert pos_args[1] == OUTPUT_BUCKET

    def test_upload_file_called_with_correct_key(
        self, service: QATService, s3: MagicMock, params: QATParams
    ) -> None:
        service._upload_tflite("/tmp/model_int8.tflite", params)
        _, pos_args, _ = s3.upload_file.mock_calls[0]
        assert pos_args[2] == f"{OUTPUT_PREFIX}/model_int8.tflite"

    def test_s3_uri_format_correct(
        self, service: QATService, s3: MagicMock, params: QATParams
    ) -> None:
        uri = service._upload_tflite("/tmp/model_int8.tflite", params)
        assert uri == f"s3://{OUTPUT_BUCKET}/{OUTPUT_PREFIX}/model_int8.tflite"


# ── MLflow Logging (FR-M-05) ──────────────────────────────────────────────────


class TestMLflowLogging:
    def test_source_run_id_logged_as_param(
        self, service: QATService, params: QATParams
    ) -> None:
        with patch("app.services.qat_service.MlflowClient") as mock_client_cls:
            client = mock_client_cls.return_value
            service._log_run("run-001", params, "s3://bucket/key.tflite")

        param_keys = [c.args[1] for c in client.log_param.call_args_list]
        assert "source_run_id" in param_keys

        source_call = next(c for c in client.log_param.call_args_list if c.args[1] == "source_run_id")
        assert source_call.args[2] == SOURCE_RUN_ID

    def test_quantization_mode_qat_logged(
        self, service: QATService, params: QATParams
    ) -> None:
        with patch("app.services.qat_service.MlflowClient") as mock_client_cls:
            client = mock_client_cls.return_value
            service._log_run("run-001", params, "s3://bucket/key.tflite")

        param_map = {c.args[1]: c.args[2] for c in client.log_param.call_args_list}
        assert param_map.get("quantization_mode") == "qat"

    def test_quantization_scheme_per_tensor_logged(
        self, service: QATService, params: QATParams
    ) -> None:
        with patch("app.services.qat_service.MlflowClient") as mock_client_cls:
            client = mock_client_cls.return_value
            service._log_run("run-001", params, "s3://bucket/key.tflite")

        param_map = {c.args[1]: c.args[2] for c in client.log_param.call_args_list}
        assert param_map.get("quantization_scheme") == "per_tensor_int8"

    def test_fold_quantize_false_logged(
        self, service: QATService, params: QATParams
    ) -> None:
        with patch("app.services.qat_service.MlflowClient") as mock_client_cls:
            client = mock_client_cls.return_value
            service._log_run("run-001", params, "s3://bucket/key.tflite")

        param_map = {c.args[1]: c.args[2] for c in client.log_param.call_args_list}
        assert param_map.get("fold_quantize") == "False"

    def test_mlflow_error_swallowed_not_raised(
        self, service: QATService, params: QATParams
    ) -> None:
        with patch("app.services.qat_service.MlflowClient") as mock_client_cls:
            mock_client_cls.return_value.log_param.side_effect = Exception("MLflow down")
            # Must not raise
            service._log_run("run-001", params, "s3://bucket/key.tflite")

    def test_tflite_s3_uri_logged(
        self, service: QATService, params: QATParams
    ) -> None:
        uri = "s3://mlops-artifacts/qat/exp-001/model_int8.tflite"
        with patch("app.services.qat_service.MlflowClient") as mock_client_cls:
            client = mock_client_cls.return_value
            service._log_run("run-001", params, uri)

        param_map = {c.args[1]: c.args[2] for c in client.log_param.call_args_list}
        assert param_map.get("tflite_s3_uri") == uri


# ── DeviceResolution ──────────────────────────────────────────────────────────


class TestDeviceResolution:
    def test_explicit_device_returned_unchanged(self, service: QATService) -> None:
        result = QATService._resolve_device("cuda:1")
        assert result == "cuda:1"

    def test_cpu_returned_when_cuda_unavailable(self) -> None:
        with patch("app.services.qat_service.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            result = QATService._resolve_device(None)
        assert result == "cpu"

    def test_cuda_returned_when_available(self) -> None:
        with patch("app.services.qat_service.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            result = QATService._resolve_device(None)
        assert result == "cuda"


# ── RunOrchestration ──────────────────────────────────────────────────────────


class TestRunOrchestration:
    """Verify run() calls steps in the correct order and returns QATResult."""

    def _run_with_mocks(
        self, service: QATService, params: QATParams, run_id: str = "qat-run-xyz"
    ) -> QATResult:
        mock_fp32 = MagicMock(name="fp32_module")
        mock_exported = MagicMock(name="exported_module")
        mock_prepared = MagicMock(name="prepared_module")
        mock_quantized = MagicMock(name="quantized_module")
        tflite_path = "/tmp/qat_out/model_int8.tflite"
        s3_uri = f"s3://{OUTPUT_BUCKET}/{OUTPUT_PREFIX}/model_int8.tflite"

        with (
            patch.object(service, "_resolve_checkpoint", return_value="/local/best.pt"),
            patch.object(service, "_load_headless_module", return_value=mock_fp32),
            patch.object(service, "_capture_graph", return_value=mock_exported),
            patch.object(service, "_prepare_qat", return_value=mock_prepared),
            patch.object(service, "_finetune"),
            patch.object(service, "_convert", return_value=mock_quantized),
            patch.object(service, "_export_tflite", return_value=tflite_path),
            patch.object(service, "_upload_tflite", return_value=s3_uri),
            patch.object(service, "_log_run"),
            patch("app.services.qat_service.mlflow") as mock_mlflow,
            patch("app.services.qat_service.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            mock_torch.zeros.return_value = MagicMock()
            active_run = MagicMock()
            active_run.info.run_id = run_id
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=active_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            return service.run(params)

    def test_returns_qat_result_instance(
        self, service: QATService, params: QATParams
    ) -> None:
        result = self._run_with_mocks(service, params)
        assert isinstance(result, QATResult)

    def test_result_mlflow_run_id(
        self, service: QATService, params: QATParams
    ) -> None:
        result = self._run_with_mocks(service, params, run_id="test-run-id")
        assert result.mlflow_run_id == "test-run-id"

    def test_result_source_run_id(
        self, service: QATService, params: QATParams
    ) -> None:
        result = self._run_with_mocks(service, params)
        assert result.source_run_id == SOURCE_RUN_ID

    def test_result_tflite_s3_uri(
        self, service: QATService, params: QATParams
    ) -> None:
        result = self._run_with_mocks(service, params)
        assert result.tflite_s3_uri.startswith("s3://")

    def test_mlflow_start_run_tags_include_source_run_id(
        self, service: QATService, params: QATParams
    ) -> None:
        with (
            patch.object(service, "_resolve_checkpoint", return_value="/local/best.pt"),
            patch.object(service, "_load_headless_module", return_value=MagicMock()),
            patch.object(service, "_capture_graph", return_value=MagicMock()),
            patch.object(service, "_prepare_qat", return_value=MagicMock()),
            patch.object(service, "_finetune"),
            patch.object(service, "_convert", return_value=MagicMock()),
            patch.object(service, "_export_tflite", return_value="/tmp/model.tflite"),
            patch.object(service, "_upload_tflite", return_value="s3://b/k"),
            patch.object(service, "_log_run"),
            patch("app.services.qat_service.mlflow") as mock_mlflow,
            patch("app.services.qat_service.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            mock_torch.zeros.return_value = MagicMock()
            active_run = MagicMock()
            active_run.info.run_id = "r"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=active_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            service.run(params)

        mock_mlflow.start_run.assert_called_once_with(
            tags={"source_run_id": SOURCE_RUN_ID}
        )

    def test_pipeline_step_order(
        self, service: QATService, params: QATParams
    ) -> None:
        """Steps must execute in dependency order."""
        call_order: list[str] = []

        def _tag(name: str) -> MagicMock:
            m = MagicMock()
            m.side_effect = lambda *a, **kw: call_order.append(name) or MagicMock()
            return m

        with (
            patch.object(service, "_resolve_checkpoint", side_effect=lambda *a, **kw: call_order.append("resolve") or "/local/best.pt"),
            patch.object(service, "_load_headless_module", side_effect=lambda *a, **kw: call_order.append("load") or MagicMock()),
            patch.object(service, "_capture_graph", side_effect=lambda *a, **kw: call_order.append("capture") or MagicMock()),
            patch.object(service, "_prepare_qat", side_effect=lambda *a, **kw: call_order.append("prepare") or MagicMock()),
            patch.object(service, "_finetune", side_effect=lambda *a, **kw: call_order.append("finetune")),
            patch.object(service, "_convert", side_effect=lambda *a, **kw: call_order.append("convert") or MagicMock()),
            patch.object(service, "_export_tflite", side_effect=lambda *a, **kw: call_order.append("export_tflite") or "/tmp/m.tflite"),
            patch.object(service, "_upload_tflite", side_effect=lambda *a, **kw: call_order.append("upload") or "s3://b/k"),
            patch.object(service, "_log_run", side_effect=lambda *a, **kw: call_order.append("log")),
            patch("app.services.qat_service.mlflow") as mock_mlflow,
            patch("app.services.qat_service.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            mock_torch.zeros.return_value = MagicMock()
            active_run = MagicMock()
            active_run.info.run_id = "r"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=active_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            service.run(params)

        assert call_order == [
            "resolve",
            "load",
            "capture",
            "prepare",
            "finetune",
            "convert",
            "export_tflite",
            "upload",
            "log",
        ]
