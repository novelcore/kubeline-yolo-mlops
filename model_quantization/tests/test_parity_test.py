"""Tests for ParityTestService (FR-M-03).

Covers:
- ParityReport model construction
- Frame loading: seeded sampling, PIL resize, CHW conversion
- Images directory discovery: val > train > images priority
- FP32 inference: headless (head-excluded) vs full-model forward, mode-aware
- TFLite inference: uint8 / int8 / float32 I/O with tensor quant params
- AutoBackend PTQ path: both .pt and .tflite routed through AutoBackend
- Max-abs-error computation: matching shapes, shape mismatch → 1.0
- Report save: JSON written, content correct
- Run orchestration: parity_passed reflects threshold, max_abs_error populated
- Skip behaviour: no fp32 checkpoint → skipped (parity_passed=True, frames=0)
"""

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import ANY, MagicMock, call, patch

import numpy as np
import pytest

from app.models.quantization import ParityReport
from app.services.parity_test import ParityTestError, ParityTestService

DATASET_DIR = "/data/dataset"
OUTPUT_DIR = "/tmp/quant_out"
TFLITE_PATH = "/tmp/model_int8.tflite"
CHECKPOINT_PATH = "/local/best.pt"
IMAGE_SIZE = 320
PARITY_FRAMES = 4
SEED = 42
THRESHOLD = 0.05


@pytest.fixture
def service() -> ParityTestService:
    return ParityTestService()


def _make_frames(n: int = PARITY_FRAMES, size: int = IMAGE_SIZE) -> list[np.ndarray]:
    """Synthetic NCHW float32 frames for testing."""
    rng = np.random.default_rng(SEED)
    return [rng.random((1, 3, size, size), dtype=np.float32) for _ in range(n)]


# ── ParityReport model ────────────────────────────────────────────────────────


class TestParityReport:
    def test_passed_report(self) -> None:
        r = ParityReport(
            parity_passed=True, max_abs_error=0.02, threshold=0.05, frames_tested=4
        )
        assert r.parity_passed is True
        assert r.max_abs_error == pytest.approx(0.02)

    def test_failed_report(self) -> None:
        r = ParityReport(
            parity_passed=False, max_abs_error=0.08, threshold=0.05, frames_tested=4
        )
        assert r.parity_passed is False

    def test_zero_frames_allowed(self) -> None:
        r = ParityReport(
            parity_passed=True, max_abs_error=0.0, threshold=0.05, frames_tested=0
        )
        assert r.frames_tested == 0


# ── Images directory discovery ────────────────────────────────────────────────


class TestFindImagesDir:
    def test_prefers_images_val(
        self, service: ParityTestService, tmp_path: Path
    ) -> None:
        val = tmp_path / "images" / "val"
        train = tmp_path / "images" / "train"
        val.mkdir(parents=True)
        train.mkdir(parents=True)
        found = service._find_images_dir(str(tmp_path))
        assert found == str(val)

    def test_falls_back_to_images_train(
        self, service: ParityTestService, tmp_path: Path
    ) -> None:
        train = tmp_path / "images" / "train"
        train.mkdir(parents=True)
        found = service._find_images_dir(str(tmp_path))
        assert found == str(train)

    def test_falls_back_to_images_root(
        self, service: ParityTestService, tmp_path: Path
    ) -> None:
        images = tmp_path / "images"
        images.mkdir()
        found = service._find_images_dir(str(tmp_path))
        assert found == str(images)

    def test_raises_when_no_images_dir(
        self, service: ParityTestService, tmp_path: Path
    ) -> None:
        with pytest.raises(ParityTestError, match="No images directory"):
            service._find_images_dir(str(tmp_path))


# ── Frame loading ─────────────────────────────────────────────────────────────


class TestLoadFrames:
    def _make_dataset(self, tmp_path: Path, n: int = 8) -> Path:
        images = tmp_path / "images" / "val"
        images.mkdir(parents=True)
        from PIL import Image

        for i in range(n):
            # Each image has a unique pixel value so content differs by file
            img = Image.fromarray(np.full((64, 64, 3), i * 10, dtype=np.uint8))
            img.save(images / f"frame_{i:03d}.jpg")
        return tmp_path

    def test_returns_requested_count(
        self, service: ParityTestService, tmp_path: Path
    ) -> None:
        self._make_dataset(tmp_path, n=8)
        frames = service._load_frames(str(tmp_path), 32, 4, SEED)
        assert len(frames) == 4

    def test_returns_all_when_fewer_images_than_count(
        self, service: ParityTestService, tmp_path: Path
    ) -> None:
        self._make_dataset(tmp_path, n=3)
        frames = service._load_frames(str(tmp_path), 32, 10, SEED)
        assert len(frames) == 3

    def test_frames_are_nchw(self, service: ParityTestService, tmp_path: Path) -> None:
        self._make_dataset(tmp_path)
        frames = service._load_frames(str(tmp_path), 32, 2, SEED)
        assert frames[0].ndim == 4
        assert frames[0].shape == (1, 3, 32, 32)

    def test_frames_normalised_0_1(
        self, service: ParityTestService, tmp_path: Path
    ) -> None:
        self._make_dataset(tmp_path)
        frames = service._load_frames(str(tmp_path), 32, 2, SEED)
        assert frames[0].min() >= 0.0
        assert frames[0].max() <= 1.0

    def test_seeded_sampling_is_reproducible(
        self, service: ParityTestService, tmp_path: Path
    ) -> None:
        self._make_dataset(tmp_path, n=10)
        frames_a = service._load_frames(str(tmp_path), 32, 4, seed=7)
        frames_b = service._load_frames(str(tmp_path), 32, 4, seed=7)
        for a, b in zip(frames_a, frames_b):
            np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_order(
        self, service: ParityTestService, tmp_path: Path
    ) -> None:
        self._make_dataset(tmp_path, n=10)
        frames_a = service._load_frames(str(tmp_path), 32, 4, seed=1)
        frames_b = service._load_frames(str(tmp_path), 32, 4, seed=2)
        # At least one frame should differ (very high probability with 10 images, 4 picks)
        any_diff = any(not np.array_equal(a, b) for a, b in zip(frames_a, frames_b))
        assert any_diff

    def test_raises_when_no_images(
        self, service: ParityTestService, tmp_path: Path
    ) -> None:
        (tmp_path / "images" / "val").mkdir(parents=True)
        with pytest.raises(ParityTestError, match="No images found"):
            service._load_frames(str(tmp_path), 32, 4, SEED)


# ── Max-abs-error computation ─────────────────────────────────────────────────


class TestComputeMaxAbsError:
    def test_identical_outputs_zero_error(self, service: ParityTestService) -> None:
        frames = [np.ones(10, dtype=np.float32)] * 4
        err = service._compute_max_abs_error(frames, frames)
        assert err == pytest.approx(0.0)

    def test_computes_correct_max(self, service: ParityTestService) -> None:
        fp32 = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        tfl = [np.array([1.02, 1.97, 3.05], dtype=np.float32)]
        err = service._compute_max_abs_error(fp32, tfl)
        assert err == pytest.approx(0.05, abs=1e-6)

    def test_takes_max_across_frames(self, service: ParityTestService) -> None:
        fp32 = [
            np.array([1.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        ]
        tfl = [
            np.array([1.01], dtype=np.float32),
            np.array([1.08], dtype=np.float32),
        ]
        err = service._compute_max_abs_error(fp32, tfl)
        assert err == pytest.approx(0.08, abs=1e-6)

    def test_shape_mismatch_returns_1(self, service: ParityTestService) -> None:
        fp32 = [np.ones(10, dtype=np.float32)]
        tfl = [np.ones(5, dtype=np.float32)]
        err = service._compute_max_abs_error(fp32, tfl)
        assert err == 1.0

    def test_count_mismatch_raises(self, service: ParityTestService) -> None:
        fp32 = [np.ones(4)] * 3
        tfl = [np.ones(4)] * 2
        with pytest.raises(ParityTestError, match="count mismatch"):
            service._compute_max_abs_error(fp32, tfl)


# ── TFLite inference ──────────────────────────────────────────────────────────


class TestTFLiteInference:
    def _build_mock_interpreter(self, input_dtype: np.dtype) -> MagicMock:
        interp = MagicMock(name="Interpreter")
        interp.get_input_details.return_value = [
            {
                "index": 0,
                "dtype": input_dtype,
                "shape": [1, 32, 32, 3],
                "quantization": (1.0 / 255.0, -128 if input_dtype == np.int8 else 0),
            }
        ]
        # Float32 output — no dequantization applied.
        interp.get_output_details.return_value = [
            {"index": 1, "dtype": np.float32, "quantization": (0.0, 0)}
        ]
        interp.get_tensor.return_value = np.zeros((1, 8), dtype=np.float32)
        return interp

    def _run_with_mock(self, service, frames, input_dtype):
        mock_interp = self._build_mock_interpreter(input_dtype)
        mock_litert = MagicMock()
        mock_litert.Interpreter.return_value = mock_interp
        with patch.dict(
            "sys.modules",
            {"ai_edge_litert": mock_litert, "ai_edge_litert.interpreter": mock_litert},
        ):
            return service._run_tflite_inference(TFLITE_PATH, frames, 32), mock_interp

    def test_float32_input_no_dtype_conversion(
        self, service: ParityTestService
    ) -> None:
        frames = _make_frames(2, 32)
        outputs, interp = self._run_with_mock(service, frames, np.float32)
        assert len(outputs) == 2
        call_args = interp.set_tensor.call_args_list
        assert call_args[0][0][1].dtype == np.float32

    def test_uint8_input_scaled(self, service: ParityTestService) -> None:
        frames = _make_frames(1, 32)
        _, interp = self._run_with_mock(service, frames, np.uint8)
        tensor_passed = interp.set_tensor.call_args_list[0][0][1]
        assert tensor_passed.dtype == np.uint8

    def test_int8_input_scaled(self, service: ParityTestService) -> None:
        frames = _make_frames(1, 32)
        _, interp = self._run_with_mock(service, frames, np.int8)
        tensor_passed = interp.set_tensor.call_args_list[0][0][1]
        assert tensor_passed.dtype == np.int8

    def test_returns_flat_concatenated_outputs(
        self, service: ParityTestService
    ) -> None:
        frames = _make_frames(3, 32)
        outputs, _ = self._run_with_mock(service, frames, np.float32)
        for out in outputs:
            assert out.ndim == 1

    def test_interpreter_invoked_per_frame(self, service: ParityTestService) -> None:
        frames = _make_frames(3, 32)
        _, interp = self._run_with_mock(service, frames, np.float32)
        assert interp.invoke.call_count == 3

    def test_int8_input_uses_tensor_quant_params(
        self, service: ParityTestService
    ) -> None:
        """Input quantization must use the tensor's scale/zero_point,
        mirroring AutoBackend (im / scale + zero_point), NOT a hardcoded
        *255-128 rule."""
        frames = [np.full((1, 3, 32, 32), 0.6, dtype=np.float32)]
        _, interp = self._run_with_mock(service, frames, np.int8)
        tensor_passed = interp.set_tensor.call_args_list[0][0][1]
        # scale=1/255, zero_point=-128 → 0.6 / (1/255) + (-128) = 153 - 128 = 25
        assert tensor_passed.dtype == np.int8
        assert int(tensor_passed.flat[0]) == 25

    def test_int8_output_dequantized(self, service: ParityTestService) -> None:
        """Int8 output tensors must be dequantized with (x - zp) * scale."""
        interp = MagicMock(name="Interpreter")
        interp.get_input_details.return_value = [
            {
                "index": 0,
                "dtype": np.float32,
                "shape": [1, 32, 32, 3],
                "quantization": (0.0, 0),
            }
        ]
        interp.get_output_details.return_value = [
            {"index": 1, "dtype": np.int8, "quantization": (0.5, -10)}
        ]
        interp.get_tensor.return_value = np.array([[0, 10]], dtype=np.int8)
        mock_litert = MagicMock()
        mock_litert.Interpreter.return_value = interp
        frames = _make_frames(1, 32)
        with patch.dict(
            "sys.modules",
            {"ai_edge_litert": mock_litert, "ai_edge_litert.interpreter": mock_litert},
        ):
            outputs = service._run_tflite_inference(TFLITE_PATH, frames, 32)
        # (0 - (-10)) * 0.5 = 5.0 ; (10 - (-10)) * 0.5 = 10.0
        np.testing.assert_allclose(outputs[0], [5.0, 10.0])


# ── AutoBackend PTQ inference ─────────────────────────────────────────────────


class TestAutoBackendInference:
    def _patch_autobackend(self, backend: MagicMock) -> Any:
        """Patch ultralytics.nn.autobackend.AutoBackend for _run_autobackend_inference.

        The service imports AutoBackend lazily inside the method, so patch the
        source module attribute (a real ultralytics import is not required —
        we inject a fake module into sys.modules).
        """
        mock_module = MagicMock()
        mock_module.AutoBackend.return_value = backend
        return patch.dict("sys.modules", {"ultralytics.nn.autobackend": mock_module})

    def test_pt_and_tflite_both_routed_through_autobackend(
        self, service: ParityTestService
    ) -> None:
        """PTQ parity runs BOTH weight types through AutoBackend (identical
        preprocessing + int8 dequant + coordinate denorm)."""
        import torch as real_torch

        frames = _make_frames(1, 32)
        backend = MagicMock(name="AutoBackend")
        backend.return_value = real_torch.zeros(1, 38, 8400)
        with self._patch_autobackend(backend) as _:
            outputs = service._run_autobackend_inference(CHECKPOINT_PATH, frames)
        assert len(outputs) == 1
        assert outputs[0].shape == (1 * 38 * 8400,)

    def test_autobackend_extracts_prediction_from_tuple(
        self, service: ParityTestService
    ) -> None:
        import torch as real_torch

        frames = _make_frames(2, 32)
        pred = real_torch.ones(1, 38, 8400)
        backend = MagicMock(name="AutoBackend")
        # AutoBackend may return (pred, extras)
        backend.return_value = (pred, {"extras": 1})
        with self._patch_autobackend(backend):
            outputs = service._run_autobackend_inference(TFLITE_PATH, frames)
        assert len(outputs) == 2
        np.testing.assert_allclose(outputs[0], pred.numpy().flatten())

    def test_full_ptq_run_uses_autobackend(self, service: ParityTestService) -> None:
        """run(headless=False) drives _run_autobackend_inference for both sides,
        NOT the headless _run_fp32_inference / _run_tflite_inference paths."""
        frames = _make_frames(1, IMAGE_SIZE)
        vec = [np.ones(6, dtype=np.float32)]
        with (
            patch.object(service, "_load_frames", return_value=frames),
            patch.object(
                service, "_run_autobackend_inference", return_value=vec
            ) as mock_ab,
            patch.object(service, "_run_fp32_inference") as mock_fp32,
            patch.object(service, "_run_tflite_inference") as mock_tfl,
        ):
            report = service.run(
                tflite_path=TFLITE_PATH,
                fp32_checkpoint_path=CHECKPOINT_PATH,
                dataset_dir=DATASET_DIR,
                image_size=IMAGE_SIZE,
                parity_frames=1,
                seed=SEED,
                max_abs_error_threshold=THRESHOLD,
                headless=False,
            )
        assert report.parity_passed is True
        # AutoBackend used for both .pt and .tflite; headless helpers unused.
        assert mock_ab.call_count == 2
        mock_fp32.assert_not_called()
        mock_tfl.assert_not_called()


# ── FP32 inference (headless) ─────────────────────────────────────────────────


class TestFP32Inference:
    def _build_mock_yolo_module(self, num_layers: int = 4) -> MagicMock:
        """Build a mock YOLO module with num_layers sub-layers."""
        layers = []
        for i in range(num_layers):
            layer = MagicMock(name=f"layer_{i}")
            layer.f = -1
            layer.return_value = MagicMock(name=f"out_{i}")
            layers.append(layer)
        # Last layer is the detection head — should be excluded
        head = MagicMock(name="detection_head")
        layers.append(head)

        module = MagicMock(name="yolo_module")
        module.model = MagicMock()
        module.model.__getitem__ = MagicMock(
            side_effect=lambda i: layers[i] if i != -1 else layers[-1]
        )
        module.model.children.return_value = iter(layers)

        return module, layers, head

    def test_head_exclusion_applied(self, service: ParityTestService) -> None:
        """Detection head must be excluded from forward pass (CON-03)."""
        frames = _make_frames(1, 32)
        module, layers, head = self._build_mock_yolo_module()

        mock_yolo = MagicMock()
        mock_yolo.model = module

        with (
            patch("app.services.parity_test.YOLO", return_value=mock_yolo),
            patch("app.services.parity_test.torch") as mock_torch,
        ):
            mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
            mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
            mock_torch.from_numpy.side_effect = lambda x: x
            mock_torch.cat.side_effect = lambda parts, dim: parts[0] if parts else None
            try:
                service._run_fp32_inference(CHECKPOINT_PATH, frames, headless=True)
            except Exception:
                pass

        # Head layer should never have been called
        head.assert_not_called()

    def test_model_set_to_eval(self, service: ParityTestService) -> None:
        frames = _make_frames(1, 32)
        mock_yolo = MagicMock()
        mock_module = MagicMock()
        mock_yolo.model = mock_module

        with (
            patch("app.services.parity_test.YOLO", return_value=mock_yolo),
            patch("app.services.parity_test.torch") as mock_torch,
        ):
            mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
            mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
            mock_torch.from_numpy.side_effect = lambda x: x
            try:
                service._run_fp32_inference(CHECKPOINT_PATH, frames, headless=True)
            except Exception:
                pass

        mock_module.eval.assert_called_once()

    def test_full_model_forward_calls_module(self, service: ParityTestService) -> None:
        """PTQ direct FP32 forward (headless=False) runs the FULL model — the
        head is NOT excluded and the module is called directly."""
        import torch as real_torch

        frames = _make_frames(1, 32)
        pred = real_torch.zeros(1, 38, 8400)
        mock_yolo = MagicMock()
        mock_module = MagicMock()
        # eval() returns the module; calling it returns (pred, extras) tuple
        mock_module.eval.return_value = mock_module
        mock_module.return_value = (pred, {"extras": 1})
        mock_yolo.model = mock_module

        with patch("app.services.parity_test.YOLO", return_value=mock_yolo):
            outputs = service._run_fp32_inference(
                CHECKPOINT_PATH, frames, headless=False
            )

        # Head exclusion flag must NOT be set for the full-model path.
        assert mock_module.model.__getitem__.call_count == 0
        assert outputs[0].shape == (1, 38, 8400)

    def test_extract_prediction_peels_nested_tuple(
        self, service: ParityTestService
    ) -> None:
        import torch as real_torch

        t = real_torch.ones(1, 38, 8400)
        # Pose-style nested output: ((tensor, proto), preds)
        nested = ((t, real_torch.zeros(1, 32, 160, 160)), {"p": 1})
        out = service._extract_prediction(nested)
        assert out is t


# ── Report save ───────────────────────────────────────────────────────────────


class TestSaveReport:
    def test_writes_json(self, service: ParityTestService, tmp_path: Path) -> None:
        report = ParityReport(
            parity_passed=True, max_abs_error=0.02, threshold=0.05, frames_tested=4
        )
        path = service.save_report(report, str(tmp_path))
        assert os.path.isfile(path)
        with open(path) as f:
            data = json.load(f)
        assert data["parity_passed"] is True
        assert data["max_abs_error"] == pytest.approx(0.02)
        assert data["frames_tested"] == 4

    def test_creates_output_dir_if_missing(
        self, service: ParityTestService, tmp_path: Path
    ) -> None:
        new_dir = tmp_path / "new_subdir"
        report = ParityReport(
            parity_passed=False, max_abs_error=0.1, threshold=0.05, frames_tested=4
        )
        path = service.save_report(report, str(new_dir))
        assert os.path.isfile(path)

    def test_filename_is_parity_report_json(
        self, service: ParityTestService, tmp_path: Path
    ) -> None:
        report = ParityReport(
            parity_passed=True, max_abs_error=0.0, threshold=0.05, frames_tested=2
        )
        path = service.save_report(report, str(tmp_path))
        assert Path(path).name == "parity_report.json"


# ── Run orchestration ─────────────────────────────────────────────────────────


class TestRunOrchestration:
    def _run_with_mocked_inference(
        self,
        service: ParityTestService,
        fp32_out: list[np.ndarray],
        tfl_out: list[np.ndarray],
        threshold: float = THRESHOLD,
    ) -> ParityReport:
        frames = _make_frames(len(fp32_out), IMAGE_SIZE)
        with (
            patch.object(service, "_load_frames", return_value=frames),
            patch.object(service, "_run_fp32_inference", return_value=fp32_out),
            patch.object(service, "_run_tflite_inference", return_value=tfl_out),
        ):
            return service.run(
                tflite_path=TFLITE_PATH,
                fp32_checkpoint_path=CHECKPOINT_PATH,
                dataset_dir=DATASET_DIR,
                image_size=IMAGE_SIZE,
                parity_frames=len(fp32_out),
                seed=SEED,
                max_abs_error_threshold=threshold,
            )

    def test_passes_when_error_below_threshold(
        self, service: ParityTestService
    ) -> None:
        fp32 = [np.array([1.0], dtype=np.float32)]
        tfl = [np.array([1.02], dtype=np.float32)]
        report = self._run_with_mocked_inference(service, fp32, tfl, threshold=0.05)
        assert report.parity_passed is True

    def test_fails_when_error_above_threshold(self, service: ParityTestService) -> None:
        fp32 = [np.array([1.0], dtype=np.float32)]
        tfl = [np.array([1.10], dtype=np.float32)]
        report = self._run_with_mocked_inference(service, fp32, tfl, threshold=0.05)
        assert report.parity_passed is False

    def test_exact_threshold_passes(self, service: ParityTestService) -> None:
        fp32 = [np.array([1.0], dtype=np.float32)]
        tfl = [np.array([1.05], dtype=np.float32)]
        report = self._run_with_mocked_inference(service, fp32, tfl, threshold=0.05)
        assert report.parity_passed is True

    def test_report_contains_frames_tested(self, service: ParityTestService) -> None:
        n = 3
        fp32 = [np.ones(4, dtype=np.float32)] * n
        tfl = [np.ones(4, dtype=np.float32)] * n
        report = self._run_with_mocked_inference(service, fp32, tfl)
        assert report.frames_tested == n

    def test_report_contains_threshold(self, service: ParityTestService) -> None:
        fp32 = [np.ones(4, dtype=np.float32)]
        tfl = [np.ones(4, dtype=np.float32)]
        report = self._run_with_mocked_inference(service, fp32, tfl, threshold=0.03)
        assert report.threshold == pytest.approx(0.03)

    def test_load_frames_called_with_seed(self, service: ParityTestService) -> None:
        frames = _make_frames(2)
        fp32 = [np.ones(4, dtype=np.float32)] * 2
        tfl = [np.ones(4, dtype=np.float32)] * 2
        with (
            patch.object(service, "_load_frames", return_value=frames) as mock_load,
            patch.object(service, "_run_fp32_inference", return_value=fp32),
            patch.object(service, "_run_tflite_inference", return_value=tfl),
        ):
            service.run(
                tflite_path=TFLITE_PATH,
                fp32_checkpoint_path=CHECKPOINT_PATH,
                dataset_dir=DATASET_DIR,
                image_size=IMAGE_SIZE,
                parity_frames=2,
                seed=99,
                max_abs_error_threshold=THRESHOLD,
            )
        mock_load.assert_called_once_with(DATASET_DIR, IMAGE_SIZE, 2, 99)

    def test_headless_flag_forwarded_to_fp32(self, service: ParityTestService) -> None:
        frames = _make_frames(1)
        fp32 = [np.ones(4, dtype=np.float32)]
        tfl = [np.ones(4, dtype=np.float32)]
        with (
            patch.object(service, "_load_frames", return_value=frames),
            patch.object(
                service, "_run_fp32_inference", return_value=fp32
            ) as mock_fp32,
            patch.object(service, "_run_tflite_inference", return_value=tfl),
        ):
            service.run(
                tflite_path=TFLITE_PATH,
                fp32_checkpoint_path=CHECKPOINT_PATH,
                dataset_dir=DATASET_DIR,
                image_size=IMAGE_SIZE,
                parity_frames=1,
                seed=SEED,
                max_abs_error_threshold=THRESHOLD,
                headless=True,
            )
        mock_fp32.assert_called_once_with(CHECKPOINT_PATH, frames, headless=True)
