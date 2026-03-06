"""Tests for ResourceMonitor.collect().

ResourceMonitor is now a synchronous, on-demand sampler.  There is no
background thread, no run_id gating, and no step counter.  Tests call
collect() directly and assert on the returned dict.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.services.resource_monitor import ResourceMonitor


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_psutil_vm() -> MagicMock:
    """Return a fake virtual_memory object."""
    vm = MagicMock()
    vm.used = 4_000_000_000
    vm.total = 16_000_000_000
    vm.percent = 25.0
    return vm


def _make_monitor(gpu_index: "int | None" = None) -> ResourceMonitor:
    return ResourceMonitor(gpu_index=gpu_index)


# ---------------------------------------------------------------------------
# CPU / RAM sampling
# ---------------------------------------------------------------------------


class TestCpuRamSampling:
    def test_cpu_and_ram_metrics_returned(self, mock_psutil_vm: MagicMock) -> None:
        """collect() returns CPU and RAM metrics as a dict."""
        monitor = _make_monitor(gpu_index=None)

        with (
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor._GPU_AVAILABLE", False),
        ):
            mock_psutil.virtual_memory.return_value = mock_psutil_vm
            mock_psutil.cpu_percent.return_value = 42.0

            result = monitor.collect()

        assert "system/ram_used_gb" in result
        assert "system/ram_percent" in result
        assert "system/cpu_percent" in result
        assert result["system/cpu_percent"] == 42.0
        assert result["system/ram_percent"] == 25.0
        assert result["system/ram_used_gb"] == pytest.approx(4.0)

    def test_no_gpu_keys_when_gpu_index_is_none(self, mock_psutil_vm: MagicMock) -> None:
        """No GPU keys appear in the result when gpu_index=None."""
        monitor = _make_monitor(gpu_index=None)

        with (
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor._GPU_AVAILABLE", True),
        ):
            mock_psutil.virtual_memory.return_value = mock_psutil_vm
            mock_psutil.cpu_percent.return_value = 5.0

            result = monitor.collect()

        assert not any("gpu" in k for k in result)

    def test_returns_dict_of_floats(self, mock_psutil_vm: MagicMock) -> None:
        """All values in the returned dict are plain Python floats."""
        monitor = _make_monitor(gpu_index=None)

        with (
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor._GPU_AVAILABLE", False),
        ):
            mock_psutil.virtual_memory.return_value = mock_psutil_vm
            mock_psutil.cpu_percent.return_value = 10.0

            result = monitor.collect()

        for key, value in result.items():
            assert isinstance(value, float), f"{key} value is not a float: {type(value)}"


# ---------------------------------------------------------------------------
# GPU sampling
# ---------------------------------------------------------------------------


class TestGpuSampling:
    def _make_pynvml_mock(
        self,
        vram_used: int = 8_000_000_000,
        vram_total: int = 40_000_000_000,
        utilization: int = 87,
    ) -> MagicMock:
        mock_mem = MagicMock()
        mock_mem.used = vram_used
        mock_mem.total = vram_total
        mock_util = MagicMock()
        mock_util.gpu = utilization
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = MagicMock()
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
        return mock_pynvml

    def test_gpu_metrics_included_when_index_provided(
        self, mock_psutil_vm: MagicMock
    ) -> None:
        """GPU VRAM and utilisation appear in the result when gpu_index is set."""
        monitor = _make_monitor(gpu_index=0)
        mock_pynvml = self._make_pynvml_mock()

        with (
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor._GPU_AVAILABLE", True),
            patch("app.services.resource_monitor.pynvml", mock_pynvml, create=True),
        ):
            mock_psutil.virtual_memory.return_value = mock_psutil_vm
            mock_psutil.cpu_percent.return_value = 10.0

            result = monitor.collect()

        assert "system/gpu_vram_used_gb" in result
        assert "system/gpu_vram_total_gb" in result
        assert "system/gpu_utilization_pct" in result
        assert result["system/gpu_utilization_pct"] == 87.0
        assert result["system/gpu_vram_used_gb"] == pytest.approx(8.0)
        assert result["system/gpu_vram_total_gb"] == pytest.approx(40.0)

    def test_correct_device_index_queried(self, mock_psutil_vm: MagicMock) -> None:
        """pynvml is called with the gpu_index supplied at construction time."""
        monitor = _make_monitor(gpu_index=0)
        mock_pynvml = self._make_pynvml_mock()

        with (
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor._GPU_AVAILABLE", True),
            patch("app.services.resource_monitor.pynvml", mock_pynvml, create=True),
        ):
            mock_psutil.virtual_memory.return_value = mock_psutil_vm
            mock_psutil.cpu_percent.return_value = 10.0
            monitor.collect()

        mock_pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)

    def test_gpu_metrics_absent_when_gpu_index_none(
        self, mock_psutil_vm: MagicMock
    ) -> None:
        """GPU metrics are NOT in the result when gpu_index=None, even if pynvml works."""
        monitor = _make_monitor(gpu_index=None)
        mock_pynvml = self._make_pynvml_mock()

        with (
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor._GPU_AVAILABLE", True),
            patch("app.services.resource_monitor.pynvml", mock_pynvml, create=True),
        ):
            mock_psutil.virtual_memory.return_value = mock_psutil_vm
            mock_psutil.cpu_percent.return_value = 5.0

            result = monitor.collect()

        assert not any("gpu" in k for k in result)
        mock_pynvml.nvmlDeviceGetHandleByIndex.assert_not_called()

    def test_pynvml_error_is_swallowed_cpu_metrics_still_returned(
        self, mock_psutil_vm: MagicMock
    ) -> None:
        """A pynvml error during GPU sampling does not raise; CPU/RAM are still returned."""
        monitor = _make_monitor(gpu_index=0)

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = RuntimeError("GPU lost")

        with (
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor._GPU_AVAILABLE", True),
            patch("app.services.resource_monitor.pynvml", mock_pynvml, create=True),
        ):
            mock_psutil.virtual_memory.return_value = mock_psutil_vm
            mock_psutil.cpu_percent.return_value = 10.0

            result = monitor.collect()

        # CPU/RAM metrics must still be present
        assert "system/cpu_percent" in result
        assert "system/ram_used_gb" in result
        # No GPU keys because the pynvml call failed
        assert not any("gpu" in k for k in result)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_psutil_error_returns_empty_dict(self) -> None:
        """An unexpected psutil failure returns an empty dict, not an exception."""
        monitor = _make_monitor(gpu_index=None)

        with patch(
            "app.services.resource_monitor.psutil"
        ) as mock_psutil:
            mock_psutil.virtual_memory.side_effect = OSError("no proc fs")

            result = monitor.collect()

        assert result == {}
