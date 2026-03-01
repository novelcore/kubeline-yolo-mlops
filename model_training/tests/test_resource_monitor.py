"""Tests for the ResourceMonitor background thread."""

import time
from unittest.mock import MagicMock, patch

import pytest

from app.services.resource_monitor import ResourceMonitor


@pytest.fixture()
def mock_psutil_vm() -> MagicMock:
    """Return a fake virtual_memory object."""
    vm = MagicMock()
    vm.used = 4_000_000_000
    vm.total = 16_000_000_000
    vm.percent = 25.0
    return vm


def _make_monitor(interval_sec: int = 1) -> ResourceMonitor:
    return ResourceMonitor(interval_sec=interval_sec)


class TestResourceMonitorLifecycle:
    def test_start_and_stop(self) -> None:
        """Monitor thread starts and stops cleanly."""
        monitor = _make_monitor(interval_sec=1)
        with (
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor.mlflow"),
            patch("app.services.resource_monitor._GPU_AVAILABLE", False),
        ):
            vm = MagicMock()
            vm.used = 1e9
            vm.percent = 10.0
            mock_psutil.virtual_memory.return_value = vm
            mock_psutil.cpu_percent.return_value = 5.0

            monitor.start()
            assert monitor._thread is not None
            assert monitor._thread.is_alive()

            time.sleep(0.1)
            monitor.stop()

            assert not monitor._thread.is_alive()

    def test_double_start_is_idempotent(self) -> None:
        """Calling start() twice does not spawn a second thread."""
        monitor = _make_monitor(interval_sec=60)
        with (
            patch("app.services.resource_monitor.psutil"),
            patch("app.services.resource_monitor.mlflow"),
            patch("app.services.resource_monitor._GPU_AVAILABLE", False),
        ):
            monitor.start()
            first_thread = monitor._thread
            monitor.start()  # second call should be a no-op
            assert monitor._thread is first_thread
            monitor.stop()


class TestResourceMonitorSampling:
    def test_cpu_and_ram_metrics_logged(self, mock_psutil_vm: MagicMock) -> None:
        """CPU and RAM metrics are logged to MLflow on each sample."""
        monitor = _make_monitor(interval_sec=60)

        with (
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor.mlflow") as mock_mlflow,
            patch("app.services.resource_monitor._GPU_AVAILABLE", False),
        ):
            mock_psutil.virtual_memory.return_value = mock_psutil_vm
            mock_psutil.cpu_percent.return_value = 42.0

            monitor._sample()

            mock_mlflow.log_metrics.assert_called_once()
            logged: dict = mock_mlflow.log_metrics.call_args[0][0]

            assert "system/ram_used_gb" in logged
            assert "system/ram_percent" in logged
            assert "system/cpu_percent" in logged
            assert logged["system/cpu_percent"] == 42.0
            assert logged["system/ram_percent"] == 25.0
            # No GPU keys when _GPU_AVAILABLE is False
            assert not any("gpu" in k for k in logged)

    def test_gpu_metrics_logged_when_available(self, mock_psutil_vm: MagicMock) -> None:
        """GPU VRAM and utilisation metrics are included when pynvml is available."""
        monitor = _make_monitor(interval_sec=60)

        mock_handle = MagicMock()
        mock_mem = MagicMock()
        mock_mem.used = 8_000_000_000
        mock_mem.total = 40_000_000_000
        mock_util = MagicMock()
        mock_util.gpu = 87

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

        with (
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor.mlflow") as mock_mlflow,
            patch("app.services.resource_monitor._GPU_AVAILABLE", True),
            patch("app.services.resource_monitor._GPU_COUNT", 1),
            patch("app.services.resource_monitor.pynvml", mock_pynvml, create=True),
        ):
            mock_psutil.virtual_memory.return_value = mock_psutil_vm
            mock_psutil.cpu_percent.return_value = 10.0

            monitor._sample()

            logged: dict = mock_mlflow.log_metrics.call_args[0][0]
            assert "system/gpu_vram_used_gb" in logged
            assert "system/gpu_vram_total_gb" in logged
            assert "system/gpu_utilization_pct" in logged
            assert logged["system/gpu_utilization_pct"] == 87.0

    def test_multi_gpu_metrics_have_suffix(self, mock_psutil_vm: MagicMock) -> None:
        """With multiple GPUs each metric key gets a _gpu{i} suffix."""
        monitor = _make_monitor(interval_sec=60)

        mock_mem = MagicMock()
        mock_mem.used = 4_000_000_000
        mock_mem.total = 40_000_000_000
        mock_util = MagicMock()
        mock_util.gpu = 50

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = MagicMock()
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

        with (
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor.mlflow") as mock_mlflow,
            patch("app.services.resource_monitor._GPU_AVAILABLE", True),
            patch("app.services.resource_monitor._GPU_COUNT", 2),
            patch("app.services.resource_monitor.pynvml", mock_pynvml, create=True),
        ):
            mock_psutil.virtual_memory.return_value = mock_psutil_vm
            mock_psutil.cpu_percent.return_value = 10.0

            monitor._sample()

            logged: dict = mock_mlflow.log_metrics.call_args[0][0]
            assert "system/gpu_vram_used_gb_gpu0" in logged
            assert "system/gpu_vram_used_gb_gpu1" in logged

    def test_mlflow_failure_does_not_raise(self, mock_psutil_vm: MagicMock) -> None:
        """A broken MLflow connection must not crash the monitor thread."""
        monitor = _make_monitor(interval_sec=60)

        with (
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor.mlflow") as mock_mlflow,
            patch("app.services.resource_monitor._GPU_AVAILABLE", False),
        ):
            mock_psutil.virtual_memory.return_value = mock_psutil_vm
            mock_psutil.cpu_percent.return_value = 5.0
            mock_mlflow.log_metrics.side_effect = ConnectionError("MLflow down")

            # Must not raise
            monitor._sample()

    def test_step_counter_increments(self, mock_psutil_vm: MagicMock) -> None:
        """The step counter increments with each sample."""
        monitor = _make_monitor(interval_sec=60)

        with (
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor.mlflow"),
            patch("app.services.resource_monitor._GPU_AVAILABLE", False),
        ):
            mock_psutil.virtual_memory.return_value = mock_psutil_vm
            mock_psutil.cpu_percent.return_value = 0.0

            assert monitor._step[0] == 0
            monitor._sample()
            assert monitor._step[0] == 1
            monitor._sample()
            assert monitor._step[0] == 2
