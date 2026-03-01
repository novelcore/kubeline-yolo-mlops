"""Background resource monitor for GPU, CPU, and RAM metrics.

Samples system resources at a configurable interval and logs them to
the active MLflow run. GPU metrics are logged only when pynvml is available
and a CUDA device is present. All errors from the sampling loop are caught
and logged as warnings so a monitoring failure never interrupts training.
"""

import logging
import threading
import time
from typing import Optional

import mlflow
import psutil

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# pynvml initialisation — optional; silently disabled when unavailable
# ---------------------------------------------------------------------------

try:
    import pynvml

    pynvml.nvmlInit()
    _GPU_COUNT: int = pynvml.nvmlDeviceGetCount()
    _GPU_AVAILABLE: bool = _GPU_COUNT > 0
except Exception as _nvml_exc:  # noqa: BLE001
    _GPU_AVAILABLE = False
    _GPU_COUNT = 0
    _logger.debug("pynvml unavailable — GPU metrics will not be logged: %s", _nvml_exc)


# ---------------------------------------------------------------------------
# ResourceMonitor
# ---------------------------------------------------------------------------


class ResourceMonitor:
    """Samples system resource metrics and logs them to the active MLflow run.

    Usage::

        monitor = ResourceMonitor(interval_sec=30)
        monitor.start()
        try:
            # ... training ...
        finally:
            monitor.stop()

    Parameters
    ----------
    interval_sec:
        Seconds between consecutive samples.
    """

    def __init__(self, interval_sec: int = 30) -> None:
        self._interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Shared mutable step counter so the background thread always increments it.
        self._step: list[int] = [0]

    # ------------------------------------------------------------------
    # Public control interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background sampling thread."""
        if self._thread is not None and self._thread.is_alive():
            _logger.warning("ResourceMonitor is already running; start() ignored.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="resource-monitor",
            daemon=True,
        )
        self._thread.start()
        _logger.debug(
            "ResourceMonitor started (interval=%ds, gpu_available=%s)",
            self._interval_sec,
            _GPU_AVAILABLE,
        )

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_sec + 5)
            if self._thread.is_alive():
                _logger.warning("ResourceMonitor thread did not stop cleanly.")
        _logger.debug("ResourceMonitor stopped after %d samples.", self._step[0])

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Main body of the background thread."""
        while not self._stop_event.is_set():
            self._sample()
            self._stop_event.wait(timeout=self._interval_sec)

    def _sample(self) -> None:
        """Collect one snapshot of system metrics and log to MLflow."""
        try:
            metrics: dict[str, float] = {}

            # ---- CPU / RAM ----
            vm = psutil.virtual_memory()
            metrics["system/ram_used_gb"] = vm.used / 1e9
            metrics["system/ram_percent"] = float(vm.percent)
            metrics["system/cpu_percent"] = psutil.cpu_percent(interval=None)

            # ---- GPU (one entry per device) ----
            if _GPU_AVAILABLE:
                for i in range(_GPU_COUNT):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        suffix = f"_gpu{i}" if _GPU_COUNT > 1 else ""
                        metrics[f"system/gpu_vram_used_gb{suffix}"] = mem.used / 1e9
                        metrics[f"system/gpu_vram_total_gb{suffix}"] = mem.total / 1e9
                        metrics[f"system/gpu_utilization_pct{suffix}"] = float(
                            util.gpu
                        )
                    except Exception as gpu_exc:  # noqa: BLE001
                        _logger.warning(
                            "Failed to read GPU %d metrics: %s", i, gpu_exc
                        )

            mlflow.log_metrics(metrics, step=self._step[0])
            self._step[0] += 1

        except Exception as exc:  # noqa: BLE001
            # Never crash training due to a monitoring failure.
            _logger.warning("ResourceMonitor sample failed: %s", exc)
