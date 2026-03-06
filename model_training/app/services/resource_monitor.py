"""System resource sampler for GPU, CPU, and RAM metrics.

Collects a single snapshot of system resource usage on demand and returns
it as a plain dict.  The caller (an Ultralytics ``on_fit_epoch_end`` callback)
is responsible for logging the returned values to MLflow at the appropriate
epoch step.

GPU metrics are collected only when pynvml is available and a CUDA device
index has been provided.  All pynvml errors are caught and logged as warnings
so a monitoring failure never interrupts training.

Usage pattern
-------------
1. Construct ``ResourceMonitor(gpu_index=0)`` before training.
2. Call ``monitor.collect()`` from an Ultralytics ``on_fit_epoch_end`` callback.
3. Log the returned dict alongside model metrics via ``mlflow.log_metrics``.
"""

import logging
from typing import Optional

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
    pynvml = None  # type: ignore[assignment]
    _GPU_AVAILABLE = False
    _GPU_COUNT = 0
    _logger.debug("pynvml unavailable — GPU metrics will not be logged: %s", _nvml_exc)


def gpu_available() -> bool:
    """Return True if at least one GPU was detected at module load time."""
    return _GPU_AVAILABLE


# ---------------------------------------------------------------------------
# ResourceMonitor
# ---------------------------------------------------------------------------


class ResourceMonitor:
    """Collects a snapshot of system resource metrics on demand.

    Call :meth:`collect` from any Ultralytics epoch-end callback to get a
    dict of metric key/value pairs ready to pass to ``mlflow.log_metrics``.

    Parameters
    ----------
    gpu_index:
        Index of the GPU to sample (e.g. ``0`` for ``--device 0``).  When
        ``None``, GPU metrics are skipped entirely even if pynvml is available.
        Pass ``0`` for single-GPU training; pass the specific device index for
        multi-GPU setups where you want to monitor one device.
    """

    def __init__(self, gpu_index: Optional[int] = None) -> None:
        self._gpu_index = gpu_index
        # Seed psutil.cpu_percent so the first collect() call returns a
        # real measurement instead of 0.0 (psutil needs a prior baseline).
        psutil.cpu_percent(interval=None)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def collect(self) -> dict[str, float]:
        """Collect one snapshot of system metrics.

        Returns
        -------
        dict[str, float]
            Metric names mapped to their current values.  GPU keys are
            included only when a valid ``gpu_index`` was supplied at
            construction time and pynvml is available.  Returns an empty
            dict if an unexpected error occurs so the caller can always
            iterate the result safely.
        """
        try:
            metrics: dict[str, float] = {}

            # ---- CPU / RAM ----
            vm = psutil.virtual_memory()
            metrics["system/ram_used_gb"] = vm.used / 1e9
            metrics["system/ram_percent"] = float(vm.percent)
            metrics["system/cpu_percent"] = psutil.cpu_percent(interval=None)

            # ---- GPU — only the training device ----
            if _GPU_AVAILABLE and self._gpu_index is not None and pynvml is not None:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics["system/gpu_vram_used_gb"] = mem.used / 1e9
                    metrics["system/gpu_vram_total_gb"] = mem.total / 1e9
                    metrics["system/gpu_utilization_pct"] = float(util.gpu)
                except Exception as gpu_exc:  # noqa: BLE001
                    _logger.warning(
                        "Failed to read GPU %d metrics: %s", self._gpu_index, gpu_exc
                    )

            return metrics

        except Exception as exc:  # noqa: BLE001
            _logger.warning("ResourceMonitor.collect() failed: %s", exc)
            return {}
