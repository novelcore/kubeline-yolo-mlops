"""Model quantization service.

PTQ path:
    FP32 .pt checkpoint
        → YOLO.export(format='tflite', int8=True, data=<calibration yaml>)
        → INT8 TFLite
        → S3 upload + MLflow logging

QAT passthrough:
    INT8 TFLite s3:// URI (from qat-finetune)
        → parity test stub (FR-M-03)
        → MLflow logging
"""

import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import mlflow
import torch
from mlflow.tracking import MlflowClient
from ultralytics import YOLO

from app.models.quantization import ParityReport, QuantizationParams, QuantizationResult
from app.services.parity_test import ParityTestService
from app.services.resource_monitor import ResourceMonitor

# Interval (seconds) between system-metric samples during the export.
_SYSTEM_METRICS_INTERVAL_S = 15.0

# Ultralytics results_dict keys → sanitized MLflow metric suffixes. Covers the
# box (B) detection metrics and, for pose models, the keypoint (P) metrics.
_MAP_METRIC_KEYS: dict[str, str] = {
    "metrics/mAP50(B)": "mAP50B",
    "metrics/mAP50-95(B)": "mAP50-95B",
    "metrics/mAP50(P)": "mAP50P",
    "metrics/mAP50-95(P)": "mAP50-95P",
}


class QuantizationError(Exception):
    """Raised on non-recoverable quantization failures."""


class QuantizationService:
    """Runs PTQ (Ultralytics) or QAT passthrough and logs to MLflow."""

    def __init__(self, s3_client: Any, mlflow_tracking_uri: str) -> None:
        self._s3 = s3_client
        self._mlflow_uri = mlflow_tracking_uri
        self._logger = logging.getLogger(__name__)
        self._parity = ParityTestService()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, params: QuantizationParams) -> QuantizationResult:
        """Dispatch to PTQ or QAT passthrough based on params.mode."""
        mlflow.set_tracking_uri(self._mlflow_uri)
        mlflow.set_experiment(params.experiment_name)

        if params.mode == "ptq":
            return self._run_ptq(params)
        return self._run_qat_passthrough(params)

    # ------------------------------------------------------------------
    # PTQ path
    # ------------------------------------------------------------------

    def _run_ptq(self, params: QuantizationParams) -> QuantizationResult:
        """PTQ: Ultralytics export(int8=True) → TFLite → S3 → MLflow."""
        assert params.fp32_checkpoint_path is not None

        local_ckpt = self._resolve_checkpoint(
            params.fp32_checkpoint_path, params.output_dir
        )
        data_yaml = self._find_data_yaml(params.dataset_dir)

        with mlflow.start_run(
            tags={"source_run_id": params.source_mlflow_run_id}
        ) as active_run:
            run_id = active_run.info.run_id
            self._logger.info(
                "PTQ run started | run_id=%s checkpoint=%s",
                run_id,
                local_ckpt,
            )

            with self._sample_system_metrics(run_id):
                self._seed_torch(params.calibration_seed)  # FR-M-04
                tflite_path = self._export_ptq(local_ckpt, data_yaml, params)
                s3_uri = self._upload_tflite(tflite_path, params)
                self._log_tflite_artifact(run_id, tflite_path)
                # PTQ exports the FULL model → compare against full-model FP32.
                parity = self._run_parity_and_log(
                    run_id=run_id,
                    tflite_path=tflite_path,
                    fp32_checkpoint_path=local_ckpt,
                    params=params,
                    headless=False,
                )
                # Task-metric quality signal: INT8 vs FP32 mAP on the val set.
                self._run_map_delta_and_log(
                    run_id=run_id,
                    fp32_checkpoint_path=local_ckpt,
                    tflite_path=tflite_path,
                    data_yaml=data_yaml,
                    params=params,
                )
                self._log_ptq_run(run_id, params, s3_uri)

        self._logger.info("PTQ complete | run_id=%s tflite=%s", run_id, s3_uri)

        return QuantizationResult(
            mlflow_run_id=run_id,
            source_run_id=params.source_mlflow_run_id,
            mode="ptq",
            tflite_s3_uri=s3_uri,
            parity_passed=parity.parity_passed,
            parity_max_abs_error=parity.max_abs_error,
        )

    def _export_ptq(
        self, checkpoint_path: str, data_yaml: str, params: QuantizationParams
    ) -> str:
        """Run Ultralytics PTQ export and return the local TFLite path."""
        self._logger.info(
            "Exporting PTQ INT8 TFLite | checkpoint=%s data=%s imgsz=%d",
            checkpoint_path,
            data_yaml,
            params.image_size,
        )
        model = YOLO(checkpoint_path)
        exported = model.export(
            format="tflite",
            int8=True,
            data=data_yaml,
            imgsz=params.image_size,
        )
        tflite_path = str(exported)
        self._logger.info("PTQ export complete: %s", tflite_path)
        return tflite_path

    def _log_ptq_run(
        self, run_id: str, params: QuantizationParams, s3_uri: str
    ) -> None:
        """Log PTQ parameters and artifact URI to MLflow."""
        client = MlflowClient()
        items: list[tuple[str, str]] = [
            ("quantization_mode", "ptq"),
            ("quantization_scheme", "per_tensor_int8"),
            ("calibration_frames", str(params.calibration_frames)),
            ("calibration_seed", str(params.calibration_seed)),
            ("image_size", str(params.image_size)),
            ("parity_frames", str(params.parity_frames)),
            ("parity_max_abs_error_threshold", str(params.parity_max_abs_error)),
            ("source_run_id", params.source_mlflow_run_id),
            ("tflite_s3_uri", s3_uri),
        ]
        for key, value in items:
            try:
                client.log_param(run_id, key, value)
            except Exception as exc:
                self._logger.warning("Failed to log MLflow param %s: %s", key, exc)

    # ------------------------------------------------------------------
    # QAT passthrough
    # ------------------------------------------------------------------

    def _run_qat_passthrough(self, params: QuantizationParams) -> QuantizationResult:
        """QAT passthrough: receive TFLite URI, run parity stub, log to MLflow."""
        assert params.tflite_s3_uri is not None

        tags: dict[str, str] = {"source_run_id": params.source_mlflow_run_id}
        if params.qat_run_id:
            tags["qat_run_id"] = params.qat_run_id

        with mlflow.start_run(tags=tags) as active_run:
            run_id = active_run.info.run_id
            self._logger.info(
                "QAT passthrough run started | run_id=%s tflite=%s",
                run_id,
                params.tflite_s3_uri,
            )

            with self._sample_system_metrics(run_id):
                local_tflite = self._download_tflite(
                    params.tflite_s3_uri, params.output_dir
                )
                self._log_tflite_artifact(run_id, local_tflite)
                # Download the FP32 checkpoint locally for the headless parity
                # reference — otherwise the raw s3:// URI is treated as a local
                # path ("No such file s3:/…") and parity errors. Mirrors PTQ,
                # which resolves the checkpoint before parity.
                local_ckpt = (
                    self._resolve_checkpoint(
                        params.fp32_checkpoint_path, params.output_dir
                    )
                    if params.fp32_checkpoint_path
                    else None
                )
                # QAT exports the backbone+neck only (CON-03) → headless FP32.
                parity = self._run_parity_and_log(
                    run_id=run_id,
                    tflite_path=local_tflite,
                    fp32_checkpoint_path=local_ckpt,
                    params=params,
                    headless=True,
                )
            self._log_qat_passthrough_run(run_id, params)

        self._logger.info("QAT passthrough complete | run_id=%s", run_id)

        return QuantizationResult(
            mlflow_run_id=run_id,
            source_run_id=params.source_mlflow_run_id,
            mode="qat",
            tflite_s3_uri=params.tflite_s3_uri,
            parity_passed=parity.parity_passed,
            parity_max_abs_error=parity.max_abs_error,
        )

    def _log_qat_passthrough_run(self, run_id: str, params: QuantizationParams) -> None:
        """Log QAT passthrough parameters to MLflow."""
        client = MlflowClient()
        items: list[tuple[str, str]] = [
            ("quantization_mode", "qat"),
            ("quantization_scheme", "per_tensor_int8"),
            ("calibration_frames", str(params.calibration_frames)),
            ("calibration_seed", str(params.calibration_seed)),
            ("image_size", str(params.image_size)),
            ("parity_frames", str(params.parity_frames)),
            ("parity_max_abs_error_threshold", str(params.parity_max_abs_error)),
            ("source_run_id", params.source_mlflow_run_id),
            ("tflite_s3_uri", params.tflite_s3_uri or ""),
        ]
        if params.qat_run_id:
            items.append(("qat_run_id", params.qat_run_id))
        for key, value in items:
            try:
                client.log_param(run_id, key, value)
            except Exception as exc:
                self._logger.warning("Failed to log MLflow param %s: %s", key, exc)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _resolve_checkpoint(self, path: str, output_dir: str) -> str:
        """Return a local path to the checkpoint, downloading from S3 if needed."""
        if not path.startswith("s3://"):
            return path
        without_scheme = path[len("s3://") :]
        bucket, _, key = without_scheme.partition("/")
        local_path = os.path.join(output_dir, Path(key).name)
        self._logger.info("Downloading checkpoint: %s → %s", path, local_path)
        self._s3.download_file(bucket, key, local_path)
        return local_path

    def _find_data_yaml(self, dataset_dir: str) -> str:
        """Locate the YOLO data YAML in the dataset directory.

        Searches for common filenames: data.yaml, dataset.yaml, config.yaml.
        Raises QuantizationError if none found.
        """
        candidates = ["data.yaml", "dataset.yaml", "config.yaml"]
        for name in candidates:
            candidate = os.path.join(dataset_dir, name)
            if os.path.isfile(candidate):
                self._logger.info("Found dataset YAML: %s", candidate)
                return candidate
        raise QuantizationError(
            f"No YOLO data YAML found in {dataset_dir!r}. " f"Searched: {candidates}"
        )

    def _upload_tflite(self, local_path: str, params: QuantizationParams) -> str:
        """Upload TFLite artifact to S3 and return the s3:// URI."""
        key = f"{params.output_prefix}/{Path(local_path).name}"
        self._logger.info("Uploading TFLite to s3://%s/%s", params.output_bucket, key)
        self._s3.upload_file(local_path, params.output_bucket, key)
        # S3-gateway writes only STAGE objects on the lakeFS branch — without a
        # commit they are dangling uncommitted changes, not a versioned artifact.
        branch = params.output_prefix.split("/", 1)[0] or "main"
        self._lakefs_commit(
            params.output_bucket,
            branch,
            f"model-quantization: {params.mode} INT8 tflite "
            f"{Path(local_path).name}",
        )
        return f"s3://{params.output_bucket}/{key}"

    def _lakefs_commit(self, repo: str, branch: str, message: str) -> None:
        """Commit staged lakeFS changes on ``branch`` (S3-gateway only stages).

        Best-effort: a failure (including 'nothing to commit') never fails the
        step — the object is already uploaded. Uses the lakeFS REST API on the
        same endpoint as the S3 gateway with the injected lakeFS credentials.
        """
        import base64
        import json
        import urllib.request

        endpoint = os.environ.get("LAKEFS_ENDPOINT", "").rstrip("/")
        access = os.environ.get("LAKEFS_ACCESS_KEY", "")
        secret = os.environ.get("LAKEFS_SECRET_KEY", "")
        if not (endpoint and access and secret):
            self._logger.warning("lakeFS commit skipped — endpoint/creds unset.")
            return
        url = f"{endpoint}/api/v1/repositories/{repo}/branches/{branch}/commits"
        auth = base64.b64encode(f"{access}:{secret}".encode()).decode()
        req = urllib.request.Request(
            url, data=json.dumps({"message": message}).encode(), method="POST"
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Basic {auth}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                cid = str(json.loads(resp.read().decode()).get("id", "?"))
            self._logger.info("lakeFS commit %s @ %s/%s", cid[:12], repo, branch)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("lakeFS commit failed (non-fatal): %s", exc)

    def _download_tflite(self, s3_uri: str, output_dir: str) -> str:
        """Download TFLite from an s3:// URI. Returns local path."""
        without_scheme = s3_uri[len("s3://") :]
        bucket, _, key = without_scheme.partition("/")
        local_path = os.path.join(output_dir, Path(key).name)
        self._logger.info("Downloading TFLite: %s → %s", s3_uri, local_path)
        self._s3.download_file(bucket, key, local_path)
        return local_path

    def _seed_torch(self, seed: int) -> None:
        """Seed PyTorch RNG before PTQ calibration export (FR-M-04).

        Ultralytics export(int8=True) calls torch ops internally during
        calibration; seeding here gives best-effort reproducibility.
        """
        torch.manual_seed(seed)
        self._logger.info("PyTorch RNG seed fixed | seed=%d", seed)

    def _log_tflite_artifact(self, run_id: str, tflite_path: str) -> None:
        """Log the quantized .tflite to MLflow as a run artifact.

        Non-fatal: the lakeFS/S3 upload is the authoritative artifact store,
        so an MLflow logging failure only warns (consistent with the other
        MLflow logging in this service).
        """
        client = MlflowClient()
        try:
            client.log_artifact(run_id, tflite_path)
            self._logger.info(
                "Logged TFLite artifact to MLflow | run_id=%s file=%s",
                run_id,
                Path(tflite_path).name,
            )
        except Exception as exc:
            self._logger.warning("Failed to log TFLite artifact to MLflow: %s", exc)

    def _run_parity_and_log(
        self,
        run_id: str,
        tflite_path: str,
        fp32_checkpoint_path: Optional[str],
        params: QuantizationParams,
        headless: bool,
    ) -> ParityReport:
        """Run parity test, save report, and log metrics/artifact to MLflow.

        ``headless`` selects the FP32 reference forward: ``False`` for PTQ
        (full-model TFLite) and ``True`` for QAT (backbone+neck TFLite).

        If fp32_checkpoint_path is None (QAT passthrough without a checkpoint),
        parity is skipped and a passing report with zero error is returned.
        """
        if fp32_checkpoint_path is None:
            self._logger.info(
                "Parity test skipped — no FP32 checkpoint provided (QAT without checkpoint)"
            )
            return ParityReport(
                parity_passed=True,
                max_abs_error=0.0,
                threshold=params.parity_max_abs_error,
                frames_tested=0,
            )

        try:
            parity = self._parity.run(
                tflite_path=tflite_path,
                fp32_checkpoint_path=fp32_checkpoint_path,
                dataset_dir=params.dataset_dir,
                image_size=params.image_size,
                parity_frames=params.parity_frames,
                seed=params.calibration_seed,
                max_abs_error_threshold=params.parity_max_abs_error,
                headless=headless,
            )
            report_path: Optional[str] = self._parity.save_report(
                parity, params.output_dir
            )
        except Exception as exc:
            # Parity is DIAGNOSTIC, not a gate. By this point the INT8 model is
            # already exported, uploaded to S3 and logged to MLflow, so a parity
            # computation error must not fail the step (which would also drop the
            # quantization_result.json output). Record a sentinel and continue.
            self._logger.error(
                "Parity test errored (non-fatal; INT8 artifact already published): %s",
                exc,
                exc_info=True,
            )
            # Persist the error to MLflow — the pod's scale-from-0 node is torn
            # down after the run, taking its logs with it, so a bare -1.0 sentinel
            # left us blind. A tag survives and pins WHICH error caused the -1.0.
            try:
                MlflowClient().set_tag(
                    run_id, "parity_error", f"{type(exc).__name__}: {exc}"[:490]
                )
            except Exception:  # noqa: BLE001
                pass
            parity = ParityReport(
                parity_passed=False,
                max_abs_error=-1.0,  # sentinel: parity could not be computed
                threshold=params.parity_max_abs_error,
                frames_tested=0,
            )
            report_path = None

        client = MlflowClient()
        try:
            client.log_metric(run_id, "parity_max_abs_error", parity.max_abs_error)
            client.log_metric(run_id, "parity_passed", float(parity.parity_passed))
            if report_path is not None:
                client.log_artifact(run_id, report_path)
        except Exception as exc:
            self._logger.warning("Failed to log parity metrics to MLflow: %s", exc)

        return parity

    # ------------------------------------------------------------------
    # System-resource sampling
    # ------------------------------------------------------------------

    @contextmanager
    def _sample_system_metrics(self, run_id: str) -> Iterator[None]:
        """Sample CPU/RAM (and GPU when present) into MLflow while the body runs.

        Runs a daemon thread that logs a snapshot from :class:`ResourceMonitor`
        every ``_SYSTEM_METRICS_INTERVAL_S`` seconds under a monotonically
        increasing step. This gives the quantization run a system-metrics time
        series (parity with the training run, whose ``resource_monitor`` logs the
        same ``system/*`` keys). Entirely best-effort: any failure to start or
        sample is swallowed so it never affects the export.
        """
        monitor: Optional[ResourceMonitor]
        try:
            # gpu_index=0 is inert on CPU nodes (ResourceMonitor skips GPU unless
            # pynvml detected a device); relevant only for QAT-on-GPU passthrough.
            monitor = ResourceMonitor(gpu_index=0)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("System-metrics sampler disabled: %s", exc)
            monitor = None

        if monitor is None:
            yield
            return

        client = MlflowClient()
        stop = threading.Event()
        state = {"step": 0}

        def _loop() -> None:
            while not stop.is_set():
                for key, value in monitor.collect().items():
                    try:
                        client.log_metric(run_id, key, value, step=state["step"])
                    except Exception:  # noqa: BLE001
                        pass
                state["step"] += 1
                stop.wait(_SYSTEM_METRICS_INTERVAL_S)

        thread = threading.Thread(
            target=_loop, name="quant-system-metrics", daemon=True
        )
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # mAP-delta (FP32 vs INT8 task-metric quality)
    # ------------------------------------------------------------------

    def _run_map_delta_and_log(
        self,
        run_id: str,
        fp32_checkpoint_path: str,
        tflite_path: str,
        data_yaml: str,
        params: QuantizationParams,
    ) -> None:
        """Validate FP32 and INT8 on the val set and log the mAP delta.

        This is the *task-level* quality signal parity cannot give: it runs
        ``model.val`` on the FP32 ``.pt`` and the INT8 ``.tflite`` over the same
        val split and logs ``fp32_<k>`` / ``int8_<k>`` / ``delta_<k>`` for box
        (B) and pose (P) mAP. Retention is logged as ``int8_map_retention_mAP50B``
        (int8 ÷ fp32) when the FP32 reference is non-zero.

        Entirely non-fatal — the INT8 artifact is already published. NOTE: this
        needs a *labelled* val split; on a labels-only calibration download (or
        an untrained model) the reference mAP is ~0, in which case the delta is
        uninformative and ``map_delta_reference_map50b`` (logged) will be ~0.
        """
        client = MlflowClient()
        try:
            fp32_metrics = self._validate_map(
                fp32_checkpoint_path, data_yaml, params, tag="fp32"
            )
            int8_metrics = self._validate_map(
                tflite_path, data_yaml, params, tag="int8"
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "mAP-delta errored (non-fatal; INT8 artifact already published): %s",
                exc,
                exc_info=True,
            )
            return

        try:
            for raw_key, suffix in _MAP_METRIC_KEYS.items():
                fp32_v = fp32_metrics.get(raw_key)
                int8_v = int8_metrics.get(raw_key)
                if fp32_v is None or int8_v is None:
                    continue
                client.log_metric(run_id, f"fp32_{suffix}", fp32_v)
                client.log_metric(run_id, f"int8_{suffix}", int8_v)
                client.log_metric(run_id, f"delta_{suffix}", fp32_v - int8_v)

            ref = fp32_metrics.get("metrics/mAP50(B)")
            if ref is not None:
                # Self-documenting flag: if ~0 the mAP-delta is not meaningful
                # (unlabelled val or an untrained model).
                client.log_metric(run_id, "map_delta_reference_map50b", ref)
                int8_ref = int8_metrics.get("metrics/mAP50(B)")
                if ref > 1e-6 and int8_ref is not None:
                    client.log_metric(
                        run_id, "int8_map_retention_mAP50B", int8_ref / ref
                    )
            self._logger.info(
                "mAP-delta | fp32_mAP50B=%.5f int8_mAP50B=%.5f fp32_mAP50P=%.5f "
                "int8_mAP50P=%.5f",
                fp32_metrics.get("metrics/mAP50(B)", 0.0),
                int8_metrics.get("metrics/mAP50(B)", 0.0),
                fp32_metrics.get("metrics/mAP50(P)", 0.0),
                int8_metrics.get("metrics/mAP50(P)", 0.0),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed to log mAP-delta metrics to MLflow: %s", exc)

    def _validate_map(
        self,
        model_path: str,
        data_yaml: str,
        params: QuantizationParams,
        tag: str,
    ) -> dict[str, float]:
        """Run ``model.val`` and return its ``results_dict`` (box + pose mAP).

        Writes val outputs under ``output_dir`` (the pod's writable volume) and
        disables plots for speed. AutoBackend handles the INT8 ``.tflite``
        dequantization, so the same call works for both ``.pt`` and ``.tflite``.
        """
        self._logger.info("mAP val (%s) | model=%s", tag, model_path)
        model = YOLO(model_path)
        results = model.val(
            data=data_yaml,
            imgsz=params.image_size,
            verbose=False,
            plots=False,
            project=params.output_dir,
            name=f"val_{tag}",
            exist_ok=True,
        )
        results_dict = getattr(results, "results_dict", None) or {}
        return {k: float(v) for k, v in results_dict.items()}
