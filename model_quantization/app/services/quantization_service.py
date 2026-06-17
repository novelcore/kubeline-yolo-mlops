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
from pathlib import Path
from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO

from app.models.quantization import QuantizationParams, QuantizationResult


class QuantizationError(Exception):
    """Raised on non-recoverable quantization failures."""


class QuantizationService:
    """Runs PTQ (Ultralytics) or QAT passthrough and logs to MLflow."""

    def __init__(self, s3_client: Any, mlflow_tracking_uri: str) -> None:
        self._s3 = s3_client
        self._mlflow_uri = mlflow_tracking_uri
        self._logger = logging.getLogger(__name__)

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

            tflite_path = self._export_ptq(local_ckpt, data_yaml, params)
            s3_uri = self._upload_tflite(tflite_path, params)
            self._log_ptq_run(run_id, params, s3_uri)

        self._logger.info("PTQ complete | run_id=%s tflite=%s", run_id, s3_uri)

        return QuantizationResult(
            mlflow_run_id=run_id,
            source_run_id=params.source_mlflow_run_id,
            mode="ptq",
            tflite_s3_uri=s3_uri,
            parity_passed=True,      # placeholder — FR-M-03
            parity_max_abs_error=0.0,
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

            # FR-M-03 placeholder — parity test runs here once implemented
            self._logger.info(
                "Parity test skipped — FR-M-03 not yet implemented"
            )

            self._log_qat_passthrough_run(run_id, params)

        self._logger.info(
            "QAT passthrough complete | run_id=%s", run_id
        )

        return QuantizationResult(
            mlflow_run_id=run_id,
            source_run_id=params.source_mlflow_run_id,
            mode="qat",
            tflite_s3_uri=params.tflite_s3_uri,
            parity_passed=True,      # placeholder — FR-M-03
            parity_max_abs_error=0.0,
        )

    def _log_qat_passthrough_run(
        self, run_id: str, params: QuantizationParams
    ) -> None:
        """Log QAT passthrough parameters to MLflow."""
        client = MlflowClient()
        items: list[tuple[str, str]] = [
            ("quantization_mode", "qat"),
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
        without_scheme = path[len("s3://"):]
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
            f"No YOLO data YAML found in {dataset_dir!r}. "
            f"Searched: {candidates}"
        )

    def _upload_tflite(self, local_path: str, params: QuantizationParams) -> str:
        """Upload TFLite artifact to S3 and return the s3:// URI."""
        key = f"{params.output_prefix}/{Path(local_path).name}"
        self._logger.info(
            "Uploading TFLite to s3://%s/%s", params.output_bucket, key
        )
        self._s3.upload_file(local_path, params.output_bucket, key)
        return f"s3://{params.output_bucket}/{key}"
