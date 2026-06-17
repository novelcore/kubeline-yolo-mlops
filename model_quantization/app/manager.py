"""Manager for the model-quantization pipeline step."""

import logging
import os
from typing import Optional

import boto3
from botocore.config import Config as BotoConfig

from app.logger import setup_logging
from app.models.config import Config
from app.models.quantization import QuantizationParams, QuantizationResult
from app.services.quantization_service import QuantizationService


class Manager:
    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or Config()
        setup_logging(level=self._config.log_level)
        self._logger = logging.getLogger(__name__)

        if self._config.mlflow_tracking_username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self._config.mlflow_tracking_username
        if self._config.mlflow_tracking_password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self._config.mlflow_tracking_password

        self._s3_client = self._build_s3_client()
        self._service = QuantizationService(
            s3_client=self._s3_client,
            mlflow_tracking_uri=self._config.mlflow_tracking_uri,
        )

    def _build_s3_client(self) -> object:
        boto_cfg = BotoConfig(retries={"max_attempts": 3, "mode": "adaptive"})
        cfg = self._config
        if cfg.lakefs_endpoint and cfg.lakefs_access_key and cfg.lakefs_secret_key:
            return boto3.client(
                "s3",
                endpoint_url=cfg.lakefs_endpoint,
                aws_access_key_id=cfg.lakefs_access_key,
                aws_secret_access_key=cfg.lakefs_secret_key,
                region_name=None,
                config=boto_cfg,
            )
        return boto3.client(
            "s3",
            endpoint_url=cfg.s3_endpoint_url,
            aws_access_key_id=cfg.aws_access_key_id,
            aws_secret_access_key=cfg.aws_secret_access_key,
            region_name=cfg.aws_default_region,
            config=boto_cfg,
        )

    def run(
        self,
        mode: str,
        source_mlflow_run_id: str,
        dataset_dir: str,
        output_dir: str,
        output_bucket: str,
        output_prefix: str,
        experiment_name: str,
        fp32_checkpoint_path: Optional[str] = None,
        tflite_s3_uri: Optional[str] = None,
        qat_run_id: Optional[str] = None,
        image_size: int = 640,
        calibration_frames: int = 512,
        calibration_seed: int = 42,
        parity_frames: int = 100,
        parity_max_abs_error: float = 0.05,
    ) -> QuantizationResult:
        self._logger.info(
            "Starting %s | mode=%s source_run=%s",
            self._config.app_name,
            mode,
            source_mlflow_run_id,
        )
        params = QuantizationParams(
            mode=mode,  # type: ignore[arg-type]
            fp32_checkpoint_path=fp32_checkpoint_path,
            tflite_s3_uri=tflite_s3_uri,
            qat_run_id=qat_run_id,
            source_mlflow_run_id=source_mlflow_run_id,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            output_bucket=output_bucket,
            output_prefix=output_prefix,
            experiment_name=experiment_name,
            image_size=image_size,
            calibration_frames=calibration_frames,
            calibration_seed=calibration_seed,
            parity_frames=parity_frames,
            parity_max_abs_error=parity_max_abs_error,
        )
        return self._service.run(params=params)
