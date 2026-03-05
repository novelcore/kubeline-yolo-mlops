"""Manager for the Config Validation pipeline step."""

import logging
from typing import Optional

import boto3
from botocore.config import Config as BotoConfig

from app.logger import setup_logging
from app.models.config import Config
from app.services.config_validation import ConfigValidationService


class Manager:
    def __init__(self, config: Config = None) -> None:
        self._config = config or Config()
        setup_logging(level=self._config.log_level)
        self._logger = logging.getLogger(__name__)

        self._s3_client = self._build_s3_client()
        self._service = ConfigValidationService(
            skip_liveness_checks=self._config.skip_liveness_checks,
            max_retries=self._config.max_retries,
            timeout=self._config.timeout,
            mlflow_tracking_uri=self._config.mlflow_tracking_uri,
            s3_client=self._s3_client,
        )

    # ------------------------------------------------------------------
    # S3 / LakeFS client factory
    # ------------------------------------------------------------------

    def _build_s3_client(self) -> object:
        """Construct a boto3 S3 client, preferring LakeFS when configured."""
        boto_cfg = BotoConfig(
            retries={"max_attempts": self._config.max_retries, "mode": "adaptive"},
            connect_timeout=self._config.timeout,
            read_timeout=self._config.timeout,
        )

        if self._config.lakefs_endpoint:
            self._logger.debug(
                "Configuring boto3 for LakeFS at %s", self._config.lakefs_endpoint
            )
            return boto3.client(
                "s3",
                endpoint_url=self._config.lakefs_endpoint,
                aws_access_key_id=self._config.lakefs_access_key,
                aws_secret_access_key=self._config.lakefs_secret_key,
                config=boto_cfg,
            )

        return boto3.client(
            "s3",
            endpoint_url=self._config.s3_endpoint_url,
            aws_access_key_id=self._config.aws_access_key_id,
            aws_secret_access_key=self._config.aws_secret_access_key,
            region_name=self._config.aws_default_region,
            config=boto_cfg,
        )

    def run(self, config_dict: dict, output_path: Optional[str] = None) -> None:
        """Run the config validation step."""
        self._logger.info(f"Starting application: {self._config.app_name}")

        try:
            self._service.run(config_dict=config_dict, output_path=output_path)
            self._logger.info("Config validation completed successfully")
        except Exception as e:
            self._logger.error(f"Config validation failed: {e}")
            raise
