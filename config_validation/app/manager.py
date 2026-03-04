"""Manager for the Config Validation pipeline step."""

import logging
from typing import Optional

from app.logger import setup_logging
from app.models.config import Config
from app.services.config_validation import ConfigValidationService


class Manager:
    def __init__(self, config: Config = None) -> None:
        self._config = config or Config()
        setup_logging(level=self._config.log_level)

        self._service = ConfigValidationService(
            skip_liveness_checks=self._config.skip_liveness_checks,
            max_retries=self._config.max_retries,
            timeout=self._config.timeout,
            mlflow_tracking_uri=self._config.mlflow_tracking_uri,
        )
        self._logger = logging.getLogger(__name__)

    def run(self, config_dict: dict, output_path: Optional[str] = None) -> None:
        """Run the config validation step."""
        self._logger.info(f"Starting application: {self._config.app_name}")

        try:
            self._service.run(config_dict=config_dict, output_path=output_path)
            self._logger.info("Config validation completed successfully")
        except Exception as e:
            self._logger.error(f"Config validation failed: {e}")
            raise
