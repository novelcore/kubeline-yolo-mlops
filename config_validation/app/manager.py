"""Manager for the Config Validation pipeline step."""

import logging

from app.models.config import Config
from app.services.config_validation import ConfigValidationService


class Manager:
    def __init__(self, config: Config = None) -> None:
        self._config = config or Config()
        self._service = ConfigValidationService()

        logging.basicConfig(
            level=getattr(logging, self._config.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self._logger = logging.getLogger(__name__)

    def run(self, config_path: str) -> None:
        """Run the config validation step."""
        self._logger.info(f"Starting application: {self._config.app_name}")

        try:
            self._service.run(
                config_path=config_path,
                strict=self._config.strict_mode,
            )
            self._logger.info("Config validation completed successfully")
        except Exception as e:
            self._logger.error(f"Config validation failed: {e}")
            raise
