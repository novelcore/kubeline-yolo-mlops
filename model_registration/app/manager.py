"""Manager for the Model Registration pipeline step."""

import logging
from typing import Optional

from app.models.config import Config
from app.models.registration import RegistrationParams
from app.services.model_registration import ModelRegistrationService


class Manager:
    def __init__(self, config: Config = None) -> None:
        self._config = config or Config()
        self._service = ModelRegistrationService()

        logging.basicConfig(
            level=getattr(logging, self._config.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self._logger = logging.getLogger(__name__)

    def run(
        self,
        model_name: str,
        checkpoint_path: str,
        registry_url: Optional[str] = None,
        version: Optional[str] = None,
        tags: Optional[list[str]] = None,
        promote_to: Optional[str] = None,
    ) -> None:
        """Run the model registration step."""
        self._logger.info(f"Starting application: {self._config.app_name}")

        try:
            params = RegistrationParams(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                registry_url=registry_url or self._config.registry_url,
                version=version,
                tags=tags or [],
                promote_to=promote_to,
            )
            self._service.run(params=params, token=self._config.registry_token)
            self._logger.info("Model registration completed successfully")
        except Exception as e:
            self._logger.error(f"Model registration failed: {e}")
            raise
