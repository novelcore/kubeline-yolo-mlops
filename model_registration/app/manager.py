"""Manager for the Model Registration pipeline step."""

import logging
from typing import Optional

from app.models.config import Config
from app.models.registration import RegistrationParams, RegistrationResult
from app.services.model_registration import ModelRegistrationService


class Manager:
    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or Config()

        logging.basicConfig(
            level=getattr(logging, self._config.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self._logger = logging.getLogger(__name__)

        self._service = ModelRegistrationService(
            mlflow_tracking_uri=self._config.mlflow_tracking_uri,
            max_retries=self._config.max_retries,
        )

    def run(
        self,
        mlflow_run_id: str,
        best_checkpoint_path: str,
        last_checkpoint_path: Optional[str] = None,
        registered_model_name: Optional[str] = None,
        promote_to: Optional[str] = None,
        dataset_version: Optional[str] = None,
        dataset_sample_size: Optional[int] = None,
        config_hash: Optional[str] = None,
        git_commit: Optional[str] = None,
        model_variant: Optional[str] = None,
        best_map50: Optional[float] = None,
    ) -> RegistrationResult:
        """Execute the model registration step."""
        self._logger.info(
            "Starting %s | run_id=%s model=%s",
            self._config.app_name,
            mlflow_run_id,
            registered_model_name or self._config.registered_model_name,
        )

        params = RegistrationParams(
            mlflow_run_id=mlflow_run_id,
            best_checkpoint_path=best_checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
            registered_model_name=registered_model_name
            or self._config.registered_model_name,
            promote_to=promote_to,
            dataset_version=dataset_version,
            dataset_sample_size=dataset_sample_size,
            config_hash=config_hash,
            git_commit=git_commit,
            model_variant=model_variant,
            best_map50=best_map50,
        )

        result = self._service.run(params=params)

        self._logger.info(
            "Registration complete | model=%s best_version=%s last_version=%s promoted_to=%s",
            result.registered_model_name,
            result.best_version,
            result.last_version,
            result.promoted_to,
        )

        return result
