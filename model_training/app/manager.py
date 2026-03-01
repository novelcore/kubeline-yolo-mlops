"""Manager for the Model Training pipeline step."""

import logging

from app.models.config import Config
from app.models.training import TrainingParams
from app.services.model_training import ModelTrainingService


class Manager:
    def __init__(self, config: Config = None) -> None:
        self._config = config or Config()
        self._service = ModelTrainingService()

        logging.basicConfig(
            level=getattr(logging, self._config.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self._logger = logging.getLogger(__name__)

    def run(
        self,
        model_name: str,
        dataset_dir: str,
        output_dir: str,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        optimizer: str = "adamw",
    ) -> None:
        """Run the model training step."""
        self._logger.info(f"Starting application: {self._config.app_name}")

        try:
            params = TrainingParams(
                model_name=model_name,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                optimizer=optimizer,
                seed=self._config.default_seed,
            )
            self._service.run(params=params, device=self._config.device)
            self._logger.info("Model training completed successfully")
        except Exception as e:
            self._logger.error(f"Model training failed: {e}")
            raise
