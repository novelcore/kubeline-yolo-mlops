"""Manager for the Dataset Loading pipeline step."""

import logging

from app.models.config import Config
from app.models.dataset import DatasetParams
from app.services.dataset_loading import DatasetLoadingService


class Manager:
    def __init__(self, config: Config = None) -> None:
        self._config = config or Config()
        self._service = DatasetLoadingService()

        logging.basicConfig(
            level=getattr(logging, self._config.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self._logger = logging.getLogger(__name__)

    def run(
        self,
        source_path: str,
        output_dir: str,
        format: str,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ) -> None:
        """Run the dataset loading step."""
        self._logger.info(f"Starting application: {self._config.app_name}")

        try:
            params = DatasetParams(
                source_path=source_path,
                output_dir=output_dir,
                format=format,
                train_split=train_split,
                val_split=val_split,
                test_split=test_split,
            )
            self._service.run(params=params, shuffle_seed=self._config.shuffle_seed)
            self._logger.info("Dataset loading completed successfully")
        except Exception as e:
            self._logger.error(f"Dataset loading failed: {e}")
            raise
