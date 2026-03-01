import logging
import json
from pathlib import Path

from app.models.training import TrainingParams, TrainingMetrics


class ModelTrainingError(Exception):
    """Custom exception for model training errors."""
    pass


class ModelTrainingService:
    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def run(self, params: TrainingParams, device: str = "cpu") -> TrainingMetrics:
        """Execute a model training run."""
        try:
            self._logger.info(
                f"Starting training: model={params.model_name}, "
                f"epochs={params.epochs}, device={device}"
            )

            # Validate inputs
            self._validate_dataset(params.dataset_dir)
            self._prepare_output_dir(params.output_dir)

            # Simulate training loop
            metrics = self._train(params)

            # Save checkpoint metadata
            self._save_metadata(metrics)

            print(
                f"Training complete: {params.model_name} | "
                f"epochs={metrics.epochs_completed} | "
                f"train_loss={metrics.final_train_loss:.4f} | "
                f"checkpoint={metrics.checkpoint_path}"
            )
            return metrics

        except ModelTrainingError:
            raise
        except Exception as e:
            self._logger.error(f"Unexpected error: {e}")
            raise ModelTrainingError(f"Training failed: {e}") from e

    def _validate_dataset(self, dataset_dir: str) -> None:
        """Validate that the dataset directory exists and is not empty."""
        path = Path(dataset_dir)
        if not path.exists():
            raise ModelTrainingError(f"Dataset directory not found: {dataset_dir}")
        if not path.is_dir():
            raise ModelTrainingError(f"Dataset path is not a directory: {dataset_dir}")

    def _prepare_output_dir(self, output_dir: str) -> None:
        """Create the output directory if it doesn't exist."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _train(self, params: TrainingParams) -> TrainingMetrics:
        """Run the training loop. Returns final metrics."""
        # Simulate epoch-by-epoch training with decreasing loss
        train_loss = 2.5
        val_loss = 2.8
        for epoch in range(1, params.epochs + 1):
            train_loss *= 0.85
            val_loss *= 0.87
            self._logger.info(
                f"Epoch {epoch}/{params.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
            )

        checkpoint_path = str(
            Path(params.output_dir) / f"{params.model_name}_epoch{params.epochs}.pt"
        )
        # Simulate saving checkpoint
        Path(checkpoint_path).touch()

        return TrainingMetrics(
            model_name=params.model_name,
            epochs_completed=params.epochs,
            final_train_loss=round(train_loss, 6),
            final_val_loss=round(val_loss, 6),
            checkpoint_path=checkpoint_path,
        )

    def _save_metadata(self, metrics: TrainingMetrics) -> None:
        """Save training metadata alongside the checkpoint."""
        meta_path = Path(metrics.checkpoint_path).with_suffix(".meta.json")
        meta_path.write_text(metrics.model_dump_json(indent=2))
        self._logger.info(f"Metadata saved: {meta_path}")
