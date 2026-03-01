import logging
from pathlib import Path

from app.models.dataset import DatasetParams, DatasetSplitResult


class DatasetLoadingError(Exception):
    """Custom exception for dataset loading errors."""
    pass


class DatasetLoadingService:
    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def run(self, params: DatasetParams, shuffle_seed: int = 42) -> DatasetSplitResult:
        """Load a dataset, validate it, and split into train/val/test."""
        try:
            self._logger.info(f"Loading dataset from: {params.source_path}")

            # Validate source
            self._validate_source(params.source_path, params.format)

            # Validate output directory
            output_path = Path(params.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Simulate loading and splitting
            total_records = self._load_records(params.source_path, params.format)
            train_n = int(total_records * params.train_split)
            val_n = int(total_records * params.val_split)
            test_n = total_records - train_n - val_n

            result = DatasetSplitResult(
                total_records=total_records,
                train_records=train_n,
                val_records=val_n,
                test_records=test_n,
                output_dir=params.output_dir,
            )

            self._logger.info(
                f"Dataset split: train={train_n}, val={val_n}, test={test_n}"
            )
            print(
                f"Dataset loaded: {total_records} records from {params.source_path} "
                f"-> train={train_n}, val={val_n}, test={test_n}"
            )
            return result

        except DatasetLoadingError:
            raise
        except Exception as e:
            self._logger.error(f"Unexpected error: {e}")
            raise DatasetLoadingError(f"Failed to load dataset: {e}") from e

    def _validate_source(self, source_path: str, fmt: str) -> None:
        """Validate the dataset source exists and matches the expected format."""
        path = Path(source_path)
        if not path.exists():
            raise DatasetLoadingError(f"Source path not found: {source_path}")

        expected_extensions = {
            "csv": {".csv"},
            "parquet": {".parquet", ".pq"},
            "jsonl": {".jsonl", ".ndjson"},
            "hf": set(),  # HuggingFace datasets don't have file extensions
        }

        valid_ext = expected_extensions.get(fmt, set())
        if valid_ext and path.is_file() and path.suffix not in valid_ext:
            raise DatasetLoadingError(
                f"File extension '{path.suffix}' does not match format '{fmt}'"
            )

    def _load_records(self, source_path: str, fmt: str) -> int:
        """Load records from the source. Returns total record count."""
        path = Path(source_path)
        if path.is_file():
            # Count lines as a proxy for records
            with open(path) as f:
                count = sum(1 for _ in f)
            # Subtract header line for CSV
            if fmt == "csv" and count > 0:
                count -= 1
            return max(count, 0)
        elif path.is_dir():
            # Count files in directory as proxy
            return sum(1 for _ in path.iterdir() if _.is_file())
        return 0
