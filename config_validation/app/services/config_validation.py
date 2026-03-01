import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from app.models.pipeline_config import PipelineConfig


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


class ConfigValidationService:
    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def run(self, config_path: str, strict: bool = True) -> PipelineConfig:
        """Load and validate a pipeline configuration file."""
        try:
            self._logger.info(f"Validating config: {config_path}")

            raw = self._load_yaml(config_path)
            config = self._validate_schema(raw)
            self._validate_splits(config, strict)

            self._logger.info("Pipeline configuration is valid")
            print(f"Config validation passed: {config.pipeline_name} v{config.version}")
            return config

        except ConfigValidationError:
            raise
        except Exception as e:
            self._logger.error(f"Unexpected error: {e}")
            raise ConfigValidationError(f"Failed to validate config: {e}") from e

    def _load_yaml(self, config_path: str) -> dict[str, Any]:
        """Load a YAML file from disk."""
        path = Path(config_path)
        if not path.exists():
            raise ConfigValidationError(f"Config file not found: {config_path}")
        if not path.suffix in (".yaml", ".yml"):
            raise ConfigValidationError(f"Config file must be YAML: {config_path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ConfigValidationError("Config file must contain a YAML mapping")
        return data

    def _validate_schema(self, raw: dict[str, Any]) -> PipelineConfig:
        """Validate raw dict against the PipelineConfig schema."""
        try:
            return PipelineConfig(**raw)
        except ValidationError as e:
            raise ConfigValidationError(f"Schema validation failed:\n{e}") from e

    def _validate_splits(self, config: PipelineConfig, strict: bool) -> None:
        """Validate that dataset splits sum to 1.0."""
        ds = config.dataset
        total = ds.train_split + ds.val_split + ds.test_split
        if strict and abs(total - 1.0) > 1e-6:
            raise ConfigValidationError(
                f"Dataset splits must sum to 1.0, got {total:.4f}"
            )
