"""Manager for the Model Registration pipeline step."""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from app.logger import setup_logging
from app.models.config import Config
from app.models.registration import RegistrationParams, RegistrationResult
from app.services.model_registration import ModelRegistrationService


class Manager:
    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or Config()

        setup_logging(level=self._config.log_level)
        self._logger = logging.getLogger(__name__)

        # Export MLflow auth credentials so the MLflow client can pick them up
        if self._config.mlflow_tracking_username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self._config.mlflow_tracking_username
        if self._config.mlflow_tracking_password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self._config.mlflow_tracking_password

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

        try:
            result = self._service.run(params=params)
        finally:
            self._cleanup(
                best_checkpoint_path=best_checkpoint_path,
                last_checkpoint_path=last_checkpoint_path,
            )

        self._logger.info(
            "Registration complete | model=%s best_version=%s last_version=%s promoted_to=%s",
            result.registered_model_name,
            result.best_version,
            result.last_version,
            result.promoted_to,
        )

        return result

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(
        self,
        best_checkpoint_path: str,
        last_checkpoint_path: Optional[str],
    ) -> None:
        """Delete local checkpoint files and release GPU memory.

        This method is called from a ``finally`` block so it always runs,
        regardless of whether the registration step succeeded or failed.

        S3 URIs (``s3://...``) are never touched — only paths that resolve
        to files or directories on the local filesystem are deleted.
        """
        local_paths: list[str] = [best_checkpoint_path]
        if last_checkpoint_path is not None:
            local_paths.append(last_checkpoint_path)

        for raw_path in local_paths:
            if raw_path.startswith("s3://"):
                self._logger.debug("Skipping cleanup of S3 URI: %s", raw_path)
                continue

            path = Path(raw_path)
            if not path.exists():
                self._logger.debug(
                    "Cleanup: path does not exist, nothing to remove: %s", path
                )
                continue

            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    self._logger.info("Cleanup: removed directory %s", path)
                else:
                    path.unlink()
                    self._logger.info("Cleanup: removed file %s", path)
            except OSError as exc:
                # Log but do not re-raise — cleanup failures must never mask
                # the original registration error or prevent a clean exit.
                self._logger.error("Cleanup: failed to remove %s: %s", path, exc)

        self._free_gpu_memory()

    def _free_gpu_memory(self) -> None:
        """Empty the CUDA cache if PyTorch and a GPU are available."""
        try:
            import torch  # type: ignore[import-untyped,import-not-found]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self._logger.debug("GPU cache cleared via torch.cuda.empty_cache()")
            else:
                self._logger.debug(
                    "torch imported but CUDA is not available; skipping cache clear"
                )
        except ImportError:
            self._logger.debug("torch not installed; skipping GPU cache clear")
