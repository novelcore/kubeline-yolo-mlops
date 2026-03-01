import json
import logging
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import boto3
import botocore.exceptions
import httpx
from pydantic import ValidationError

from app.models.pipeline_config import PipelineConfig


class ConfigValidationError(Exception):
    """Raised when pipeline configuration validation fails."""
    pass


class ConfigValidationService:
    def __init__(
        self,
        skip_liveness_checks: bool,
        max_retries: int,
        timeout: int,
        mlflow_tracking_uri: Optional[str],
    ) -> None:
        self._skip_liveness_checks = skip_liveness_checks
        self._max_retries = max_retries
        self._timeout = timeout
        self._mlflow_tracking_uri = mlflow_tracking_uri
        self._logger = logging.getLogger(__name__)

    def run(self, config_dict: dict, output_path: Optional[str] = None) -> PipelineConfig:
        """Validate and optionally write the pipeline configuration."""
        try:
            self._logger.info("Validating pipeline configuration")

            config = self._validate_schema(config_dict)

            if not self._skip_liveness_checks:
                self._logger.info("Running liveness checks")
                self._check_dataset_path(config)
                self._check_pretrained_weights(config)
                self._check_mlflow_uri()
                self._check_checkpoint_resume(config)
            else:
                self._logger.info("Liveness checks skipped (SKIP_LIVENESS_CHECKS=true)")

            if output_path is not None:
                self._write_output(config, output_path)

            self._logger.info(
                f"Config validation passed: {config.experiment.name}"
            )
            print(f"Config validation passed: {config.experiment.name}")
            return config

        except ConfigValidationError:
            raise
        except Exception as e:
            self._logger.error(f"Unexpected error during config validation: {e}")
            raise ConfigValidationError(f"Failed to validate config: {e}") from e

    def _validate_schema(self, raw: dict[str, Any]) -> PipelineConfig:
        """Validate raw dict against the PipelineConfig schema."""
        try:
            return PipelineConfig(**raw)
        except ValidationError as e:
            raise ConfigValidationError(f"Schema validation failed:\n{e}") from e

    def _check_dataset_path(self, config: PipelineConfig) -> None:
        """Verify the dataset path exists in S3/LakeFS."""
        if config.dataset.path_override is not None:
            s3_path = config.dataset.path_override
        else:
            # Convention: datasets live at s3://{bucket}/datasets/speedplus_yolo/{version}/
            # The storage_path is s3://{bucket}/checkpoints; strip the /checkpoints suffix
            # to get the bucket base URL.
            base = config.checkpointing.storage_path.rstrip("/")
            if base.endswith("/checkpoints"):
                base = base[: -len("/checkpoints")]
            s3_path = f"{base}/datasets/speedplus_yolo/{config.dataset.version}/"

        self._logger.info(f"Checking dataset path: {s3_path}")
        parsed = urlparse(s3_path)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")

        s3 = boto3.client("s3")
        for attempt in range(1, self._max_retries + 1):
            try:
                response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
                if response.get("KeyCount", 0) == 0:
                    raise ConfigValidationError(
                        f"Dataset path not found or empty in S3: {s3_path}"
                    )
                self._logger.info(f"Dataset path confirmed: {s3_path}")
                return
            except ConfigValidationError:
                raise
            except botocore.exceptions.ClientError as e:
                msg = f"S3 error checking dataset path {s3_path}: {e}"
                if attempt < self._max_retries:
                    self._logger.warning(f"{msg} — retrying ({attempt}/{self._max_retries})")
                    time.sleep(1)
                else:
                    raise ConfigValidationError(msg) from e
            except Exception as e:
                raise ConfigValidationError(
                    f"Unexpected error checking dataset path {s3_path}: {e}"
                ) from e

    def _check_pretrained_weights(self, config: PipelineConfig) -> None:
        """Verify the pretrained weights file exists in S3 (if a custom path is set)."""
        weights = config.model.pretrained_weights
        if weights is None:
            return  # standard Ultralytics model names are always reachable from the training container

        if not weights.startswith("s3://"):
            raise ConfigValidationError(
                f"model.pretrained_weights must be null or an S3 path starting with 's3://', "
                f"got: {weights!r}"
            )

        self._logger.info(f"Checking pretrained weights: {weights}")
        parsed = urlparse(weights)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        s3 = boto3.client("s3")
        for attempt in range(1, self._max_retries + 1):
            try:
                s3.head_object(Bucket=bucket, Key=key)
                self._logger.info(f"Pretrained weights confirmed: {weights}")
                return
            except botocore.exceptions.ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "404":
                    raise ConfigValidationError(
                        f"Pretrained weights file not found: {weights}"
                    ) from e
                msg = f"S3 error checking weights {weights}: {e}"
                if attempt < self._max_retries:
                    self._logger.warning(f"{msg} — retrying ({attempt}/{self._max_retries})")
                    time.sleep(1)
                else:
                    raise ConfigValidationError(msg) from e
            except Exception as e:
                raise ConfigValidationError(
                    f"Unexpected error checking weights {weights}: {e}"
                ) from e

    def _check_mlflow_uri(self) -> None:
        """Verify the MLflow tracking server is reachable via its /health endpoint."""
        if not self._mlflow_tracking_uri:
            raise ConfigValidationError(
                "MLFLOW_TRACKING_URI is not set. It is required when liveness checks are enabled."
            )

        health_url = self._mlflow_tracking_uri.rstrip("/") + "/health"
        self._logger.info(f"Checking MLflow URI: {health_url}")

        for attempt in range(1, self._max_retries + 1):
            try:
                response = httpx.get(health_url, timeout=self._timeout)
                if response.is_success:
                    self._logger.info(f"MLflow reachable at: {self._mlflow_tracking_uri}")
                    return
                raise ConfigValidationError(
                    f"MLflow health check failed at {health_url} "
                    f"with status {response.status_code}"
                )
            except ConfigValidationError:
                raise
            except httpx.RequestError as e:
                msg = f"MLflow unreachable at {health_url}: {type(e).__name__}"
                if attempt < self._max_retries:
                    self._logger.warning(f"{msg} — retrying ({attempt}/{self._max_retries})")
                    time.sleep(1)
                else:
                    raise ConfigValidationError(msg) from e

    def _check_checkpoint_resume(self, config: PipelineConfig) -> None:
        """Verify checkpoint exists when resume_from is set."""
        resume_from = config.checkpointing.resume_from
        if resume_from is None:
            return

        s3 = boto3.client("s3")

        if resume_from == "auto":
            # Scan the experiment's checkpoint directory for any .pt file
            checkpoint_dir = (
                f"{config.checkpointing.storage_path.rstrip('/')}/"
                f"{config.experiment.name}/"
            )
            self._logger.info(
                f"resume_from='auto': scanning for checkpoints at {checkpoint_dir}"
            )
            parsed = urlparse(checkpoint_dir)
            bucket = parsed.netloc
            prefix = parsed.path.lstrip("/")

            try:
                response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=100)
                contents = response.get("Contents", [])
                pt_files = [obj for obj in contents if obj["Key"].endswith(".pt")]
                if not pt_files:
                    raise ConfigValidationError(
                        f"checkpointing.resume_from is 'auto' but no .pt checkpoint found at "
                        f"{checkpoint_dir}. Start a new run or set resume_from: null."
                    )
                latest = max(pt_files, key=lambda obj: obj["LastModified"])
                self._logger.info(
                    f"Found checkpoint for auto-resume: s3://{bucket}/{latest['Key']}"
                )
            except ConfigValidationError:
                raise
            except Exception as e:
                raise ConfigValidationError(
                    f"Error scanning checkpoint directory {checkpoint_dir}: {e}"
                ) from e

        else:
            # Specific S3 path — verify the object exists
            self._logger.info(f"Checking checkpoint file: {resume_from}")
            parsed = urlparse(resume_from)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")

            try:
                s3.head_object(Bucket=bucket, Key=key)
                self._logger.info(f"Checkpoint file confirmed: {resume_from}")
            except botocore.exceptions.ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "404":
                    raise ConfigValidationError(
                        f"Checkpoint file not found: {resume_from}"
                    ) from e
                raise ConfigValidationError(
                    f"S3 error checking checkpoint {resume_from}: {e}"
                ) from e
            except Exception as e:
                raise ConfigValidationError(
                    f"Unexpected error checking checkpoint {resume_from}: {e}"
                ) from e

    def _write_output(self, config: PipelineConfig, output_path: str) -> None:
        """Write the validated config as a JSON artifact."""
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(config.model_dump_json(indent=2))
            self._logger.info(f"Validated config written to: {output_path}")
        except OSError as e:
            raise ConfigValidationError(
                f"Failed to write validated config to {output_path}: {e}"
            ) from e