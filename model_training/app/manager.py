"""Manager for the Model Training pipeline step.

Wires together Config, the boto3 S3 client, and TrainingService.
The ``run`` method is the single orchestration entry point called by the CLI.
"""

import logging
import os
from typing import Optional

import boto3
from botocore.config import Config as BotoConfig

from app.logger import setup_logging
from app.models.config import Config
from app.models.training import AugmentationParams, TrainingParams, TrainingResult
from app.services.model_training import TrainingService


class Manager:
    """Orchestrates the model training step.

    Parameters
    ----------
    config:
        Optional pre-built ``Config`` instance. When *None* a default
        ``Config()`` is constructed (which reads from env vars / .env).
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or Config()

        setup_logging(level=self._config.log_level)
        self._logger = logging.getLogger(__name__)

        # Export MLflow auth credentials so the MLflow client can pick them up
        if self._config.mlflow_tracking_username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self._config.mlflow_tracking_username
        if self._config.mlflow_tracking_password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self._config.mlflow_tracking_password

        self._s3_client = self._build_s3_client()
        self._service = TrainingService(
            s3_client=self._s3_client,
            mlflow_tracking_uri=self._config.mlflow_tracking_uri,
        )

    # ------------------------------------------------------------------
    # S3 client factory
    # ------------------------------------------------------------------

    def _build_s3_client(self) -> object:
        """Construct a boto3 S3 client from the step config.

        LakeFS credentials take precedence when the LakeFS endpoint is
        configured, mirroring the pattern established in dataset_loading.
        """
        boto_cfg = BotoConfig(
            retries={"max_attempts": 3, "mode": "adaptive"},
        )

        if self._config.lakefs_endpoint:
            self._logger.debug(
                "Configuring boto3 for LakeFS at %s", self._config.lakefs_endpoint
            )
            return boto3.client(
                "s3",
                endpoint_url=self._config.lakefs_endpoint,
                aws_access_key_id=self._config.lakefs_access_key,
                aws_secret_access_key=self._config.lakefs_secret_key,
                config=boto_cfg,
            )

        return boto3.client(
            "s3",
            endpoint_url=self._config.s3_endpoint_url,
            aws_access_key_id=self._config.aws_access_key_id,
            aws_secret_access_key=self._config.aws_secret_access_key,
            region_name=self._config.aws_default_region,
            config=boto_cfg,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        # Identity
        model_variant: str,
        experiment_name: str,
        dataset_dir: str,
        output_dir: str,
        # Dataset source
        source: str = "local",
        s3_bucket: Optional[str] = None,
        s3_prefix: Optional[str] = None,
        disk_cache_bytes: int = 2 * 1024**3,
        # Weight init / resume
        pretrained_weights: Optional[str] = None,
        resume_from: Optional[str] = None,
        # Device
        device: Optional[str] = None,
        # Core schedule (keyword-only from here; all callers use explicit kwargs)
        *,
        epochs: int,
        batch_size: int,
        image_size: int,
        # LR
        learning_rate: float,
        cos_lr: bool,
        lrf: float,
        # Optimizer
        optimizer: str,
        momentum: float,
        weight_decay: float,
        # Warmup
        warmup_epochs: float,
        warmup_momentum: float,
        # Regularization
        dropout: float,
        label_smoothing: float,
        # Efficiency
        nbs: int,
        freeze: Optional[int],
        amp: bool,
        close_mosaic: int,
        seed: int,
        deterministic: bool,
        # Loss gains
        pose: float,
        kobj: float,
        box: float,
        cls: float,
        dfl: float,
        # Early stopping
        patience: int,
        # Checkpointing
        checkpoint_interval: int,
        checkpoint_bucket: str,
        checkpoint_prefix: str,
        # Augmentation
        hsv_h: float,
        hsv_s: float,
        hsv_v: float,
        degrees: float,
        translate: float,
        scale: float,
        shear: float,
        perspective: float,
        flipud: float,
        fliplr: float,
        mosaic: float,
        mixup: float,
        copy_paste: float,
        erasing: float,
        bgr: float,
    ) -> TrainingResult:
        """Execute the model training step end-to-end.

        All parameters originate from CLI flags passed by orchestrate.sh.
        """
        self._logger.info(
            "Starting %s | experiment=%s model=%s source=%s epochs=%d",
            self._config.app_name,
            experiment_name,
            model_variant,
            source,
            epochs,
        )

        params = TrainingParams(
            model_variant=model_variant,
            experiment_name=experiment_name,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            source=source,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            disk_cache_bytes=disk_cache_bytes,
            pretrained_weights=pretrained_weights,
            resume_from=resume_from,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            learning_rate=learning_rate,
            cos_lr=cos_lr,
            lrf=lrf,
            optimizer=optimizer,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            dropout=dropout,
            label_smoothing=label_smoothing,
            nbs=nbs,
            freeze=freeze,
            amp=amp,
            close_mosaic=close_mosaic,
            seed=seed,
            deterministic=deterministic,
            pose=pose,
            kobj=kobj,
            box=box,
            cls=cls,
            dfl=dfl,
            patience=patience,
            checkpoint_interval=checkpoint_interval,
            checkpoint_bucket=checkpoint_bucket,
            checkpoint_prefix=checkpoint_prefix,
            augmentation=AugmentationParams(
                hsv_h=hsv_h,
                hsv_s=hsv_s,
                hsv_v=hsv_v,
                degrees=degrees,
                translate=translate,
                scale=scale,
                shear=shear,
                perspective=perspective,
                flipud=flipud,
                fliplr=fliplr,
                mosaic=mosaic,
                mixup=mixup,
                copy_paste=copy_paste,
                erasing=erasing,
                bgr=bgr,
            ),
        )

        result = self._service.run(params=params)

        self._logger.info(
            "Training finished | run_id=%s mAP50=%.4f mAP50-95=%.4f best=%s",
            result.mlflow_run_id,
            result.final_map50,
            result.final_map50_95,
            result.best_checkpoint_s3,
        )

        return result
