"""CLI interface for the Model Training step.

All hyperparameters are accepted as Typer options so that orchestrate.sh
can pass them directly as flags without any YAML parsing in this step.
Secrets (S3 credentials, MLflow URI) are read from environment variables
via Config / .env.
"""

from typing import Optional

import typer

from app.manager import Manager


def main() -> None:
    """Main CLI entry point for the Model Training step."""
    app = typer.Typer(
        name="model-training",
        help="Train a YOLO pose-estimation model with MLflow experiment tracking.",
        no_args_is_help=True,
    )

    @app.command(name="run", help="Run a YOLO training job end-to-end.")
    def run(
        # ---- Identity ----
        model_variant: str = typer.Option(
            ...,
            help="YOLO model variant file (e.g. 'yolov8n-pose.pt').",
        ),
        experiment_name: str = typer.Option(
            ...,
            help=(
                "MLflow experiment name and S3 checkpoint path prefix "
                "(e.g. 'spacecraft-pose-v1-yolov8n')."
            ),
        ),
        dataset_dir: str = typer.Option(
            ...,
            help="Local path to the YOLO dataset directory produced by dataset_loading.",
        ),
        output_dir: str = typer.Option(
            ...,
            help="Local directory for Ultralytics runs/ output.",
        ),
        # ---- Dataset source ----
        source: str = typer.Option(
            "local",
            help="Dataset read mode: 'local' reads from dataset_dir, 's3' streams from S3.",
        ),
        s3_bucket: Optional[str] = typer.Option(
            None,
            help="S3 bucket containing images (required when --source=s3).",
        ),
        s3_prefix: Optional[str] = typer.Option(
            None,
            help="S3 key prefix for images (required when --source=s3).",
        ),
        # ---- Weight init / resume ----
        pretrained_weights: Optional[str] = typer.Option(
            None,
            help=(
                "Local path or s3:// URI to a custom .pt file for weight-only "
                "initialisation. Training starts from epoch 0. "
                "Mutually exclusive with --resume-from."
            ),
        ),
        resume_from: Optional[str] = typer.Option(
            None,
            help=(
                "'auto', local path, or s3:// URI for full Ultralytics resume "
                "(restores epoch counter, optimizer, and scheduler state). "
                "Mutually exclusive with --pretrained-weights."
            ),
        ),
        # ---- Core schedule ----
        epochs: int = typer.Option(100, help="Number of training epochs."),
        batch_size: int = typer.Option(16, help="Training batch size."),
        image_size: int = typer.Option(
            640, help="Input image size (must be a multiple of 32)."
        ),
        # ---- Learning rate ----
        learning_rate: float = typer.Option(
            0.01, help="Initial learning rate (Ultralytics lr0)."
        ),
        cos_lr: bool = typer.Option(True, help="Use cosine learning rate schedule."),
        lrf: float = typer.Option(
            0.01,
            help="Final LR ratio: final_lr = learning_rate * lrf. Must be in (0, 1].",
        ),
        # ---- Optimizer ----
        optimizer: str = typer.Option("SGD", help="Optimizer: SGD | Adam | AdamW."),
        momentum: float = typer.Option(0.937, help="SGD momentum / Adam beta1."),
        weight_decay: float = typer.Option(0.0005, help="Optimizer weight decay."),
        # ---- Warmup ----
        warmup_epochs: float = typer.Option(3.0, help="Number of warmup epochs."),
        warmup_momentum: float = typer.Option(0.8, help="Initial momentum during warmup."),
        # ---- Regularization ----
        dropout: float = typer.Option(
            0.0, help="Dropout rate applied in the classifier head."
        ),
        label_smoothing: float = typer.Option(
            0.0, help="Label smoothing epsilon [0.0, 1.0)."
        ),
        # ---- Training efficiency ----
        nbs: int = typer.Option(
            64,
            help=(
                "Nominal batch size. Ultralytics scales gradients so effective LR "
                "matches training at this batch size."
            ),
        ),
        freeze: Optional[int] = typer.Option(
            None,
            help="Freeze the first N layers. Omit to train all layers.",
        ),
        amp: bool = typer.Option(True, help="Enable Automatic Mixed Precision (FP16)."),
        close_mosaic: int = typer.Option(
            10,
            help="Disable mosaic augmentation for the last N epochs (0 = never disable).",
        ),
        seed: int = typer.Option(0, help="Global training RNG seed."),
        deterministic: bool = typer.Option(
            True,
            help="Enable cuDNN deterministic mode (slight speed penalty).",
        ),
        # ---- Pose loss gains ----
        pose: float = typer.Option(12.0, help="Keypoint regression loss gain."),
        kobj: float = typer.Option(2.0, help="Keypoint objectness loss gain."),
        box: float = typer.Option(7.5, help="Bounding-box regression loss gain."),
        cls: float = typer.Option(0.5, help="Classification loss gain."),
        dfl: float = typer.Option(1.5, help="Distribution Focal Loss gain."),
        # ---- Early stopping ----
        patience: int = typer.Option(
            50, help="Epochs without improvement before stopping."
        ),
        # ---- Checkpointing ----
        checkpoint_interval: int = typer.Option(
            10,
            help="Upload an intermediate checkpoint to S3 every N epochs.",
        ),
        checkpoint_bucket: str = typer.Option(
            "temp-mlops",
            help="S3 bucket for checkpoint storage.",
        ),
        checkpoint_prefix: str = typer.Option(
            "checkpoints",
            help="S3 key prefix for checkpoints (full path: <prefix>/<experiment_name>/).",
        ),
        # ---- Augmentation ----
        hsv_h: float = typer.Option(0.015, help="HSV hue augmentation range."),
        hsv_s: float = typer.Option(0.7, help="HSV saturation augmentation range."),
        hsv_v: float = typer.Option(0.4, help="HSV value augmentation range."),
        degrees: float = typer.Option(0.0, help="Random rotation range (degrees)."),
        translate: float = typer.Option(0.1, help="Random translation fraction."),
        scale: float = typer.Option(0.5, help="Random scale gain."),
        shear: float = typer.Option(0.0, help="Random shear range (degrees)."),
        perspective: float = typer.Option(
            0.0, help="Random perspective warp [0.0, 0.001]."
        ),
        flipud: float = typer.Option(0.0, help="Vertical flip probability."),
        fliplr: float = typer.Option(0.0, help="Horizontal flip probability."),
        mosaic: float = typer.Option(1.0, help="Mosaic augmentation probability."),
        mixup: float = typer.Option(0.0, help="MixUp augmentation probability."),
        copy_paste: float = typer.Option(0.0, help="Segment copy-paste probability."),
        erasing: float = typer.Option(
            0.4, help="Random erasing probability [0.0, 0.9]."
        ),
        bgr: float = typer.Option(0.0, help="BGR channel flip probability."),
    ) -> None:
        try:
            manager = Manager()
            manager.run(
                model_variant=model_variant,
                experiment_name=experiment_name,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                source=source,
                s3_bucket=s3_bucket,
                s3_prefix=s3_prefix,
                pretrained_weights=pretrained_weights,
                resume_from=resume_from,
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
            )
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

    app()
