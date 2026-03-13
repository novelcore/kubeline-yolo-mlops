"""CLI interface for the Config Validation step."""

from typing import Optional

import typer

from app.manager import Manager


def main() -> None:
    """Main CLI entry point for the Config Validation step."""
    app = typer.Typer(
        name="config-validation",
        help="Validate the Infinite Orbits MLOps pipeline configuration.",
        no_args_is_help=True,
    )

    @app.command(name="run", help="Validate a pipeline configuration.")
    def run(
        # Experiment
        experiment_name: str = typer.Option(..., help="Experiment name (alphanumeric and hyphens)."),
        experiment_description: Optional[str] = typer.Option(default=None, help="Optional experiment description."),
        # Dataset
        dataset_version: str = typer.Option(..., help="Dataset version string."),
        dataset_source: str = typer.Option(..., help="Dataset source: 's3' or 'lakefs'."),
        dataset_lakefs_repo: Optional[str] = typer.Option(default=None, help="LakeFS repository name (required when source=lakefs)."),
        dataset_lakefs_branch: Optional[str] = typer.Option(default=None, help="LakeFS branch name (required when source=lakefs)."),
        dataset_path_override: Optional[str] = typer.Option(default=None, help="Optional S3/LakeFS path override for the dataset."),
        dataset_sample_size: Optional[int] = typer.Option(default=None, help="Optional sample size (must be > 0)."),
        dataset_seed: int = typer.Option(default=42, help="Random seed for dataset splitting."),
        # Model
        model_variant: str = typer.Option(..., help="YOLO Pose model variant (e.g. yolov8n-pose.pt)."),
        model_pretrained_weights: Optional[str] = typer.Option(default=None, help="Optional S3 path to pretrained weights."),
        # Training — core schedule
        training_epochs: int = typer.Option(..., help="Number of training epochs (> 0)."),
        training_batch_size: int = typer.Option(..., help="Training batch size (> 0)."),
        training_image_size: int = typer.Option(..., help="Input image size in pixels (multiple of 32)."),
        training_learning_rate: float = typer.Option(..., help="Learning rate (> 0)."),
        training_optimizer: str = typer.Option(..., help="Optimizer: 'SGD', 'Adam', or 'AdamW'."),
        # Training — learning rate schedule
        training_cos_lr: bool = typer.Option(default=True, help="Use cosine learning rate schedule."),
        training_lrf: float = typer.Option(default=0.01, help="Final LR ratio: final_lr = learning_rate * lrf."),
        # Training — optimizer extras
        training_momentum: float = typer.Option(default=0.937, help="SGD momentum / Adam beta1."),
        training_weight_decay: float = typer.Option(default=0.0005, help="Weight decay."),
        # Training — warmup
        training_warmup_epochs: float = typer.Option(default=3.0, help="Warmup epochs."),
        training_warmup_momentum: float = typer.Option(default=0.8, help="Warmup momentum."),
        # Training — regularization
        training_dropout: float = typer.Option(default=0.0, help="Dropout rate."),
        training_label_smoothing: float = typer.Option(default=0.0, help="Label smoothing epsilon."),
        # Training — gradient accumulation
        training_nbs: int = typer.Option(default=64, help="Nominal batch size for gradient scaling."),
        # Training — fine-tuning
        training_freeze: Optional[int] = typer.Option(default=None, help="Freeze the first N layers."),
        # Training — efficiency
        training_amp: bool = typer.Option(default=True, help="Enable Automatic Mixed Precision."),
        training_close_mosaic: int = typer.Option(default=10, help="Disable mosaic for last N epochs."),
        # Training — reproducibility
        training_seed: int = typer.Option(default=0, help="Global training RNG seed."),
        training_deterministic: bool = typer.Option(default=True, help="Enable deterministic mode."),
        # Training — pose loss gains
        training_pose: float = typer.Option(default=12.0, help="Keypoint regression loss gain."),
        training_kobj: float = typer.Option(default=2.0, help="Keypoint objectness loss gain."),
        training_box: float = typer.Option(default=7.5, help="Bounding-box regression loss gain."),
        training_cls: float = typer.Option(default=0.5, help="Classification loss gain."),
        training_dfl: float = typer.Option(default=1.5, help="Distribution Focal Loss gain."),
        # Checkpointing
        checkpointing_interval_epochs: int = typer.Option(..., help="Save a checkpoint every N epochs (> 0)."),
        checkpointing_storage_path: str = typer.Option(..., help="S3 or LakeFS URI for checkpoint storage."),
        checkpointing_resume_from: Optional[str] = typer.Option(default=None, help="Checkpoint to resume from: null, 'auto', or an S3 path."),
        # Early stopping
        early_stopping_patience: int = typer.Option(..., help="Early stopping patience in epochs (> 0)."),
        # Augmentation
        aug_hsv_h: float = typer.Option(default=0.015, help="HSV hue augmentation range."),
        aug_hsv_s: float = typer.Option(default=0.7, help="HSV saturation augmentation range."),
        aug_hsv_v: float = typer.Option(default=0.4, help="HSV value augmentation range."),
        aug_degrees: float = typer.Option(default=0.0, help="Random rotation range (degrees)."),
        aug_translate: float = typer.Option(default=0.1, help="Random translation fraction."),
        aug_scale: float = typer.Option(default=0.5, help="Random scale gain."),
        aug_shear: float = typer.Option(default=0.0, help="Random shear range (degrees)."),
        aug_perspective: float = typer.Option(default=0.0, help="Random perspective warp [0.0, 0.001]."),
        aug_flipud: float = typer.Option(default=0.0, help="Vertical flip probability."),
        aug_fliplr: float = typer.Option(default=0.0, help="Horizontal flip probability."),
        aug_mosaic: float = typer.Option(default=1.0, help="Mosaic augmentation probability."),
        aug_mixup: float = typer.Option(default=0.0, help="MixUp augmentation probability."),
        aug_copy_paste: float = typer.Option(default=0.0, help="Segment copy-paste probability."),
        aug_erasing: float = typer.Option(default=0.4, help="Random erasing probability [0.0, 0.9]."),
        aug_bgr: float = typer.Option(default=0.0, help="BGR channel flip probability."),
        # Output
        output_path: Optional[str] = typer.Option(
            default=None,
            help="Path to write the validated config JSON artifact. If not set, no file is written.",
        ),
    ) -> None:
        try:
            config_dict = {
                "experiment": {
                    "name": experiment_name,
                    "description": experiment_description,
                },
                "dataset": {
                    "version": dataset_version,
                    "source": dataset_source,
                    "lakefs_repo": dataset_lakefs_repo,
                    "lakefs_branch": dataset_lakefs_branch,
                    "path_override": dataset_path_override,
                    "sample_size": dataset_sample_size,
                    "seed": dataset_seed,
                },
                "model": {
                    "variant": model_variant,
                    "pretrained_weights": model_pretrained_weights,
                },
                "training": {
                    "epochs": training_epochs,
                    "batch_size": training_batch_size,
                    "image_size": training_image_size,
                    "learning_rate": training_learning_rate,
                    "optimizer": training_optimizer,
                    "cos_lr": training_cos_lr,
                    "lrf": training_lrf,
                    "momentum": training_momentum,
                    "weight_decay": training_weight_decay,
                    "warmup_epochs": training_warmup_epochs,
                    "warmup_momentum": training_warmup_momentum,
                    "dropout": training_dropout,
                    "label_smoothing": training_label_smoothing,
                    "nbs": training_nbs,
                    "freeze": training_freeze,
                    "amp": training_amp,
                    "close_mosaic": training_close_mosaic,
                    "seed": training_seed,
                    "deterministic": training_deterministic,
                    "pose": training_pose,
                    "kobj": training_kobj,
                    "box": training_box,
                    "cls": training_cls,
                    "dfl": training_dfl,
                },
                "checkpointing": {
                    "interval_epochs": checkpointing_interval_epochs,
                    "storage_path": checkpointing_storage_path,
                    "resume_from": checkpointing_resume_from,
                },
                "early_stopping": {
                    "patience": early_stopping_patience,
                },
                "augmentation": {
                    "hsv_h": aug_hsv_h,
                    "hsv_s": aug_hsv_s,
                    "hsv_v": aug_hsv_v,
                    "degrees": aug_degrees,
                    "translate": aug_translate,
                    "scale": aug_scale,
                    "shear": aug_shear,
                    "perspective": aug_perspective,
                    "flipud": aug_flipud,
                    "fliplr": aug_fliplr,
                    "mosaic": aug_mosaic,
                    "mixup": aug_mixup,
                    "copy_paste": aug_copy_paste,
                    "erasing": aug_erasing,
                    "bgr": aug_bgr,
                },
            }

            manager = Manager()
            manager.run(config_dict=config_dict, output_path=output_path)
        except Exception as e:
            print(f"Error: {e}")
            raise typer.Exit(code=1)

    app()
