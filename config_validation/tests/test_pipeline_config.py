import pytest
from pydantic import ValidationError

from app.models.pipeline_config import PipelineConfig


VALID_CONFIG: dict = {
    "experiment": {
        "name": "spacecraft-pose-v1-yolov8n",
        "description": "Baseline run",
        "tags": {"project": "infinite-orbits", "phase": "1"},
    },
    "dataset": {
        "version": "v1",
        "source": "s3",
        "path_override": None,
        "sample_size": None,
        "seed": 42,
    },
    "model": {
        "variant": "yolov8n-pose.pt",
        "pretrained_weights": None,
    },
    "training": {
        "epochs": 100,
        "batch_size": 16,
        "image_size": 640,
        "learning_rate": 0.01,
        "optimizer": "SGD",
    },
    "checkpointing": {
        "interval_epochs": 10,
        "storage_path": "s3://io-mlops/checkpoints",
        "resume_from": None,
    },
    "early_stopping": {
        "patience": 50,
    },
}


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_valid_config_parses_successfully():
    config = PipelineConfig(**VALID_CONFIG)
    assert config.experiment.name == "spacecraft-pose-v1-yolov8n"
    assert config.model.variant == "yolov8n-pose.pt"
    assert config.training.epochs == 100


def test_resources_section_is_ignored():
    data = {**VALID_CONFIG, "resources": {"gpu_count": 2, "gpu_type": "A100"}}
    config = PipelineConfig(**data)
    assert not hasattr(config, "resources")


def test_scheduler_sub_object_is_ignored():
    data = dict(VALID_CONFIG)
    data["training"] = {**VALID_CONFIG["training"], "scheduler": {"cos_lr": True, "lrf": 0.01}}
    config = PipelineConfig(**data)
    assert not hasattr(config.training, "scheduler")


def test_augmentation_defaults_applied_when_section_omitted():
    data = {k: v for k, v in VALID_CONFIG.items() if k != "augmentation"}
    config = PipelineConfig(**data)
    assert config.augmentation.mosaic == 1.0
    assert config.augmentation.fliplr == 0.0


def test_sample_size_none_is_valid():
    data = dict(VALID_CONFIG)
    data["dataset"] = {**VALID_CONFIG["dataset"], "sample_size": None}
    PipelineConfig(**data)  # must not raise


def test_sample_size_positive_integer_is_valid():
    data = dict(VALID_CONFIG)
    data["dataset"] = {**VALID_CONFIG["dataset"], "sample_size": 1000}
    config = PipelineConfig(**data)
    assert config.dataset.sample_size == 1000


def test_resume_from_null_is_valid():
    PipelineConfig(**VALID_CONFIG)  # resume_from: null by default


def test_resume_from_auto_is_valid():
    data = dict(VALID_CONFIG)
    data["checkpointing"] = {**VALID_CONFIG["checkpointing"], "resume_from": "auto"}
    PipelineConfig(**data)


def test_resume_from_s3_path_is_valid():
    data = dict(VALID_CONFIG)
    data["checkpointing"] = {
        **VALID_CONFIG["checkpointing"],
        "resume_from": "s3://io-mlops/checkpoints/exp/last.pt",
    }
    PipelineConfig(**data)


def test_all_yolo_pose_variants_are_valid():
    valid_variants = [
        f"yolov{ver}{size}-pose.pt"
        for ver in ("8", "9", "10", "11")
        for size in ("n", "s", "m", "l", "x")
    ]
    for variant in valid_variants:
        data = dict(VALID_CONFIG)
        data["model"] = {**VALID_CONFIG["model"], "variant": variant}
        config = PipelineConfig(**data)
        assert config.model.variant == variant


# ---------------------------------------------------------------------------
# Schema failures — experiment
# ---------------------------------------------------------------------------

def test_missing_experiment_name_fails():
    data = dict(VALID_CONFIG)
    data["experiment"] = {"description": "no name"}
    with pytest.raises(ValidationError):
        PipelineConfig(**data)


def test_empty_experiment_name_fails():
    data = dict(VALID_CONFIG)
    data["experiment"] = {**VALID_CONFIG["experiment"], "name": ""}
    with pytest.raises(ValidationError, match="must not be empty"):
        PipelineConfig(**data)


def test_invalid_experiment_name_characters_fails():
    data = dict(VALID_CONFIG)
    data["experiment"] = {**VALID_CONFIG["experiment"], "name": "invalid name!"}
    with pytest.raises(ValidationError, match="alphanumeric"):
        PipelineConfig(**data)


# ---------------------------------------------------------------------------
# Schema failures — model
# ---------------------------------------------------------------------------

def test_invalid_model_variant_not_pose_fails():
    data = dict(VALID_CONFIG)
    data["model"] = {**VALID_CONFIG["model"], "variant": "yolov8n.pt"}
    with pytest.raises(ValidationError, match="not a valid YOLO Pose variant"):
        PipelineConfig(**data)


def test_invalid_model_variant_unknown_version_fails():
    data = dict(VALID_CONFIG)
    data["model"] = {**VALID_CONFIG["model"], "variant": "yolov7n-pose.pt"}
    with pytest.raises(ValidationError, match="not a valid YOLO Pose variant"):
        PipelineConfig(**data)


# ---------------------------------------------------------------------------
# Schema failures — training
# ---------------------------------------------------------------------------

def test_epochs_zero_fails():
    data = dict(VALID_CONFIG)
    data["training"] = {**VALID_CONFIG["training"], "epochs": 0}
    with pytest.raises(ValidationError):
        PipelineConfig(**data)


def test_epochs_negative_fails():
    data = dict(VALID_CONFIG)
    data["training"] = {**VALID_CONFIG["training"], "epochs": -1}
    with pytest.raises(ValidationError):
        PipelineConfig(**data)


def test_image_size_multiple_of_32_passes():
    data = dict(VALID_CONFIG)
    data["training"] = {**VALID_CONFIG["training"], "image_size": 640}
    PipelineConfig(**data)


def test_image_size_not_multiple_of_32_fails():
    data = dict(VALID_CONFIG)
    data["training"] = {**VALID_CONFIG["training"], "image_size": 641}
    with pytest.raises(ValidationError, match="multiple of 32"):
        PipelineConfig(**data)


def test_invalid_optimizer_fails():
    data = dict(VALID_CONFIG)
    data["training"] = {**VALID_CONFIG["training"], "optimizer": "rmsprop"}
    with pytest.raises(ValidationError, match="optimizer"):
        PipelineConfig(**data)


def test_learning_rate_zero_fails():
    data = dict(VALID_CONFIG)
    data["training"] = {**VALID_CONFIG["training"], "learning_rate": 0.0}
    with pytest.raises(ValidationError):
        PipelineConfig(**data)


# ---------------------------------------------------------------------------
# Schema failures — checkpointing
# ---------------------------------------------------------------------------

def test_checkpointing_interval_zero_fails():
    data = dict(VALID_CONFIG)
    data["checkpointing"] = {**VALID_CONFIG["checkpointing"], "interval_epochs": 0}
    with pytest.raises(ValidationError):
        PipelineConfig(**data)


def test_invalid_storage_path_no_scheme_fails():
    data = dict(VALID_CONFIG)
    data["checkpointing"] = {**VALID_CONFIG["checkpointing"], "storage_path": "/local/path"}
    with pytest.raises(ValidationError, match="s3://"):
        PipelineConfig(**data)


def test_invalid_resume_from_local_path_fails():
    data = dict(VALID_CONFIG)
    data["checkpointing"] = {**VALID_CONFIG["checkpointing"], "resume_from": "local/path.pt"}
    with pytest.raises(ValidationError, match="resume_from"):
        PipelineConfig(**data)


# ---------------------------------------------------------------------------
# Schema failures — early stopping
# ---------------------------------------------------------------------------

def test_early_stopping_patience_zero_fails():
    data = dict(VALID_CONFIG)
    data["early_stopping"] = {"patience": 0}
    with pytest.raises(ValidationError):
        PipelineConfig(**data)


# ---------------------------------------------------------------------------
# Schema failures — dataset
# ---------------------------------------------------------------------------

def test_dataset_sample_size_zero_fails():
    data = dict(VALID_CONFIG)
    data["dataset"] = {**VALID_CONFIG["dataset"], "sample_size": 0}
    with pytest.raises(ValidationError, match="must be > 0"):
        PipelineConfig(**data)


def test_invalid_dataset_source_fails():
    data = dict(VALID_CONFIG)
    data["dataset"] = {**VALID_CONFIG["dataset"], "source": "gcs"}
    with pytest.raises(ValidationError, match="source"):
        PipelineConfig(**data)


def test_missing_dataset_version_fails():
    data = dict(VALID_CONFIG)
    data["dataset"] = {k: v for k, v in VALID_CONFIG["dataset"].items() if k != "version"}
    with pytest.raises(ValidationError):
        PipelineConfig(**data)