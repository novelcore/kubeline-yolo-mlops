import pytest
from pydantic import ValidationError

from app.models.pipeline_config import ExportConfig, PipelineConfig, RegistrationConfig


VALID_CONFIG: dict = {
    "experiment": {
        "name": "object-pose-v1-yolov8n",
        "description": "Baseline run",
        "tags": {"project": "example-project", "phase": "1"},
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
    assert config.experiment.name == "object-pose-v1-yolov8n"
    assert config.model.variant == "yolov8n-pose.pt"
    assert config.training.epochs == 100


def test_resources_section_is_ignored():
    data = {**VALID_CONFIG, "resources": {"gpu_count": 2, "gpu_type": "A100"}}
    config = PipelineConfig(**data)
    assert not hasattr(config, "resources")


def test_scheduler_sub_object_is_now_rejected():
    # cos_lr and lrf are now first-class fields in TrainingConfig (extra="forbid").
    # A legacy training.scheduler sub-object must raise ValidationError instead of
    # being silently dropped as it was in the previous schema.
    data = dict(VALID_CONFIG)
    data["training"] = {**VALID_CONFIG["training"], "scheduler": {"cos_lr": True, "lrf": 0.01}}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PipelineConfig(**data)


def test_cos_lr_and_lrf_as_first_class_fields():
    # cos_lr and lrf must be accepted as direct fields under training.
    data = dict(VALID_CONFIG)
    data["training"] = {**VALID_CONFIG["training"], "cos_lr": False, "lrf": 0.05}
    config = PipelineConfig(**data)
    assert config.training.cos_lr is False
    assert config.training.lrf == 0.05


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


# ---------------------------------------------------------------------------
# RegistrationConfig — model-level validation (F-01, F-02, F-03)
# ---------------------------------------------------------------------------

class TestRegistrationConfig:
    """Validation rules for RegistrationConfig and its integration with PipelineConfig."""

    def test_defaults_are_none(self) -> None:
        """Both fields default to None — no required fields (CON-02)."""
        r = RegistrationConfig()
        assert r.registered_model_name is None
        assert r.promote_to is None

    def test_promote_to_champion_is_valid(self) -> None:
        r = RegistrationConfig(promote_to="champion")
        assert r.promote_to == "champion"

    def test_promote_to_challenger_is_valid(self) -> None:
        r = RegistrationConfig(promote_to="challenger")
        assert r.promote_to == "challenger"

    def test_promote_to_invalid_value_fails(self) -> None:
        """AC-01: any value outside champion/challenger is rejected at parse time."""
        with pytest.raises(ValidationError, match="champion"):
            RegistrationConfig(promote_to="invalid-value")

    def test_registered_model_name_valid_characters(self) -> None:
        r = RegistrationConfig(registered_model_name="my-yolo-model_v1.0")
        assert r.registered_model_name == "my-yolo-model_v1.0"

    def test_registered_model_name_with_slash_fails(self) -> None:
        """AC-07: slash is not an MLflow-safe character."""
        with pytest.raises(ValidationError, match="invalid characters"):
            RegistrationConfig(registered_model_name="bad/name")

    def test_extra_key_is_rejected(self) -> None:
        """extra='forbid' catches typos like 'promoteto' (OQ-02 / T-03 mitigation)."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RegistrationConfig(promoteto="champion")

    def test_pipeline_config_without_registration_section_uses_defaults(self) -> None:
        """AC-02: omitting registration: entirely is backward-compatible (CON-02)."""
        config = PipelineConfig(**VALID_CONFIG)
        assert config.registration.registered_model_name is None
        assert config.registration.promote_to is None

    def test_pipeline_config_with_full_registration_section(self) -> None:
        data = {**VALID_CONFIG, "registration": {
            "registered_model_name": "my-yolo-model",
            "promote_to": "champion",
        }}
        config = PipelineConfig(**data)
        assert config.registration.registered_model_name == "my-yolo-model"


class TestExportConfig:
    """Validation rules for ExportConfig calibration and validation fields (F-01, F-08)."""

    def test_defaults_are_applied(self) -> None:
        """CON-02: ExportConfig with no args uses safe defaults for all new fields."""
        cfg = ExportConfig()
        assert cfg.calibration_method == "entropy"
        assert cfg.calibration_samples == 512
        assert cfg.per_channel is True
        assert cfg.symmetric is True
        assert cfg.validate_exports is False
        assert cfg.validation_samples == 100

    def test_calibration_method_lowercase_accepted(self) -> None:
        """AC-09: lowercase calibration_method values are accepted."""
        assert ExportConfig(calibration_method="entropy").calibration_method == "entropy"
        assert ExportConfig(calibration_method="minmax").calibration_method == "minmax"
        assert ExportConfig(calibration_method="percentile").calibration_method == "percentile"

    def test_calibration_method_uppercase_normalised(self) -> None:
        """AC-09: CON-03 — uppercase input is normalised to lowercase."""
        assert ExportConfig(calibration_method="Entropy").calibration_method == "entropy"
        assert ExportConfig(calibration_method="ENTROPY").calibration_method == "entropy"
        assert ExportConfig(calibration_method="MinMax").calibration_method == "minmax"

    def test_calibration_method_invalid_raises(self) -> None:
        with pytest.raises(ValidationError, match="calibration_method"):
            ExportConfig(calibration_method="linear")

    def test_calibration_samples_lower_boundary(self) -> None:
        """100 is the minimum valid value (ge=100)."""
        assert ExportConfig(calibration_samples=100).calibration_samples == 100

    def test_calibration_samples_upper_boundary(self) -> None:
        """10000 is the maximum valid value (le=10000)."""
        assert ExportConfig(calibration_samples=10000).calibration_samples == 10000

    def test_calibration_samples_below_minimum_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExportConfig(calibration_samples=99)

    def test_calibration_samples_above_maximum_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExportConfig(calibration_samples=10001)

    def test_pipeline_config_without_export_section_uses_defaults(self) -> None:
        """AC-08: existing pipeline YAML without export: section parses with defaults."""
        config = PipelineConfig(**VALID_CONFIG)
        assert config.export.enabled is False
        assert config.export.calibration_method == "entropy"
        assert config.export.validate_exports is False

    def test_pipeline_config_with_export_calibration_fields(self) -> None:
        """New calibration fields are read correctly from pipeline config."""
        data = {**VALID_CONFIG, "export": {
            "enabled": True,
            "formats": ["engine"],
            "precisions": ["int8"],
            "calibration_method": "Percentile",
            "calibration_samples": 256,
            "per_channel": False,
            "symmetric": False,
            "validate_exports": True,
            "validation_samples": 50,
        }}
        config = PipelineConfig(**data)
        assert config.export.calibration_method == "percentile"
        assert config.export.calibration_samples == 256
        assert config.export.per_channel is False
        assert config.export.symmetric is False
        assert config.export.validate_exports is True
        assert config.export.validation_samples == 50