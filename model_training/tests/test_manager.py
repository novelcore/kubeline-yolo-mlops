"""Tests for the Manager class."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.models.config import Config
from app.manager import Manager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_run_kwargs(dataset_dir: str, output_dir: str) -> dict[str, Any]:
    """Minimal complete set of Manager.run() keyword arguments."""
    return dict(
        model_variant="yolov8n-pose.pt",
        experiment_name="test-exp",
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        source="local",
        s3_bucket=None,
        s3_prefix=None,
        pretrained_weights=None,
        resume_from=None,
        epochs=2,
        batch_size=2,
        image_size=640,
        learning_rate=0.01,
        cos_lr=True,
        lrf=0.01,
        optimizer="SGD",
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        warmup_momentum=0.8,
        dropout=0.0,
        label_smoothing=0.0,
        nbs=64,
        freeze=None,
        amp=False,
        close_mosaic=0,
        seed=0,
        deterministic=False,
        pose=12.0,
        kobj=2.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        patience=5,
        checkpoint_interval=1,
        checkpoint_bucket="test-bucket",
        checkpoint_prefix="checkpoints",
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.4,
        bgr=0.0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dataset_dir(tmp_path: Path) -> str:
    d = tmp_path / "dataset"
    d.mkdir()
    (d / "data.yaml").write_text("path: /tmp\ntrain: images/train\n")
    return str(d)


@pytest.fixture()
def output_dir(tmp_path: Path) -> str:
    d = tmp_path / "runs"
    d.mkdir()
    return str(d)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestManagerInit:
    def test_builds_with_default_config(self) -> None:
        """Manager constructs successfully with env-provided Config defaults."""
        with patch("app.manager.boto3"):
            manager = Manager()
        assert manager._config.app_name == "io-model-training"

    def test_accepts_injected_config(self) -> None:
        """Manager uses an injected Config instance."""
        config = Config(log_level="DEBUG")
        with patch("app.manager.boto3"):
            manager = Manager(config=config)
        assert manager._config.log_level == "DEBUG"

    def test_lakefs_client_used_when_endpoint_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When LAKEFS_ENDPOINT is set, the S3 client is pointed at LakeFS."""
        monkeypatch.setenv("LAKEFS_ENDPOINT", "https://lakefs.test")
        monkeypatch.setenv("LAKEFS_ACCESS_KEY", "lf-key")
        monkeypatch.setenv("LAKEFS_SECRET_KEY", "lf-secret")

        with patch("app.manager.boto3") as mock_boto3:
            Manager()
            call_kwargs = mock_boto3.client.call_args[1]
            assert call_kwargs["endpoint_url"] == "https://lakefs.test"
            assert call_kwargs["aws_access_key_id"] == "lf-key"


class TestManagerRun:
    def test_delegates_to_service(
        self, dataset_dir: str, output_dir: str
    ) -> None:
        """Manager.run() calls TrainingService.run() with a TrainingParams object."""
        from app.models.training import TrainingResult

        fake_result = TrainingResult(
            experiment_name="test-exp",
            model_variant="yolov8n-pose.pt",
            mlflow_run_id="run-abc",
            best_checkpoint_local="/tmp/best.pt",
            best_checkpoint_s3="s3://bucket/checkpoints/test-exp/best.pt",
            epochs_completed=2,
            final_map50=0.7,
            final_map50_95=0.5,
        )

        with (
            patch("app.manager.boto3"),
            patch("app.manager.TrainingService") as mock_svc_cls,
        ):
            mock_svc = MagicMock()
            mock_svc.run.return_value = fake_result
            mock_svc_cls.return_value = mock_svc

            manager = Manager()
            result = manager.run(**_default_run_kwargs(dataset_dir, output_dir))

        mock_svc.run.assert_called_once()
        call_params = mock_svc.run.call_args[1]["params"]
        assert call_params.model_variant == "yolov8n-pose.pt"
        assert call_params.experiment_name == "test-exp"
        assert call_params.epochs == 2
        assert result is fake_result

    def test_augmentation_params_forwarded(
        self, dataset_dir: str, output_dir: str
    ) -> None:
        """Augmentation kwargs are packaged into AugmentationParams correctly."""
        from app.models.training import TrainingResult

        fake_result = TrainingResult(
            experiment_name="test-exp",
            model_variant="yolov8n-pose.pt",
            mlflow_run_id="run-abc",
            best_checkpoint_local="/tmp/best.pt",
            best_checkpoint_s3="s3://bucket/best.pt",
            epochs_completed=2,
            final_map50=0.0,
            final_map50_95=0.0,
        )

        kwargs = _default_run_kwargs(dataset_dir, output_dir)
        kwargs["hsv_h"] = 0.05
        kwargs["flipud"] = 0.3

        with (
            patch("app.manager.boto3"),
            patch("app.manager.TrainingService") as mock_svc_cls,
        ):
            mock_svc = MagicMock()
            mock_svc.run.return_value = fake_result
            mock_svc_cls.return_value = mock_svc

            Manager().run(**kwargs)

        call_params = mock_svc.run.call_args[1]["params"]
        assert call_params.augmentation.hsv_h == 0.05
        assert call_params.augmentation.flipud == 0.3