"""Tests for TrainingService.

All external dependencies (ultralytics YOLO, mlflow, boto3, pynvml) are
mocked so that no real training, S3 calls, or GPU initialisation occur.
"""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from app.models.training import AugmentationParams, TrainingParams
from app.services.model_training import TrainingError, TrainingService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(
    dataset_dir: str,
    output_dir: str,
    source: str = "local",
    pretrained_weights: str | None = None,
    resume_from: str | None = None,
    **overrides: Any,
) -> TrainingParams:
    defaults: dict[str, Any] = dict(
        model_variant="yolov8n-pose.pt",
        experiment_name="test-exp",
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        source=source,
        s3_bucket=None,
        s3_prefix=None,
        pretrained_weights=pretrained_weights,
        resume_from=resume_from,
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
        augmentation=AugmentationParams(),
    )
    defaults.update(overrides)
    return TrainingParams(**defaults)


def _make_service(s3_client: Any = None) -> TrainingService:
    return TrainingService(
        s3_client=s3_client or MagicMock(),
        mlflow_tracking_uri="http://localhost:5000",
        resource_monitor_interval_sec=1,
    )


def _fake_trainer_save_dir(tmp_path: Path) -> Path:
    """Create a minimal Ultralytics-style save directory with expected artifacts."""
    save_dir = tmp_path / "runs" / "test-exp"
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_bytes(b"fake-best")
    (weights_dir / "last.pt").write_bytes(b"fake-last")
    (save_dir / "results.csv").write_text("epoch,mAP50\n0,0.5\n")
    (save_dir / "confusion_matrix.png").write_bytes(b"fake-png")
    return save_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dataset_dir(tmp_path: Path) -> str:
    """A minimal local YOLO dataset directory with data.yaml."""
    d = tmp_path / "dataset"
    d.mkdir()
    (d / "data.yaml").write_text(
        "path: /tmp/dataset\ntrain: images/train\nval: images/val\n"
        "test: images/test\nkpt_shape: [11, 3]\nnames: {0: spacecraft}\n"
    )
    return str(d)


@pytest.fixture()
def output_dir(tmp_path: Path) -> str:
    d = tmp_path / "runs"
    d.mkdir()
    return str(d)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_raises_when_pretrained_and_resume_both_set(
        self, dataset_dir: str, output_dir: str
    ) -> None:
        """Setting both --pretrained-weights and --resume-from is an error."""
        service = _make_service()
        params = _make_params(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            pretrained_weights="s3://bucket/weights.pt",
            resume_from="auto",
        )
        with pytest.raises(TrainingError, match="mutually exclusive"):
            service._validate_params(params)

    def test_raises_when_s3_source_missing_bucket(
        self, dataset_dir: str, output_dir: str
    ) -> None:
        """source='s3' without --s3-bucket raises TrainingError."""
        service = _make_service()
        params = _make_params(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            source="s3",
            s3_bucket=None,
            s3_prefix="some/prefix",
        )
        with pytest.raises(TrainingError, match="--s3-bucket"):
            service._validate_params(params)

    def test_raises_when_s3_source_missing_prefix(
        self, dataset_dir: str, output_dir: str
    ) -> None:
        """source='s3' without --s3-prefix raises TrainingError."""
        service = _make_service()
        params = _make_params(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            source="s3",
            s3_bucket="my-bucket",
            s3_prefix=None,
        )
        with pytest.raises(TrainingError, match="--s3-prefix"):
            service._validate_params(params)

    def test_raises_when_dataset_dir_missing(self, output_dir: str) -> None:
        """A non-existent dataset_dir raises TrainingError."""
        service = _make_service()
        params = _make_params(
            dataset_dir="/nonexistent/path",
            output_dir=output_dir,
        )
        with pytest.raises(TrainingError, match="does not exist"):
            service._validate_params(params)


# ---------------------------------------------------------------------------
# data.yaml tests
# ---------------------------------------------------------------------------


class TestWriteDataYaml:
    def test_uses_existing_data_yaml(
        self, dataset_dir: str, tmp_path: Path
    ) -> None:
        """Existing data.yaml is read and path field is overridden."""
        import yaml

        service = _make_service()
        params = _make_params(dataset_dir=dataset_dir, output_dir=str(tmp_path))
        result = service._write_data_yaml(params, tmp_path)

        with result.open() as fh:
            content = yaml.safe_load(fh)

        assert content["path"] == str(Path(dataset_dir).resolve())
        assert "train" in content

    def test_generates_default_yaml_when_missing(self, tmp_path: Path) -> None:
        """When data.yaml is absent, a default template is generated."""
        import yaml

        dataset_no_yaml = tmp_path / "no_yaml_dataset"
        dataset_no_yaml.mkdir()

        service = _make_service()
        params = _make_params(
            dataset_dir=str(dataset_no_yaml), output_dir=str(tmp_path)
        )
        dest = tmp_path / "tmp_yaml"
        dest.mkdir()
        result = service._write_data_yaml(params, dest)

        with result.open() as fh:
            content = yaml.safe_load(fh)

        assert content["path"] == str(dataset_no_yaml.resolve())
        assert "kpt_shape" in content
        assert content["names"] == {0: "spacecraft"}


# ---------------------------------------------------------------------------
# S3 download helpers
# ---------------------------------------------------------------------------


class TestMaybeDownloadPt:
    def test_downloads_s3_uri(self, tmp_path: Path) -> None:
        """An s3:// URI triggers a download_file call."""
        mock_s3 = MagicMock()
        service = _make_service(s3_client=mock_s3)

        result = service._maybe_download_pt(
            "s3://my-bucket/models/best.pt", tmp_path, "pretrained"
        )

        mock_s3.download_file.assert_called_once_with(
            "my-bucket", "models/best.pt", str(result)
        )
        assert result.name == "pretrained_weights.pt"

    def test_returns_local_path_unchanged(self, tmp_path: Path) -> None:
        """A local path is returned as-is without any S3 call."""
        local_pt = tmp_path / "my_weights.pt"
        local_pt.write_bytes(b"weights")

        mock_s3 = MagicMock()
        service = _make_service(s3_client=mock_s3)

        result = service._maybe_download_pt(str(local_pt), tmp_path, "pretrained")

        mock_s3.download_file.assert_not_called()
        assert result == local_pt

    def test_raises_for_missing_local_path(self, tmp_path: Path) -> None:
        """A local path that doesn't exist raises TrainingError."""
        service = _make_service()
        with pytest.raises(TrainingError, match="does not exist"):
            service._maybe_download_pt("/no/such/file.pt", tmp_path, "pretrained")


# ---------------------------------------------------------------------------
# Artifact logging
# ---------------------------------------------------------------------------


class TestLogArtifacts:
    def test_logs_expected_artifacts(self, tmp_path: Path) -> None:
        """best.pt, last.pt, .png plots, and results.csv are all logged."""
        save_dir = _fake_trainer_save_dir(tmp_path)
        service = _make_service()

        with patch("app.services.model_training.mlflow") as mock_mlflow:
            service._log_artifacts(save_dir)

        logged_calls = mock_mlflow.log_artifact.call_args_list
        logged_paths = [str(c[0][0]) for c in logged_calls]
        artifact_paths = [c[0][1] for c in logged_calls]

        assert any("best.pt" in p for p in logged_paths)
        assert any("last.pt" in p for p in logged_paths)
        assert any("results.csv" in p for p in logged_paths)
        assert any("confusion_matrix.png" in p for p in logged_paths)
        assert "weights" in artifact_paths
        assert "metrics" in artifact_paths
        assert "plots" in artifact_paths

    def test_missing_weight_file_does_not_raise(self, tmp_path: Path) -> None:
        """If best.pt or last.pt is absent, a warning is logged, not an exception."""
        save_dir = tmp_path / "empty_run"
        (save_dir / "weights").mkdir(parents=True)

        service = _make_service()
        with patch("app.services.model_training.mlflow"):
            # Should not raise even with no artifact files
            service._log_artifacts(save_dir)


# ---------------------------------------------------------------------------
# Full run — end-to-end with all heavy deps mocked
# ---------------------------------------------------------------------------


class TestTrainingServiceRun:
    def _run_with_mocks(
        self,
        dataset_dir: str,
        output_dir: str,
        tmp_path: Path,
        save_dir: Path,
        **param_overrides: Any,
    ) -> Any:
        """Execute service.run() with ultralytics, mlflow, and S3 fully mocked."""
        mock_s3 = MagicMock()
        service = _make_service(s3_client=mock_s3)

        params = _make_params(
            dataset_dir=dataset_dir, output_dir=output_dir, **param_overrides
        )

        # Build a mock trainer that references our fake save_dir
        mock_trainer = MagicMock()
        mock_trainer.save_dir = str(save_dir)
        mock_trainer.epoch = params.epochs - 1
        mock_trainer.metrics = {
            "metrics/mAP50(B)": 0.75,
            "metrics/mAP50-95(B)": 0.55,
            "metrics/precision(B)": 0.8,
            "metrics/recall(B)": 0.7,
        }
        mock_trainer.loss_items = [0.05, 0.03, 0.02, 0.01, 0.04]
        mock_trainer.last = str(save_dir / "weights" / "last.pt")

        mock_model = MagicMock()
        mock_model.train.return_value = mock_trainer
        mock_model.trainer = mock_trainer

        mock_yolo_cls = MagicMock(return_value=mock_model)

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id-123"

        with (
            patch("app.services.model_training.mlflow") as mock_mlflow,
            patch("app.services.resource_monitor.mlflow"),
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor._GPU_AVAILABLE", False),
        ):
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(
                return_value=mock_run
            )
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            vm = MagicMock()
            vm.used = 1e9
            vm.percent = 10.0
            mock_psutil.virtual_memory.return_value = vm
            mock_psutil.cpu_percent.return_value = 5.0

            result = service._run_with_mlflow(params, mock_run, mock_yolo_cls)

        return result, mock_model, mock_mlflow, mock_s3

    def test_successful_run_returns_training_result(
        self, dataset_dir: str, output_dir: str, tmp_path: Path
    ) -> None:
        """A successful run returns a populated TrainingResult."""
        from app.models.training import TrainingResult

        save_dir = _fake_trainer_save_dir(tmp_path)
        result, _, _, _ = self._run_with_mocks(
            dataset_dir, output_dir, tmp_path, save_dir
        )

        assert isinstance(result, TrainingResult)
        assert result.mlflow_run_id == "test-run-id-123"
        assert result.experiment_name == "test-exp"
        assert result.model_variant == "yolov8n-pose.pt"
        assert result.final_map50 == 0.75
        assert result.final_map50_95 == 0.55

    def test_model_train_called_with_correct_kwargs(
        self, dataset_dir: str, output_dir: str, tmp_path: Path
    ) -> None:
        """model.train() receives the expected keyword arguments."""
        save_dir = _fake_trainer_save_dir(tmp_path)
        result, mock_model, _, _ = self._run_with_mocks(
            dataset_dir, output_dir, tmp_path, save_dir
        )

        call_kwargs = mock_model.train.call_args[1]
        assert call_kwargs["epochs"] == 2
        assert call_kwargs["batch"] == 2
        assert call_kwargs["optimizer"] == "SGD"
        assert call_kwargs["lr0"] == 0.01
        assert call_kwargs["project"] == output_dir
        assert call_kwargs["name"] == "test-exp"

    def test_best_pt_uploaded_to_s3(
        self, dataset_dir: str, output_dir: str, tmp_path: Path
    ) -> None:
        """best.pt is uploaded to S3 after training completes."""
        save_dir = _fake_trainer_save_dir(tmp_path)
        _, _, _, mock_s3 = self._run_with_mocks(
            dataset_dir, output_dir, tmp_path, save_dir
        )

        upload_calls = [str(c) for c in mock_s3.upload_file.call_args_list]
        assert any("best.pt" in c for c in upload_calls)

    def test_resume_sets_resume_flag(
        self, dataset_dir: str, output_dir: str, tmp_path: Path
    ) -> None:
        """When resume_from='auto', model.train() is called with resume=True."""
        save_dir = _fake_trainer_save_dir(tmp_path)
        _, mock_model, _, _ = self._run_with_mocks(
            dataset_dir,
            output_dir,
            tmp_path,
            save_dir,
            resume_from="auto",
        )

        call_kwargs = mock_model.train.call_args[1]
        assert call_kwargs.get("resume") is True

    def test_training_failure_raises_training_error(
        self, dataset_dir: str, output_dir: str, tmp_path: Path
    ) -> None:
        """A crash inside model.train() is wrapped in TrainingError."""
        mock_s3 = MagicMock()
        service = _make_service(s3_client=mock_s3)
        params = _make_params(dataset_dir=dataset_dir, output_dir=output_dir)

        mock_model = MagicMock()
        mock_model.train.side_effect = RuntimeError("CUDA OOM")
        mock_model.trainer = MagicMock()
        mock_yolo_cls = MagicMock(return_value=mock_model)

        mock_run = MagicMock()
        mock_run.info.run_id = "fail-run"

        with (
            patch("app.services.model_training.mlflow") as mock_mlflow,
            patch("app.services.resource_monitor.mlflow"),
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor._GPU_AVAILABLE", False),
        ):
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(
                return_value=mock_run
            )
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            vm = MagicMock()
            vm.used = 1e9
            vm.percent = 10.0
            mock_psutil.virtual_memory.return_value = vm
            mock_psutil.cpu_percent.return_value = 5.0

            with pytest.raises(TrainingError, match="Training failed"):
                service.run(params)


# ---------------------------------------------------------------------------
# MLflow param logging
# ---------------------------------------------------------------------------


class TestLogParamsAndTags:
    def test_all_sections_logged(
        self, dataset_dir: str, output_dir: str
    ) -> None:
        """log_params is called with params from all config sections."""
        service = _make_service()
        params = _make_params(dataset_dir=dataset_dir, output_dir=output_dir)

        mock_run = MagicMock()
        mock_run.info.run_id = "x"

        with patch("app.services.model_training.mlflow") as mock_mlflow:
            service._log_params_and_tags(params, mock_run)

        # Collect all param keys logged across all log_params calls
        all_keys: set[str] = set()
        for c in mock_mlflow.log_params.call_args_list:
            all_keys.update(c[0][0].keys())

        assert "model.variant" in all_keys
        assert "training.epochs" in all_keys
        assert "training.optimizer" in all_keys
        assert "augmentation.hsv_h" in all_keys
        assert "training.pose" in all_keys

    def test_tags_set(self, dataset_dir: str, output_dir: str) -> None:
        """set_tags is called with the expected pipeline metadata."""
        service = _make_service()
        params = _make_params(dataset_dir=dataset_dir, output_dir=output_dir)

        mock_run = MagicMock()
        with patch("app.services.model_training.mlflow") as mock_mlflow:
            service._log_params_and_tags(params, mock_run)

        tags = mock_mlflow.set_tags.call_args[0][0]
        assert tags["pipeline.step"] == "model_training"
        assert tags["project"] == "infinite-orbits"
        assert tags["training.status"] == "RUNNING"