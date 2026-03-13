"""Tests for TrainingService.

All external dependencies (ultralytics YOLO, mlflow, boto3, pynvml) are
mocked so that no real training, S3 calls, or GPU initialisation occur.
"""

import json
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
    """A minimal local YOLO dataset directory with data.yaml and image files."""
    d = tmp_path / "dataset"
    d.mkdir()
    (d / "data.yaml").write_text(
        "path: /tmp/dataset\ntrain: images/train\nval: images/val\n"
        "test: images/test\nkpt_shape: [11, 3]\nnames: {0: spacecraft}\n"
    )
    # Create image directories with at least one file so local validation passes
    for split in ("train", "val"):
        img_dir = d / "images" / split
        img_dir.mkdir(parents=True)
        (img_dir / "img_001.jpg").write_bytes(b"fake-image")
    return str(d)


@pytest.fixture()
def output_dir(tmp_path: Path) -> str:
    d = tmp_path / "runs"
    d.mkdir()
    return str(d)


# ---------------------------------------------------------------------------
# GPU index parsing
# ---------------------------------------------------------------------------


class TestParseGpuIndex:
    """Tests for TrainingService._parse_gpu_index."""

    def test_single_digit_string(self) -> None:
        assert TrainingService._parse_gpu_index("0") == 0

    def test_integer_input(self) -> None:
        assert TrainingService._parse_gpu_index(0) == 0

    def test_multi_digit_string(self) -> None:
        assert TrainingService._parse_gpu_index("2") == 2

    def test_multi_gpu_string_returns_none(self) -> None:
        assert TrainingService._parse_gpu_index("0,1") is None

    def test_cpu_string_returns_none(self) -> None:
        assert TrainingService._parse_gpu_index("cpu") is None

    def test_none_with_gpu_available(self) -> None:
        with patch("app.services.resource_monitor._GPU_AVAILABLE", True):
            assert TrainingService._parse_gpu_index(None) == 0

    def test_none_without_gpu(self) -> None:
        with patch("app.services.resource_monitor._GPU_AVAILABLE", False):
            assert TrainingService._parse_gpu_index(None) is None


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestTrainingParamsValidation:
    """Tests for TrainingParams Pydantic validation rules."""

    def test_image_size_must_be_multiple_of_32(self, dataset_dir: str, output_dir: str) -> None:
        """image_size=500 raises ValidationError (not a multiple of 32)."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="multiple"):
            _make_params(dataset_dir=dataset_dir, output_dir=output_dir, image_size=500)

    def test_image_size_valid_multiple_of_32(self, dataset_dir: str, output_dir: str) -> None:
        """image_size=320 is accepted."""
        params = _make_params(dataset_dir=dataset_dir, output_dir=output_dir, image_size=320)
        assert params.image_size == 320

    def test_optimizer_auto_accepted(self, dataset_dir: str, output_dir: str) -> None:
        """optimizer='auto' is accepted."""
        params = _make_params(dataset_dir=dataset_dir, output_dir=output_dir, optimizer="auto")
        assert params.optimizer == "auto"

    def test_degrees_upper_bound(self, dataset_dir: str, output_dir: str) -> None:
        """degrees > 360 raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="less than or equal"):
            _make_params(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                augmentation=AugmentationParams(degrees=400.0),
            )


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
# Local dataset validation tests
# ---------------------------------------------------------------------------


class TestLocalDatasetValidation:
    def test_passes_with_valid_structure(self, dataset_dir: str, output_dir: str) -> None:
        """_validate_local_dataset does not raise for a valid dataset directory."""
        service = _make_service()
        params = _make_params(dataset_dir=dataset_dir, output_dir=output_dir)
        # Should not raise
        service._validate_local_dataset(params)

    def test_raises_when_data_yaml_missing(self, tmp_path: Path, output_dir: str) -> None:
        """Missing data.yaml raises TrainingError in local mode."""
        d = tmp_path / "no_yaml"
        d.mkdir()
        # Create image dirs but no data.yaml
        for split in ("train", "val"):
            img_dir = d / "images" / split
            img_dir.mkdir(parents=True)
            (img_dir / "img.jpg").write_bytes(b"x")

        service = _make_service()
        params = _make_params(dataset_dir=str(d), output_dir=output_dir)
        with pytest.raises(TrainingError, match="data.yaml not found"):
            service._validate_local_dataset(params)

    def test_raises_when_train_images_dir_missing(
        self, tmp_path: Path, output_dir: str
    ) -> None:
        """Missing images/train/ raises TrainingError."""
        d = tmp_path / "no_train"
        d.mkdir()
        (d / "data.yaml").write_text("path: /tmp\ntrain: images/train\n")
        # Only create val, not train
        val_dir = d / "images" / "val"
        val_dir.mkdir(parents=True)
        (val_dir / "img.jpg").write_bytes(b"x")

        service = _make_service()
        params = _make_params(dataset_dir=str(d), output_dir=output_dir)
        with pytest.raises(TrainingError, match="images/train"):
            service._validate_local_dataset(params)

    def test_raises_when_train_images_dir_empty(
        self, tmp_path: Path, output_dir: str
    ) -> None:
        """Empty images/train/ (no image files) raises TrainingError."""
        d = tmp_path / "empty_train"
        d.mkdir()
        (d / "data.yaml").write_text("path: /tmp\ntrain: images/train\n")
        (d / "images" / "train").mkdir(parents=True)  # exists but empty
        val_dir = d / "images" / "val"
        val_dir.mkdir(parents=True)
        (val_dir / "img.jpg").write_bytes(b"x")

        service = _make_service()
        params = _make_params(dataset_dir=str(d), output_dir=output_dir)
        with pytest.raises(TrainingError, match="empty"):
            service._validate_local_dataset(params)

    def test_non_image_files_not_counted(self, tmp_path: Path, output_dir: str) -> None:
        """Only files with recognised image extensions count."""
        d = tmp_path / "non_img"
        d.mkdir()
        (d / "data.yaml").write_text("path: /tmp\ntrain: images/train\n")
        train_dir = d / "images" / "train"
        train_dir.mkdir(parents=True)
        # .txt file should not count
        (train_dir / "label.txt").write_text("0 0.5 0.5 0.1 0.1")
        val_dir = d / "images" / "val"
        val_dir.mkdir(parents=True)
        (val_dir / "img.jpg").write_bytes(b"x")

        service = _make_service()
        params = _make_params(dataset_dir=str(d), output_dir=output_dir)
        with pytest.raises(TrainingError, match="empty"):
            service._validate_local_dataset(params)


# ---------------------------------------------------------------------------
# Manifest auto-detection tests
# ---------------------------------------------------------------------------


class TestManifestAutoDetection:
    def _write_manifest(self, directory: Path, bucket: str, prefix: str) -> None:
        manifest = {
            "bucket": bucket,
            "prefix": prefix,
            "splits": {
                "train": ["images/train/img1.jpg"],
                "val": ["images/val/img1.jpg"],
                "test": [],
            },
            "total_images": 2,
        }
        (directory / "dataset_manifest.json").write_text(json.dumps(manifest))

    def test_manifest_present_switches_to_s3_mode(
        self, tmp_path: Path, output_dir: str
    ) -> None:
        """When dataset_manifest.json exists, source is overridden to 's3'."""
        d = tmp_path / "manifest_dataset"
        d.mkdir()
        (d / "data.yaml").write_text("path: /tmp\ntrain: images/train\n")
        self._write_manifest(d, bucket="my-bucket", prefix="datasets/v1/")

        service = _make_service()
        params = _make_params(
            dataset_dir=str(d),
            output_dir=output_dir,
            source="local",  # initially local
            s3_bucket=None,
            s3_prefix=None,
        )
        updated = service._apply_manifest_if_present(params)

        assert updated.source == "s3"
        assert updated.s3_bucket == "my-bucket"
        assert updated.s3_prefix == "datasets/v1/"

    def test_manifest_absent_leaves_params_unchanged(
        self, dataset_dir: str, output_dir: str
    ) -> None:
        """Without dataset_manifest.json, params are returned unchanged."""
        service = _make_service()
        params = _make_params(dataset_dir=dataset_dir, output_dir=output_dir, source="local")
        updated = service._apply_manifest_if_present(params)

        assert updated.source == "local"
        assert updated.s3_bucket is None
        assert updated.s3_prefix is None

    def test_manifest_overrides_explicit_s3_flags(
        self, tmp_path: Path, output_dir: str
    ) -> None:
        """Manifest takes precedence even when explicit s3_bucket/prefix are passed."""
        d = tmp_path / "override_dataset"
        d.mkdir()
        (d / "data.yaml").write_text("path: /tmp\ntrain: images/train\n")
        self._write_manifest(d, bucket="manifest-bucket", prefix="manifest/prefix/")

        service = _make_service()
        params = _make_params(
            dataset_dir=str(d),
            output_dir=output_dir,
            source="s3",
            s3_bucket="old-bucket",
            s3_prefix="old/prefix/",
        )
        updated = service._apply_manifest_if_present(params)

        assert updated.s3_bucket == "manifest-bucket"
        assert updated.s3_prefix == "manifest/prefix/"

    def test_malformed_manifest_falls_back_gracefully(
        self, tmp_path: Path, output_dir: str
    ) -> None:
        """Invalid JSON in manifest logs a warning and leaves params unchanged."""
        d = tmp_path / "bad_manifest"
        d.mkdir()
        (d / "data.yaml").write_text("path: /tmp\ntrain: images/train\n")
        (d / "dataset_manifest.json").write_text("{ not valid json }")

        service = _make_service()
        params = _make_params(dataset_dir=str(d), output_dir=output_dir, source="local")
        updated = service._apply_manifest_if_present(params)

        assert updated.source == "local"

    def test_manifest_missing_bucket_field_falls_back(
        self, tmp_path: Path, output_dir: str
    ) -> None:
        """Manifest without 'bucket' field is ignored."""
        d = tmp_path / "no_bucket"
        d.mkdir()
        (d / "data.yaml").write_text("path: /tmp\ntrain: images/train\n")
        (d / "dataset_manifest.json").write_text(
            json.dumps({"prefix": "some/prefix/", "splits": {}, "total_images": 0})
        )

        service = _make_service()
        params = _make_params(dataset_dir=str(d), output_dir=output_dir)
        updated = service._apply_manifest_if_present(params)

        assert updated.source == "local"

    def test_s3_validation_passes_when_manifest_supplies_credentials(
        self, tmp_path: Path, output_dir: str
    ) -> None:
        """After manifest auto-detection, s3 validation should not raise."""
        d = tmp_path / "manifest_s3"
        d.mkdir()
        (d / "data.yaml").write_text("path: /tmp\ntrain: images/train\n")
        self._write_manifest(d, bucket="my-bucket", prefix="datasets/v1/")

        service = _make_service()
        params = _make_params(
            dataset_dir=str(d),
            output_dir=output_dir,
            source="local",
            s3_bucket=None,
            s3_prefix=None,
        )
        updated = service._apply_manifest_if_present(params)
        # Validate should now pass since s3_bucket and s3_prefix are populated
        service._validate_params(updated)  # should not raise


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
        """Execute service._run_training() with all heavy deps mocked.

        The mock YOLO model captures registered callbacks via add_callback and
        fires them synchronously when train() is called so that per-epoch
        metrics (epoch_metrics dict) are populated and result fields are correct.
        """
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

        # Capture registered callbacks so we can fire them from within train()
        registered_callbacks: dict[str, Any] = {}

        def _capture_callback(event: str, fn: Any) -> None:
            registered_callbacks[event] = fn

        def _fake_train(**kwargs: Any) -> Any:
            # Fire on_fit_epoch_end so epoch_metrics gets populated
            cb = registered_callbacks.get("on_fit_epoch_end")
            if cb is not None:
                cb(mock_trainer)
            return mock_trainer

        mock_model = MagicMock()
        mock_model.add_callback.side_effect = _capture_callback
        mock_model.train.side_effect = _fake_train
        mock_model.trainer = mock_trainer

        mock_yolo_cls = MagicMock(return_value=mock_model)

        with (
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor._GPU_AVAILABLE", False),
            patch("mlflow.log_metrics"),
            patch("mlflow.active_run", return_value=MagicMock()),
        ):
            vm = MagicMock()
            vm.used = 1e9
            vm.percent = 10.0
            mock_psutil.virtual_memory.return_value = vm
            mock_psutil.cpu_percent.return_value = 5.0

            result = service._run_training(params, mock_yolo_cls)

        return result, mock_model, mock_s3

    def test_successful_run_returns_training_result(
        self, dataset_dir: str, output_dir: str, tmp_path: Path
    ) -> None:
        """A successful run returns a populated TrainingResult."""
        from app.models.training import TrainingResult

        save_dir = _fake_trainer_save_dir(tmp_path)
        result, _, _ = self._run_with_mocks(
            dataset_dir, output_dir, tmp_path, save_dir
        )

        assert isinstance(result, TrainingResult)
        assert result.experiment_name == "test-exp"
        assert result.model_variant == "yolov8n-pose.pt"
        assert result.final_map50 == 0.75
        assert result.final_map50_95 == 0.55

    def test_model_train_called_with_correct_kwargs(
        self, dataset_dir: str, output_dir: str, tmp_path: Path
    ) -> None:
        """model.train() receives the expected keyword arguments."""
        save_dir = _fake_trainer_save_dir(tmp_path)
        result, mock_model, _ = self._run_with_mocks(
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
        _, _, mock_s3 = self._run_with_mocks(
            dataset_dir, output_dir, tmp_path, save_dir
        )

        upload_calls = [str(c) for c in mock_s3.upload_file.call_args_list]
        assert any("best.pt" in c for c in upload_calls)

    def test_resume_sets_resume_flag(
        self, dataset_dir: str, output_dir: str, tmp_path: Path
    ) -> None:
        """When resume_from='auto', model.train() is called with resume=True."""
        save_dir = _fake_trainer_save_dir(tmp_path)
        _, mock_model, _ = self._run_with_mocks(
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

        with (
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor._GPU_AVAILABLE", False),
        ):
            vm = MagicMock()
            vm.used = 1e9
            vm.percent = 10.0
            mock_psutil.virtual_memory.return_value = vm
            mock_psutil.cpu_percent.return_value = 5.0

            with pytest.raises(TrainingError, match="Training failed"):
                service.run(params)

    def test_local_mode_validation_called_in_run(
        self, dataset_dir: str, output_dir: str, tmp_path: Path
    ) -> None:
        """_validate_local_dataset is called when source='local' via _run_training."""
        save_dir = _fake_trainer_save_dir(tmp_path)
        service = _make_service()
        params = _make_params(
            dataset_dir=dataset_dir, output_dir=output_dir, source="local"
        )

        with patch.object(service, "_validate_local_dataset") as mock_validate:
            mock_trainer = MagicMock()
            mock_trainer.save_dir = str(save_dir)
            mock_trainer.epoch = 1
            mock_trainer.metrics = {}
            mock_trainer.loss_items = []
            mock_trainer.last = str(save_dir / "weights" / "last.pt")
            mock_model = MagicMock()
            mock_model.train.return_value = mock_trainer
            mock_model.trainer = mock_trainer
            mock_yolo_cls = MagicMock(return_value=mock_model)

            with (
                patch("app.services.resource_monitor.psutil") as mock_psutil,
                patch("app.services.resource_monitor._GPU_AVAILABLE", False),
                patch("mlflow.log_metrics"),
                patch("mlflow.active_run", return_value=MagicMock()),
            ):
                vm = MagicMock()
                vm.used = 1e9
                vm.percent = 10.0
                mock_psutil.virtual_memory.return_value = vm
                mock_psutil.cpu_percent.return_value = 5.0

                service._run_training(params, mock_yolo_cls)

        mock_validate.assert_called_once_with(params)

    def test_local_validation_skipped_in_s3_mode(
        self, tmp_path: Path, output_dir: str
    ) -> None:
        """_validate_local_dataset is NOT called when source='s3'."""
        d = tmp_path / "s3_dataset"
        d.mkdir()
        (d / "data.yaml").write_text("path: /tmp\ntrain: images/train\n")
        save_dir = _fake_trainer_save_dir(tmp_path)

        service = _make_service()
        params = _make_params(
            dataset_dir=str(d),
            output_dir=output_dir,
            source="s3",
            s3_bucket="my-bucket",
            s3_prefix="datasets/v1/",
        )

        with patch.object(service, "_validate_local_dataset") as mock_validate:
            mock_trainer = MagicMock()
            mock_trainer.save_dir = str(save_dir)
            mock_trainer.epoch = 1
            mock_trainer.metrics = {}
            mock_trainer.loss_items = []
            mock_trainer.last = str(save_dir / "weights" / "last.pt")
            mock_model = MagicMock()
            mock_model.train.return_value = mock_trainer
            mock_model.trainer = mock_trainer
            mock_yolo_cls = MagicMock(return_value=mock_model)

            with (
                patch("app.services.resource_monitor.psutil") as mock_psutil,
                patch("app.services.resource_monitor._GPU_AVAILABLE", False),
                patch("mlflow.log_metrics"),
                patch("mlflow.active_run", return_value=MagicMock()),
                patch("app.services.s3_pose_trainer.make_s3_pose_trainer"),
            ):
                vm = MagicMock()
                vm.used = 1e9
                vm.percent = 10.0
                mock_psutil.virtual_memory.return_value = vm
                mock_psutil.cpu_percent.return_value = 5.0

                service._run_training(params, mock_yolo_cls)

        mock_validate.assert_not_called()

    def test_system_metrics_logged_at_epoch_end(
        self, dataset_dir: str, output_dir: str, tmp_path: Path
    ) -> None:
        """on_fit_epoch_end calls mlflow.log_metrics with system metric keys."""
        mock_s3 = MagicMock()
        service = _make_service(s3_client=mock_s3)
        params = _make_params(dataset_dir=dataset_dir, output_dir=output_dir)
        save_dir = _fake_trainer_save_dir(tmp_path)

        mock_trainer = MagicMock()
        mock_trainer.save_dir = str(save_dir)
        mock_trainer.epoch = 0
        mock_trainer.metrics = {
            "metrics/mAP50(B)": 0.6,
            "metrics/mAP50-95(B)": 0.4,
            "metrics/precision(B)": 0.7,
            "metrics/recall(B)": 0.65,
        }
        mock_trainer.loss_items = []
        mock_trainer.last = str(save_dir / "weights" / "last.pt")

        registered_callbacks: dict[str, Any] = {}

        def _capture(event: str, fn: Any) -> None:
            registered_callbacks[event] = fn

        def _fake_train(**kwargs: Any) -> Any:
            cb = registered_callbacks.get("on_fit_epoch_end")
            if cb:
                cb(mock_trainer)
            return mock_trainer

        mock_model = MagicMock()
        mock_model.add_callback.side_effect = _capture
        mock_model.train.side_effect = _fake_train
        mock_model.trainer = mock_trainer
        mock_yolo_cls = MagicMock(return_value=mock_model)

        with (
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor._GPU_AVAILABLE", False),
            patch("mlflow.log_metrics") as mock_log_metrics,
            patch("mlflow.active_run", return_value=MagicMock()),
        ):
            vm = MagicMock()
            vm.used = 2e9
            vm.percent = 15.0
            mock_psutil.virtual_memory.return_value = vm
            mock_psutil.cpu_percent.return_value = 30.0

            service._run_training(params, mock_yolo_cls)

        # mlflow.log_metrics must have been called at least once
        assert mock_log_metrics.called

        # Collect all keys that were ever logged
        all_logged: dict[str, float] = {}
        for c in mock_log_metrics.call_args_list:
            metrics_arg: dict[str, float] = c.args[0] if c.args else c.kwargs.get("metrics", {})
            all_logged.update(metrics_arg)

        # System metrics must be present in the logged dict
        assert "system/ram_used_gb" in all_logged
        assert "system/ram_percent" in all_logged
        assert "system/cpu_percent" in all_logged
        # Val metrics must also be present
        assert "val/mAP50" in all_logged


# ---------------------------------------------------------------------------
# Upload final weights
# ---------------------------------------------------------------------------


class TestUploadFinalWeights:
    def test_returns_empty_string_when_best_pt_missing(
        self, dataset_dir: str, output_dir: str, tmp_path: Path
    ) -> None:
        """When best.pt doesn't exist, returns empty string (not a phantom URI)."""
        save_dir = tmp_path / "runs" / "no-best"
        weights_dir = save_dir / "weights"
        weights_dir.mkdir(parents=True)
        # Only last.pt, no best.pt
        (weights_dir / "last.pt").write_bytes(b"fake-last")

        service = _make_service()
        params = _make_params(dataset_dir=dataset_dir, output_dir=output_dir)
        result = service._upload_final_weights(params, save_dir)

        assert result == ""

    def test_returns_s3_uri_when_best_pt_exists(
        self, dataset_dir: str, output_dir: str, tmp_path: Path
    ) -> None:
        """When best.pt exists, returns the S3 URI."""
        save_dir = _fake_trainer_save_dir(tmp_path)
        service = _make_service()
        params = _make_params(dataset_dir=dataset_dir, output_dir=output_dir)
        result = service._upload_final_weights(params, save_dir)

        assert result.startswith("s3://")
        assert "best.pt" in result


# ---------------------------------------------------------------------------
# train() kwargs
# ---------------------------------------------------------------------------


class TestBuildTrainKwargs:
    def test_contains_core_training_params(
        self, dataset_dir: str, output_dir: str
    ) -> None:
        """Essential training params are forwarded to model.train()."""
        service = _make_service()
        params = _make_params(dataset_dir=dataset_dir, output_dir=output_dir)
        kwargs = service._build_train_kwargs(params, "/tmp/data.yaml")

        assert kwargs["data"] == "/tmp/data.yaml"
        assert kwargs["epochs"] == 2
        assert kwargs["batch"] == 2
        assert kwargs["lr0"] == 0.01
        assert kwargs["optimizer"] == "SGD"

    def test_resume_flag_set_when_resume_from(
        self, dataset_dir: str, output_dir: str
    ) -> None:
        """resume=True is in kwargs when resume_from is set."""
        service = _make_service()
        params = _make_params(
            dataset_dir=dataset_dir, output_dir=output_dir, resume_from="auto"
        )
        kwargs = service._build_train_kwargs(params, "/tmp/data.yaml")

        assert kwargs["resume"] is True

    def test_no_resume_flag_without_resume_from(
        self, dataset_dir: str, output_dir: str
    ) -> None:
        """resume is absent from kwargs when resume_from is not set."""
        service = _make_service()
        params = _make_params(dataset_dir=dataset_dir, output_dir=output_dir)
        kwargs = service._build_train_kwargs(params, "/tmp/data.yaml")

        assert "resume" not in kwargs

    def test_augmentation_params_forwarded(
        self, dataset_dir: str, output_dir: str
    ) -> None:
        """Augmentation params are included in train kwargs."""
        service = _make_service()
        params = _make_params(dataset_dir=dataset_dir, output_dir=output_dir)
        kwargs = service._build_train_kwargs(params, "/tmp/data.yaml")

        assert "hsv_h" in kwargs
        assert "mosaic" in kwargs
        assert "fliplr" in kwargs


# ---------------------------------------------------------------------------
# S3 checkpoint upload callback
# ---------------------------------------------------------------------------


class TestCheckpointUploadCallback:
    def test_uploads_at_checkpoint_interval(
        self, dataset_dir: str, output_dir: str, tmp_path: Path
    ) -> None:
        """on_train_epoch_end uploads checkpoint when epoch % interval == 0."""
        mock_s3 = MagicMock()
        service = _make_service(s3_client=mock_s3)
        save_dir = _fake_trainer_save_dir(tmp_path)

        params = _make_params(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            checkpoint_interval=2,
        )

        registered_callbacks: dict[str, Any] = {}

        def _capture(event: str, fn: Any) -> None:
            registered_callbacks[event] = fn

        mock_trainer = MagicMock()
        mock_trainer.save_dir = str(save_dir)
        mock_trainer.epoch = 1  # 0-indexed, so epoch 2
        mock_trainer.last = str(save_dir / "weights" / "last.pt")
        mock_trainer.metrics = {}

        def _fake_train(**kwargs: Any) -> Any:
            cb = registered_callbacks.get("on_train_epoch_end")
            if cb:
                cb(mock_trainer)
            return mock_trainer

        mock_model = MagicMock()
        mock_model.add_callback.side_effect = _capture
        mock_model.train.side_effect = _fake_train
        mock_model.trainer = mock_trainer
        mock_yolo_cls = MagicMock(return_value=mock_model)

        with (
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor._GPU_AVAILABLE", False),
            patch("mlflow.log_metrics"),
            patch("mlflow.active_run", return_value=MagicMock()),
        ):
            vm = MagicMock()
            vm.used = 1e9
            vm.percent = 10.0
            mock_psutil.virtual_memory.return_value = vm
            mock_psutil.cpu_percent.return_value = 5.0

            service._run_training(params, mock_yolo_cls)

        # Should have uploaded checkpoint (epoch 2 % 2 == 0) plus final weights
        upload_calls = [str(c) for c in mock_s3.upload_file.call_args_list]
        assert any("epoch_0002.pt" in c for c in upload_calls)

    def test_skips_non_checkpoint_epoch(
        self, dataset_dir: str, output_dir: str, tmp_path: Path
    ) -> None:
        """on_train_epoch_end does NOT upload when epoch % interval != 0."""
        mock_s3 = MagicMock()
        service = _make_service(s3_client=mock_s3)
        save_dir = _fake_trainer_save_dir(tmp_path)

        params = _make_params(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            checkpoint_interval=5,
        )

        registered_callbacks: dict[str, Any] = {}

        def _capture(event: str, fn: Any) -> None:
            registered_callbacks[event] = fn

        mock_trainer = MagicMock()
        mock_trainer.save_dir = str(save_dir)
        mock_trainer.epoch = 1  # 0-indexed, so epoch 2 (not divisible by 5)
        mock_trainer.last = str(save_dir / "weights" / "last.pt")
        mock_trainer.metrics = {}

        def _fake_train(**kwargs: Any) -> Any:
            cb = registered_callbacks.get("on_train_epoch_end")
            if cb:
                cb(mock_trainer)
            return mock_trainer

        mock_model = MagicMock()
        mock_model.add_callback.side_effect = _capture
        mock_model.train.side_effect = _fake_train
        mock_model.trainer = mock_trainer
        mock_yolo_cls = MagicMock(return_value=mock_model)

        with (
            patch("app.services.resource_monitor.psutil") as mock_psutil,
            patch("app.services.resource_monitor._GPU_AVAILABLE", False),
            patch("mlflow.log_metrics"),
            patch("mlflow.active_run", return_value=MagicMock()),
        ):
            vm = MagicMock()
            vm.used = 1e9
            vm.percent = 10.0
            mock_psutil.virtual_memory.return_value = vm
            mock_psutil.cpu_percent.return_value = 5.0

            service._run_training(params, mock_yolo_cls)

        # Only final weight uploads (best.pt + last.pt), no intermediate checkpoint
        upload_calls = [str(c) for c in mock_s3.upload_file.call_args_list]
        assert not any("epoch_" in c for c in upload_calls)
