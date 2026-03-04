"""Tests for the Manager class."""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, call, patch

import pytest

from app.manager import Manager
from app.models.config import Config
from app.models.registration import RegistrationResult

TRACKING_URI = "http://mlflow.test.local:5000"
MODEL_NAME = "spacecraft-pose-yolo"
RUN_ID = "run123"
BEST_S3 = "s3://io-mlops/checkpoints/exp/best.pt"
LAST_S3 = "s3://io-mlops/checkpoints/exp/last.pt"


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", TRACKING_URI)
    return Config()


def _make_result(
    best_version: int = 1,
    last_version: Optional[int] = None,
    promoted_to: Optional[str] = None,
) -> RegistrationResult:
    return RegistrationResult(
        registered_model_name=MODEL_NAME,
        best_version=best_version,
        last_version=last_version,
        registered_at=datetime.now(timezone.utc),
        promoted_to=promoted_to,
    )


class TestManagerRun:
    def test_delegates_to_service(self, config: Config) -> None:
        with patch("app.manager.ModelRegistrationService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.run.return_value = _make_result(best_version=1)

            manager = Manager(config=config)
            result = manager.run(
                mlflow_run_id=RUN_ID,
                best_checkpoint_path=BEST_S3,
            )

        mock_service.run.assert_called_once()
        assert result.best_version == 1

    def test_passes_all_lineage_params(self, config: Config) -> None:
        with patch("app.manager.ModelRegistrationService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.run.return_value = _make_result(best_version=2, last_version=3)

            manager = Manager(config=config)
            manager.run(
                mlflow_run_id=RUN_ID,
                best_checkpoint_path=BEST_S3,
                last_checkpoint_path=LAST_S3,
                dataset_version="v1",
                dataset_sample_size=5000,
                config_hash="abc123",
                git_commit="def456",
                model_variant="yolov8n-pose.pt",
                best_map50=0.85,
            )

        called_params = mock_service.run.call_args.kwargs["params"]
        assert called_params.dataset_version == "v1"
        assert called_params.dataset_sample_size == 5000
        assert called_params.config_hash == "abc123"
        assert called_params.git_commit == "def456"
        assert called_params.model_variant == "yolov8n-pose.pt"
        assert called_params.best_map50 == pytest.approx(0.85)

    def test_uses_config_registered_model_name_by_default(self, config: Config) -> None:
        with patch("app.manager.ModelRegistrationService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.run.return_value = _make_result()

            manager = Manager(config=config)
            manager.run(
                mlflow_run_id=RUN_ID,
                best_checkpoint_path=BEST_S3,
            )

        called_params = mock_service.run.call_args.kwargs["params"]
        assert called_params.registered_model_name == config.registered_model_name

    def test_cli_override_of_registered_model_name(self, config: Config) -> None:
        with patch("app.manager.ModelRegistrationService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.run.return_value = _make_result()

            manager = Manager(config=config)
            manager.run(
                mlflow_run_id=RUN_ID,
                best_checkpoint_path=BEST_S3,
                registered_model_name="override-model",
            )

        called_params = mock_service.run.call_args.kwargs["params"]
        assert called_params.registered_model_name == "override-model"

    def test_passes_promote_to(self, config: Config) -> None:
        with patch("app.manager.ModelRegistrationService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.run.return_value = _make_result(promoted_to="Staging")

            manager = Manager(config=config)
            result = manager.run(
                mlflow_run_id=RUN_ID,
                best_checkpoint_path=BEST_S3,
                promote_to="Staging",
            )

        called_params = mock_service.run.call_args.kwargs["params"]
        assert called_params.promote_to == "Staging"
        assert result.promoted_to == "Staging"

    def test_constructs_service_with_tracking_uri_from_config(
        self, config: Config
    ) -> None:
        with patch("app.manager.ModelRegistrationService") as mock_service_cls:
            mock_service_cls.return_value.run.return_value = _make_result()

            Manager(config=config)

        mock_service_cls.assert_called_once_with(
            mlflow_tracking_uri=TRACKING_URI,
            max_retries=config.max_retries,
        )


class TestManagerLogger:
    def test_setup_logging_called_on_init(self, config: Config) -> None:
        with (
            patch("app.manager.ModelRegistrationService"),
            patch("app.manager.setup_logging") as mock_setup,
        ):
            Manager(config=config)

        mock_setup.assert_called_once_with(level=config.log_level)

    def test_setup_logging_uses_config_log_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", TRACKING_URI)
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        config = Config()

        with (
            patch("app.manager.ModelRegistrationService"),
            patch("app.manager.setup_logging") as mock_setup,
        ):
            Manager(config=config)

        mock_setup.assert_called_once_with(level="DEBUG")


class TestManagerCleanup:
    def test_cleanup_called_on_success(self, config: Config) -> None:
        with patch("app.manager.ModelRegistrationService") as mock_service_cls:
            mock_service_cls.return_value.run.return_value = _make_result()

            manager = Manager(config=config)
            with patch.object(manager, "_cleanup") as mock_cleanup:
                manager.run(
                    mlflow_run_id=RUN_ID,
                    best_checkpoint_path=BEST_S3,
                    last_checkpoint_path=LAST_S3,
                )

        mock_cleanup.assert_called_once_with(
            best_checkpoint_path=BEST_S3,
            last_checkpoint_path=LAST_S3,
        )

    def test_cleanup_called_on_service_failure(self, config: Config) -> None:
        """Cleanup must run even when the service raises an exception."""
        with patch("app.manager.ModelRegistrationService") as mock_service_cls:
            mock_service_cls.return_value.run.side_effect = RuntimeError("boom")

            manager = Manager(config=config)
            with patch.object(manager, "_cleanup") as mock_cleanup:
                with pytest.raises(RuntimeError, match="boom"):
                    manager.run(
                        mlflow_run_id=RUN_ID,
                        best_checkpoint_path=BEST_S3,
                    )

        mock_cleanup.assert_called_once()

    def test_cleanup_skips_s3_uris(self, config: Config) -> None:
        """S3 URIs must never be passed to the filesystem deletion logic."""
        manager = Manager.__new__(Manager)
        manager._config = config
        manager._logger = MagicMock()

        with patch("app.manager.shutil.rmtree") as mock_rmtree:
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.unlink") as mock_unlink:
                    manager._cleanup(
                        best_checkpoint_path=BEST_S3,
                        last_checkpoint_path=LAST_S3,
                    )

        mock_rmtree.assert_not_called()
        mock_unlink.assert_not_called()

    def test_cleanup_removes_local_file(self, config: Config) -> None:
        """A local file path must be unlinked during cleanup."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            local_path = tmp.name

        try:
            manager = Manager.__new__(Manager)
            manager._config = config
            manager._logger = MagicMock()

            manager._cleanup(
                best_checkpoint_path=local_path,
                last_checkpoint_path=None,
            )

            assert not Path(local_path).exists()
        finally:
            # Belt-and-suspenders: remove the file if the test failed
            if Path(local_path).exists():
                os.unlink(local_path)

    def test_cleanup_removes_local_directory(self, config: Config) -> None:
        """A local directory path must be removed recursively during cleanup."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a nested file so rmtree is actually exercised
            nested = Path(tmp_dir) / "weights" / "best.pt"
            nested.parent.mkdir()
            nested.write_bytes(b"fake")

            manager = Manager.__new__(Manager)
            manager._config = config
            manager._logger = MagicMock()

            manager._cleanup(
                best_checkpoint_path=tmp_dir,
                last_checkpoint_path=None,
            )

            assert not Path(tmp_dir).exists()

    def test_cleanup_tolerates_missing_path(self, config: Config) -> None:
        """Cleanup must not raise when the path does not exist."""
        manager = Manager.__new__(Manager)
        manager._config = config
        manager._logger = MagicMock()

        # Should complete without raising
        manager._cleanup(
            best_checkpoint_path="/nonexistent/path/best.pt",
            last_checkpoint_path=None,
        )

    def test_cleanup_tolerates_os_error(self, config: Config) -> None:
        """An OSError during deletion must be logged, not re-raised."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            local_path = tmp.name

        try:
            manager = Manager.__new__(Manager)
            manager._config = config
            manager._logger = MagicMock()

            with patch("pathlib.Path.unlink", side_effect=OSError("permission denied")):
                # Must not raise
                manager._cleanup(
                    best_checkpoint_path=local_path,
                    last_checkpoint_path=None,
                )

            manager._logger.error.assert_called_once()
        finally:
            if Path(local_path).exists():
                os.unlink(local_path)


class TestManagerFreeGpuMemory:
    def test_calls_empty_cache_when_cuda_available(self, config: Config) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        manager = Manager.__new__(Manager)
        manager._config = config
        manager._logger = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            manager._free_gpu_memory()

        mock_torch.cuda.empty_cache.assert_called_once()

    def test_skips_empty_cache_when_cuda_unavailable(self, config: Config) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        manager = Manager.__new__(Manager)
        manager._config = config
        manager._logger = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            manager._free_gpu_memory()

        mock_torch.cuda.empty_cache.assert_not_called()

    def test_no_error_when_torch_not_installed(self, config: Config) -> None:
        manager = Manager.__new__(Manager)
        manager._config = config
        manager._logger = MagicMock()

        with patch.dict("sys.modules", {"torch": None}):
            # ImportError path — must not raise
            manager._free_gpu_memory()
