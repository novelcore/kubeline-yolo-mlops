"""Tests for run_state / MLflow-lineage wiring inside TrainingService (F-03/F-04).

Drives the callbacks directly with a fake trainer and a mocked S3 client /
MLflow so no real training, S3, or tracking server is touched.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from app.models.training import AugmentationParams, TrainingParams
from app.services.model_training import TrainingService
from app.services.run_state import CheckpointEntry, RunState, now_rfc3339

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(s3_client: Any) -> TrainingService:
    return TrainingService(
        s3_client=s3_client, mlflow_tracking_uri="http://localhost:5000"
    )


def _params(tmp_path: Path, **overrides: Any) -> TrainingParams:
    defaults: dict[str, Any] = dict(
        model_variant="yolov8n-pose.pt",
        experiment_name="exp-1",
        dataset_dir=str(tmp_path),
        output_dir=str(tmp_path / "runs"),
        source="local",
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
        checkpoint_bucket="ckpt-bucket",
        checkpoint_prefix="checkpoints",
        augmentation=AugmentationParams(),
    )
    defaults.update(overrides)
    return TrainingParams(**defaults)


def _written_state(s3: MagicMock) -> RunState:
    """Return the RunState from the most recent put_object call."""
    body = s3.put_object.call_args.kwargs["Body"]
    return RunState.model_validate_json(body)


# ---------------------------------------------------------------------------
# Start callback (F-03 write-at-start + F-04 run-id-at-start)
# ---------------------------------------------------------------------------


class TestStartCallback:
    def test_fresh_run_writes_state_with_run_id(self, tmp_path: Path) -> None:
        s3 = MagicMock()
        service = _make_service(s3)
        params = _params(tmp_path)
        store = service._build_run_state_store(params)
        holder: dict[str, Any] = {"state": None}
        cb = service._make_run_state_start_callback(params, store, holder)

        active = MagicMock()
        active.info.run_id = "run-abc"
        with patch("mlflow.active_run", return_value=active):
            cb(MagicMock())

        # run_state written at start with the active run id
        s3.put_object.assert_called_once()
        written = _written_state(s3)
        assert written.mlflow_run_id == "run-abc"
        assert written.experiment_name == "exp-1"
        assert written.last_completed_epoch == 0
        assert written.resume is None
        assert written.heartbeat  # non-empty RFC3339 timestamp
        # fresh run does NOT read an existing manifest
        s3.get_object.assert_not_called()
        assert holder["state"] is written or holder["state"].mlflow_run_id == "run-abc"

    def test_dataset_identity_falls_back_to_lakefs_commit(self, tmp_path: Path) -> None:
        s3 = MagicMock()
        service = _make_service(s3)
        params = _params(tmp_path, lakefs_commit="commit-xyz")
        store = service._build_run_state_store(params)
        holder: dict[str, Any] = {"state": None}
        cb = service._make_run_state_start_callback(params, store, holder)

        with patch("mlflow.active_run", return_value=None):
            cb(MagicMock())

        assert _written_state(s3).dataset_manifest_sha256 == "commit-xyz"

    def test_manifest_hash_preferred_over_lakefs_commit(self, tmp_path: Path) -> None:
        s3 = MagicMock()
        service = _make_service(s3)
        params = _params(
            tmp_path,
            lakefs_commit="commit-xyz",
            dataset_manifest_sha256="manifest-hash",
        )
        store = service._build_run_state_store(params)
        holder: dict[str, Any] = {"state": None}
        cb = service._make_run_state_start_callback(params, store, holder)

        with patch("mlflow.active_run", return_value=None):
            cb(MagicMock())

        assert _written_state(s3).dataset_manifest_sha256 == "manifest-hash"

    def test_workflow_identity_from_env(self, tmp_path: Path) -> None:
        s3 = MagicMock()
        service = _make_service(s3)
        params = _params(tmp_path)
        store = service._build_run_state_store(params)
        holder: dict[str, Any] = {"state": None}
        cb = service._make_run_state_start_callback(params, store, holder)

        env = {"ARGO_WORKFLOW_UID": "uid-1", "ARGO_WORKFLOW_NAME": "wf-1"}
        with (
            patch("mlflow.active_run", return_value=None),
            patch.dict("os.environ", env, clear=False),
        ):
            cb(MagicMock())

        written = _written_state(s3)
        assert written.source_workflow_uid == "uid-1"
        assert written.source_workflow_name == "wf-1"

    def test_write_failure_is_non_fatal(self, tmp_path: Path) -> None:
        s3 = MagicMock()
        s3.put_object.side_effect = RuntimeError("s3 down")
        service = _make_service(s3)
        params = _params(tmp_path)
        store = service._build_run_state_store(params)
        holder: dict[str, Any] = {"state": None}
        cb = service._make_run_state_start_callback(params, store, holder)

        with patch("mlflow.active_run", return_value=None):
            cb(MagicMock())  # must not raise


class TestResumeStartCallback:
    def test_resume_carries_lineage_and_checkpoints(self, tmp_path: Path) -> None:
        import io

        existing = RunState(
            experiment_name="exp-1",
            mlflow_run_id="orig-run",
            last_completed_epoch=10,
            checkpoints=[
                CheckpointEntry(
                    epoch=10,
                    uri="s3://ckpt-bucket/checkpoints/exp-1/epoch_0010.pt",
                    sha256="hash10",
                )
            ],
            heartbeat=now_rfc3339(),
        )
        s3 = MagicMock()
        s3.get_object.return_value = {
            "Body": io.BytesIO(existing.model_dump_json().encode("utf-8"))
        }
        service = _make_service(s3)
        params = _params(tmp_path, resume_from="auto")
        store = service._build_run_state_store(params)
        holder: dict[str, Any] = {"state": None}
        cb = service._make_run_state_start_callback(params, store, holder)

        active = MagicMock()
        active.info.run_id = "new-run"
        with (
            patch("mlflow.active_run", return_value=active),
            patch("mlflow.set_tag") as set_tag,
        ):
            cb(MagicMock())

        written = _written_state(s3)
        assert written.mlflow_run_id == "new-run"
        assert written.last_completed_epoch == 10
        assert len(written.checkpoints) == 1
        assert written.resume is not None
        assert written.resume.resumed_from == "orig-run"
        assert written.resume.attempt == 1
        # lineage tags set on the new MLflow run
        tagged = {c.args[0]: c.args[1] for c in set_tag.call_args_list}
        assert tagged["resumed_from"] == "orig-run"
        assert tagged["resume.attempt"] == "1"

    def test_resume_attempt_increments(self, tmp_path: Path) -> None:
        import io

        from app.services.run_state import ResumeInfo

        existing = RunState(
            experiment_name="exp-1",
            mlflow_run_id="run-2",
            last_completed_epoch=20,
            resume=ResumeInfo(resumed_from="run-1", attempt=1),
            heartbeat=now_rfc3339(),
        )
        s3 = MagicMock()
        s3.get_object.return_value = {
            "Body": io.BytesIO(existing.model_dump_json().encode("utf-8"))
        }
        service = _make_service(s3)
        params = _params(tmp_path, resume_from="auto")
        store = service._build_run_state_store(params)
        holder: dict[str, Any] = {"state": None}
        cb = service._make_run_state_start_callback(params, store, holder)

        with patch("mlflow.active_run", return_value=None):
            cb(MagicMock())

        assert _written_state(s3).resume.attempt == 2


# ---------------------------------------------------------------------------
# Checkpoint callback (F-03 record checkpoint)
# ---------------------------------------------------------------------------


class TestCheckpointRecording:
    def test_checkpoint_appended_and_epoch_advanced(self, tmp_path: Path) -> None:
        s3 = MagicMock()
        service = _make_service(s3)
        params = _params(tmp_path, checkpoint_interval=1)
        store = service._build_run_state_store(params)

        ckpt = tmp_path / "last.pt"
        ckpt.write_bytes(b"weights-bytes")
        from app.services.run_state import compute_sha256

        expected_sha = compute_sha256(ckpt)

        holder: dict[str, Any] = {
            "state": RunState(experiment_name="exp-1", mlflow_run_id="run-abc")
        }
        cb = service._make_checkpoint_callback(params, store, holder)

        trainer = MagicMock()
        trainer.epoch = 0  # 0-indexed -> epoch 1
        trainer.last = str(ckpt)
        with patch("mlflow.active_run", return_value=None):
            cb(trainer)

        state = holder["state"]
        assert len(state.checkpoints) == 1
        entry = state.checkpoints[0]
        assert entry.epoch == 1
        assert entry.sha256 == expected_sha
        assert entry.uri == "s3://ckpt-bucket/checkpoints/exp-1/epoch_0001.pt"
        assert state.last_completed_epoch == 1
        # checkpoint .pt uploaded AND run_state written
        s3.upload_file.assert_called_once()
        s3.put_object.assert_called_once()

    def test_no_record_when_state_absent(self, tmp_path: Path) -> None:
        s3 = MagicMock()
        service = _make_service(s3)
        params = _params(tmp_path, checkpoint_interval=1)
        store = service._build_run_state_store(params)

        ckpt = tmp_path / "last.pt"
        ckpt.write_bytes(b"w")
        holder: dict[str, Any] = {"state": None}
        cb = service._make_checkpoint_callback(params, store, holder)

        trainer = MagicMock()
        trainer.epoch = 0
        trainer.last = str(ckpt)
        with patch("mlflow.active_run", return_value=None):
            cb(trainer)

        # .pt still uploaded, but no run_state write without a live state
        s3.upload_file.assert_called_once()
        s3.put_object.assert_not_called()

    def test_skips_off_interval_epochs(self, tmp_path: Path) -> None:
        s3 = MagicMock()
        service = _make_service(s3)
        params = _params(tmp_path, checkpoint_interval=10)
        store = service._build_run_state_store(params)

        holder: dict[str, Any] = {
            "state": RunState(experiment_name="exp-1", mlflow_run_id="r")
        }
        cb = service._make_checkpoint_callback(params, store, holder)

        trainer = MagicMock()
        trainer.epoch = 2  # epoch 3, not a multiple of 10
        cb(trainer)

        s3.upload_file.assert_not_called()
        s3.put_object.assert_not_called()


# ---------------------------------------------------------------------------
# Heartbeat (T-05) via epoch-end callback
# ---------------------------------------------------------------------------


class TestHeartbeat:
    def test_epoch_end_refreshes_heartbeat(self, tmp_path: Path) -> None:
        s3 = MagicMock()
        service = _make_service(s3)
        store = service._build_run_state_store(_params(tmp_path))

        state = RunState(
            experiment_name="exp-1",
            mlflow_run_id="r",
            heartbeat="2000-01-01T00:00:00+00:00",
        )
        holder: dict[str, Any] = {"state": state}
        monitor = MagicMock()
        monitor.collect.return_value = {}
        cb = service._make_epoch_end_callback({}, monitor, store, holder)

        trainer = MagicMock()
        trainer.epoch = 0
        trainer.metrics = {}
        with patch("mlflow.active_run", return_value=None):
            cb(trainer)

        assert holder["state"].heartbeat != "2000-01-01T00:00:00+00:00"
        s3.put_object.assert_called_once()

    def test_epoch_end_no_state_no_write(self, tmp_path: Path) -> None:
        s3 = MagicMock()
        service = _make_service(s3)
        store = service._build_run_state_store(_params(tmp_path))
        monitor = MagicMock()
        monitor.collect.return_value = {}
        cb = service._make_epoch_end_callback({}, monitor, store, {"state": None})

        trainer = MagicMock()
        trainer.epoch = 0
        trainer.metrics = {}
        with patch("mlflow.active_run", return_value=None):
            cb(trainer)

        s3.put_object.assert_not_called()
