"""Unit tests for the run_state.json schema + atomic S3 writer (F-03).

No real S3 calls — the boto3 client is mocked.
"""

import hashlib
import io
import json
from pathlib import Path
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from app.services.run_state import (
    SCHEMA_VERSION,
    CheckpointEntry,
    ResumeInfo,
    RunState,
    RunStateStore,
    compute_sha256,
    now_rfc3339,
    run_state_key,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_state() -> RunState:
    return RunState(
        experiment_name="exp-1",
        mlflow_run_id="run-123",
        last_completed_epoch=10,
        checkpoints=[
            CheckpointEntry(
                epoch=10,
                uri="s3://bucket/checkpoints/exp-1/epoch_0010.pt",
                sha256="abc123",
                ultralytics_version="8.3.0",
            )
        ],
        dataset_manifest_sha256="commit-deadbeef",
        config_hash="cfg-hash",
        source_workflow_uid="wf-uid",
        source_workflow_name="wf-name",
        heartbeat=now_rfc3339(),
        resume=ResumeInfo(resumed_from="orig-run", attempt=2),
    )


# ---------------------------------------------------------------------------
# compute_sha256
# ---------------------------------------------------------------------------


class TestComputeSha256:
    def test_matches_hashlib(self, tmp_path: Path) -> None:
        f = tmp_path / "ckpt.pt"
        payload = b"some-fake-checkpoint-bytes" * 1000
        f.write_bytes(payload)
        assert compute_sha256(f) == hashlib.sha256(payload).hexdigest()

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.pt"
        f.write_bytes(b"")
        assert compute_sha256(f) == hashlib.sha256(b"").hexdigest()


# ---------------------------------------------------------------------------
# run_state_key
# ---------------------------------------------------------------------------


class TestRunStateKey:
    def test_basic(self) -> None:
        assert (
            run_state_key("checkpoints", "exp-1") == "checkpoints/exp-1/run_state.json"
        )

    def test_strips_slashes(self) -> None:
        assert (
            run_state_key("/checkpoints/", "exp-1")
            == "checkpoints/exp-1/run_state.json"
        )


# ---------------------------------------------------------------------------
# Schema round-trip + contract
# ---------------------------------------------------------------------------


class TestSchemaRoundTrip:
    def test_round_trip_preserves_fields(self) -> None:
        state = _sample_state()
        restored = RunState.model_validate_json(state.model_dump_json())
        assert restored == state

    def test_contract_field_names_present(self) -> None:
        """The exact keys F-02/F-06 (operator side) parse must be serialised."""
        doc = json.loads(_sample_state().model_dump_json())
        for key in (
            "schema_version",
            "experiment_name",
            "mlflow_run_id",
            "last_completed_epoch",
            "checkpoints",
            "dataset_manifest_sha256",
            "config_hash",
            "source_workflow_uid",
            "source_workflow_name",
            "heartbeat",
            "resume",
        ):
            assert key in doc, f"missing contract field: {key}"
        assert doc["schema_version"] == SCHEMA_VERSION
        ckpt = doc["checkpoints"][0]
        for key in ("epoch", "uri", "sha256", "ultralytics_version"):
            assert key in ckpt
        assert set(doc["resume"]) == {"resumed_from", "attempt"}

    def test_fresh_state_has_no_resume_block(self) -> None:
        doc = json.loads(
            RunState(experiment_name="x", mlflow_run_id="r").model_dump_json()
        )
        assert doc["resume"] is None


# ---------------------------------------------------------------------------
# RunStateStore
# ---------------------------------------------------------------------------


class TestRunStateStoreWrite:
    def test_write_is_single_put_object(self) -> None:
        s3 = MagicMock()
        store = RunStateStore(s3, "bucket", "checkpoints/exp-1/run_state.json")
        state = _sample_state()

        store.write(state)

        s3.put_object.assert_called_once()
        kwargs = s3.put_object.call_args.kwargs
        assert kwargs["Bucket"] == "bucket"
        assert kwargs["Key"] == "checkpoints/exp-1/run_state.json"
        # Body must be the full, valid JSON manifest (atomic single object).
        restored = RunState.model_validate_json(kwargs["Body"])
        assert restored == state

    def test_uri(self) -> None:
        store = RunStateStore(MagicMock(), "bucket", "checkpoints/exp-1/run_state.json")
        assert store.uri == "s3://bucket/checkpoints/exp-1/run_state.json"


class TestRunStateStoreRead:
    def test_read_returns_state(self) -> None:
        state = _sample_state()
        s3 = MagicMock()
        s3.get_object.return_value = {
            "Body": io.BytesIO(state.model_dump_json().encode("utf-8"))
        }
        store = RunStateStore(s3, "bucket", "key")

        assert store.read() == state

    def test_read_missing_returns_none(self) -> None:
        s3 = MagicMock()
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )
        store = RunStateStore(s3, "bucket", "key")

        assert store.read() is None

    def test_read_other_error_propagates(self) -> None:
        s3 = MagicMock()
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "GetObject"
        )
        store = RunStateStore(s3, "bucket", "key")

        try:
            store.read()
        except ClientError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected ClientError to propagate")


# ---------------------------------------------------------------------------
# now_rfc3339
# ---------------------------------------------------------------------------


def test_now_rfc3339_is_parseable() -> None:
    from datetime import datetime

    ts = now_rfc3339()
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None
