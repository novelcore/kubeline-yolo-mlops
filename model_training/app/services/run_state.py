"""Run-state manifest: schema + atomic S3 writer (F-03, PRD #9 D-02).

``run_state.json`` is the single source of truth both resume paths read
(auto-resume on Argo retry, and explicit operator resume). It lives next to
the checkpoints at::

    s3://{checkpoint_bucket}/{checkpoint_prefix}/{experiment_name}/run_state.json

and is written by the training service at run start (so the MLflow run ID is
known *before* completion) and updated at every checkpoint interval.

Atomicity (CON-04)
------------------
On S3 / the lakeFS S3-compatible gateway a single ``put_object`` is atomic at
the object level: a concurrent reader observes either the previous complete
object or the new complete object, never a partial body. We therefore serialise
the whole manifest and write it in one ``put_object`` call rather than doing a
write-then-rename dance, which buys nothing on an object store and adds a second
failure mode. Every checkpoint entry — and the resume logic that consumes it —
carries a SHA-256 so a truncated ``.pt`` is detected before it is ever resumed.

This module is intentionally free of any MLflow / Ultralytics imports so it can
be unit-tested in isolation with a mock boto3 client.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from botocore.exceptions import ClientError
from pydantic import BaseModel, Field

_logger = logging.getLogger(__name__)

# Bump only on a breaking change to the field contract below. F-02 / F-06
# (meter-peter, operator side) parse these exact keys — see issue #21.
SCHEMA_VERSION = 1

RUN_STATE_FILENAME = "run_state.json"

# Read in 1 MiB blocks so hashing a checkpoint never loads the whole file.
_HASH_CHUNK_BYTES = 1024 * 1024


class CheckpointEntry(BaseModel):
    """One checkpoint recorded in the run-state index (newest last)."""

    epoch: int = Field(description="1-indexed epoch this checkpoint completed.")
    uri: str = Field(description="s3:// URI of the uploaded .pt file.")
    sha256: str = Field(
        description="SHA-256 of the .pt — verified before any resume (CON-04)."
    )
    ultralytics_version: str = Field(
        default="unknown",
        description="Ultralytics version that produced the checkpoint (T-06 mismatch warning).",
    )


class ResumeInfo(BaseModel):
    """Lineage block, present only on resumed runs (written by F-04)."""

    resumed_from: str = Field(
        description="MLflow run ID of the run this one continues."
    )
    attempt: int = Field(
        default=1, description="Resume attempt counter (1 = first resume)."
    )


class RunState(BaseModel):
    """The ``run_state.json`` document. Field names are the frozen F-02/F-06 contract."""

    schema_version: int = Field(default=SCHEMA_VERSION)
    experiment_name: str
    mlflow_run_id: str = Field(
        default="",
        description="Active MLflow run id, written at training START (F-04).",
    )
    last_completed_epoch: int = Field(
        default=0,
        description="Resume starts at this + 1. Tracks the latest *uploaded* checkpoint.",
    )
    checkpoints: list[CheckpointEntry] = Field(default_factory=list)
    dataset_manifest_sha256: str = Field(
        default="",
        description="Dataset identity (F-05); resume validation requires equality (D-04).",
    )
    config_hash: str = Field(
        default="", description="Config hash for incompatible-field diff (T-03)."
    )
    source_workflow_uid: str = Field(
        default="",
        description="From ARGO_WORKFLOW_UID env injected by F-01.",
    )
    source_workflow_name: str = Field(
        default="", description="From ARGO_WORKFLOW_NAME (readability)."
    )
    heartbeat: str = Field(
        default="",
        description="RFC3339 timestamp of the last update — live-experiment guard (T-05).",
    )
    resume: Optional[ResumeInfo] = None


def now_rfc3339() -> str:
    """Return the current UTC time as an RFC3339 string (heartbeat / T-05)."""
    return datetime.now(timezone.utc).isoformat()


def compute_sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file, streamed in fixed-size blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def run_state_key(checkpoint_prefix: str, experiment_name: str) -> str:
    """Return the S3 key for an experiment's run_state.json (matches checkpoint layout)."""
    prefix = checkpoint_prefix.strip("/")
    return f"{prefix}/{experiment_name}/{RUN_STATE_FILENAME}"


class RunStateStore:
    """Reads/writes a single experiment's ``run_state.json`` on S3-compatible storage.

    The same boto3 client used for checkpoint upload is reused, so on GCP the
    writes go through the lakeFS S3 gateway exactly like the checkpoints do
    (bug/480).
    """

    def __init__(self, s3_client: Any, bucket: str, key: str) -> None:
        self._s3 = s3_client
        self._bucket = bucket
        self._key = key

    @property
    def uri(self) -> str:
        return f"s3://{self._bucket}/{self._key}"

    def write(self, state: RunState) -> None:
        """Atomically write the manifest (single put_object of the full body)."""
        body = state.model_dump_json(indent=2).encode("utf-8")
        self._s3.put_object(Bucket=self._bucket, Key=self._key, Body=body)

    def read(self) -> Optional[RunState]:
        """Return the existing manifest, or None when it does not exist yet.

        A missing object (fresh experiment) is the normal first-run case and
        returns None. Any other error propagates to the caller, which decides
        whether to treat it as fatal.
        """
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=self._key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "NoSuchBucket", "404", "NotFound"}:
                return None
            raise
        raw = resp["Body"].read()
        return RunState.model_validate_json(raw)
