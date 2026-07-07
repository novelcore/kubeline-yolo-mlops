"""Sync-diff tests — the plan must upload new/changed files, DELETE remote-only
files (the whole point vs the old additive-only tools), and skip unchanged
files, keyed by content checksum."""
from __future__ import annotations

import hashlib
import pathlib

from dataset_cli.sync import plan_sync


class FakeLakeFS:
    """Minimal stand-in exposing list_objects(repo, ref, prefix)."""

    def __init__(self, remote: dict[str, str]):
        # remote: {repo_path: checksum}
        self._remote = remote

    def list_objects(self, repo, ref, prefix=""):
        for path, checksum in self._remote.items():
            if path.startswith(prefix):
                yield {"path": path, "checksum": checksum, "path_type": "object"}


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _write(root: pathlib.Path, rel: str, content: bytes):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return _md5(content)


def test_add_change_delete_unchanged(tmp_path):
    # local tree
    m_same = _write(tmp_path, "images/train/keep.jpg", b"same")
    _write(tmp_path, "images/train/new.jpg", b"brand-new")     # add
    m_changed_local = _write(tmp_path, "labels/train/edit.txt", b"NEW CONTENT")  # changed

    # remote has: keep (same), edit (old checksum), and gone.jpg (remote-only)
    remote = {
        "images/train/keep.jpg": m_same,
        "labels/train/edit.txt": _md5(b"OLD CONTENT"),
        "images/train/gone.jpg": _md5(b"stale"),
    }
    client = FakeLakeFS(remote)

    plan = plan_sync(tmp_path, client, "repo", "main", prefix="")

    added = {rp for _, rp in plan.add}
    assert "images/train/new.jpg" in added        # new file uploaded
    assert "labels/train/edit.txt" in added       # changed content re-uploaded
    assert "images/train/keep.jpg" not in added   # unchanged skipped
    assert plan.unchanged == 1

    assert plan.delete == ["images/train/gone.jpg"]  # remote-only DELETED
    assert m_changed_local != remote["labels/train/edit.txt"]


def test_no_changes_is_empty_plan(tmp_path):
    m = _write(tmp_path, "images/train/a.jpg", b"x")
    client = FakeLakeFS({"images/train/a.jpg": m})
    plan = plan_sync(tmp_path, client, "repo", "main", prefix="")
    assert plan.changes == 0
    assert plan.unchanged == 1


def test_empty_remote_uploads_everything(tmp_path):
    _write(tmp_path, "images/train/a.jpg", b"x")
    _write(tmp_path, "images/val/b.jpg", b"y")
    client = FakeLakeFS({})
    plan = plan_sync(tmp_path, client, "repo", "main", prefix="")
    assert len(plan.add) == 2
    assert plan.delete == []
