"""Tests for the Manager class.

The Manager's boto3 client construction is patched so no real AWS calls are
made.  The DatasetLoadingService is also replaced with a lightweight mock to
keep the test focused on the Manager's wiring logic.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from app.manager import Manager
from app.models.config import Config
from app.services.dataset_loading import DatasetLoadingService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_minimal_yolo_tree(root: Path, n: int = 2) -> None:
    """Create a minimal YOLO tree under *root* for use in manager tests."""
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(1, n + 1):
            stem = f"img{i:06d}"
            (root / "images" / split / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
            # class=0 + 4 bbox + 11 keypoints × (x, y, vis=2)
            parts = ["0"] + ["0.5"] * 4
            for _ in range(11):
                parts.extend(["0.5", "0.5", "2"])
            line = " ".join(parts) + "\n"
            (root / "labels" / split / f"{stem}.txt").write_text(line)

    data = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "kpt_shape": [11, 3],
        "flip_idx": [],
        "names": {0: "spacecraft"},
    }
    (root / "data.yaml").write_text(yaml.dump(data))


def _make_s3_mock(
    source_root: Path, prefix: str = "datasets/speedplus_yolo/v1/"
) -> MagicMock:
    """Return a mock S3 client that serves files from *source_root*."""
    mock_s3 = MagicMock()
    all_keys = [
        str(p.relative_to(source_root))
        for p in sorted(source_root.rglob("*"))
        if p.is_file()
    ]

    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Contents": [{"Key": prefix + k} for k in all_keys]}
    ]
    mock_s3.get_paginator.return_value = paginator

    def download_file(bucket: str, key: str, dest: str) -> None:
        relative = key[len(prefix) :]
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes((source_root / relative).read_bytes())

    mock_s3.download_file.side_effect = download_file
    return mock_s3


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestManagerRun:
    def test_manager_runs_and_writes_stats(self, tmp_path: Path) -> None:
        """Manager.run() completes and writes dataset_stats.json."""
        source = tmp_path / "source"
        _build_minimal_yolo_tree(source)
        mock_s3 = _make_s3_mock(source)

        with patch("app.manager.boto3") as mock_boto3_module:
            mock_boto3_module.client.return_value = mock_s3
            manager = Manager()
            manager._s3_client = mock_s3
            manager._service = DatasetLoadingService(s3_client=mock_s3)

            manager.run(
                version="v1",
                source="s3",
                output_dir=str(tmp_path / "output"),
            )

        stats_path = tmp_path / "output" / "dataset_stats.json"
        assert stats_path.exists()

        with stats_path.open() as fh:
            data = json.load(fh)

        assert data["version"] == "v1"
        assert data["source"] == "s3"
        assert data["train_images"] == 2
        assert data["val_images"] == 2
        assert data["test_images"] == 2

    def test_manager_passes_sample_size(self, tmp_path: Path) -> None:
        """Manager.run() propagates sample_size and seed to the service."""
        source = tmp_path / "source"
        _build_minimal_yolo_tree(source, n=5)
        mock_s3 = _make_s3_mock(source)

        with patch("app.manager.boto3"):
            manager = Manager()
            manager._s3_client = mock_s3
            manager._service = DatasetLoadingService(s3_client=mock_s3)

            manager.run(
                version="v1",
                source="s3",
                output_dir=str(tmp_path / "output"),
                sample_size=2,
                seed=7,
            )

        stats_path = tmp_path / "output" / "dataset_stats.json"
        with stats_path.open() as fh:
            data = json.load(fh)

        assert data["sampled"] is True
        assert data["sample_size"] == 2
        assert data["seed"] == 7

    def test_manager_uses_config_for_log_level(self, tmp_path: Path) -> None:
        """Manager respects the log_level from the injected Config."""
        import logging

        source = tmp_path / "source"
        _build_minimal_yolo_tree(source)
        mock_s3 = _make_s3_mock(source)

        config = Config(log_level="DEBUG")
        with patch("app.manager.boto3"):
            manager = Manager(config=config)
            manager._s3_client = mock_s3
            manager._service = DatasetLoadingService(s3_client=mock_s3)

            manager.run(
                version="v1",
                source="s3",
                output_dir=str(tmp_path / "output"),
            )

        # Simply verify the manager was constructed without error with DEBUG level
        assert manager._config.log_level == "DEBUG"

    def test_manager_selects_lakefs_endpoint_when_configured(
        self, tmp_path: Path
    ) -> None:
        """When lakefs_endpoint is set, boto3.client is called with that endpoint."""
        config = Config(
            lakefs_endpoint="https://lakefs.example.com",
            lakefs_access_key="key",
            lakefs_secret_key="secret",
        )

        with patch("app.manager.boto3") as mock_boto3_module:
            mock_boto3_module.client.return_value = MagicMock()
            manager = Manager(config=config)

        call_kwargs = mock_boto3_module.client.call_args
        assert call_kwargs.kwargs.get("endpoint_url") == "https://lakefs.example.com"
        assert call_kwargs.kwargs.get("aws_access_key_id") == "key"
        assert call_kwargs.kwargs.get("aws_secret_access_key") == "secret"

    def test_manager_service_error_propagates(self, tmp_path: Path) -> None:
        """Errors raised by the service bubble up through Manager.run()."""
        from app.services.dataset_loading import DatasetLoadingError

        mock_s3 = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": []}]
        mock_s3.get_paginator.return_value = paginator

        with patch("app.manager.boto3"):
            manager = Manager()
            manager._s3_client = mock_s3
            manager._service = DatasetLoadingService(s3_client=mock_s3)

        with pytest.raises(DatasetLoadingError):
            manager.run(
                version="v1",
                source="s3",
                output_dir=str(tmp_path / "output"),
            )
