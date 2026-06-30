"""Tests for S3YoloDataset — the S3 streaming dataset class.

All tests mock both boto3 and ultralytics so that no real S3 calls or GPU
initialisation occur in CI.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_s3_client() -> MagicMock:
    """A boto3 S3 client mock with a pre-configured paginator."""
    client = MagicMock()

    paginator = MagicMock()
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": "datasets/v1/images/train/img001.jpg"},
                {"Key": "datasets/v1/images/train/img002.jpg"},
            ]
        }
    ]
    client.get_paginator.return_value = paginator

    return client


@pytest.fixture()
def tiny_jpeg_bytes() -> bytes:
    """Minimal valid JPEG bytes (1x1 white pixel, encoded by cv2)."""
    import cv2
    import numpy as np

    img = np.full((1, 1, 3), 255, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Patch ultralytics before importing the module under test
# ---------------------------------------------------------------------------


def _make_dataset(
    s3_client: MagicMock,
    local_labels_root: str,
    split: str = "train",
) -> "S3YoloDataset":  # type: ignore[name-defined]  # noqa: F821
    """Import S3YoloDataset with ultralytics patched out."""
    # We patch the ultralytics base class so __init__ does nothing heavy
    with patch.dict(
        "sys.modules",
        {
            "ultralytics": MagicMock(),
            "ultralytics.data": MagicMock(),
            "ultralytics.utils": MagicMock(),
        },
    ):
        # Re-import to pick up the patched modules
        import importlib
        import app.services.s3_dataset as s3_mod

        importlib.reload(s3_mod)

        # Override _ULTRALYTICS_AVAILABLE so the guard doesn't block us
        s3_mod._ULTRALYTICS_AVAILABLE = True  # type: ignore[attr-defined]

        # Build a lightweight subclass whose __init__ skips the super() call
        class _TestableDataset(s3_mod.S3YoloDataset):
            def __init__(self, **kw: object) -> None:
                # Bypass Ultralytics __init__; set attributes manually
                self._s3_client = kw["s3_client"]
                self._s3_bucket = kw["s3_bucket"]
                self._s3_prefix = str(kw["s3_prefix"]).rstrip("/") + "/"
                self._local_labels_root = Path(str(kw["local_labels_root"]))
                self._split = kw["split"]
                self.imgsz = 64  # small size for tests
                # RAM-cache attrs that Ultralytics' __init__ would normally set;
                # load_image() reads these (single-image tests use index 0).
                self.ims = [None]
                self.im_hw0 = [None]
                self.im_hw = [None]
                self.augment = False
                # Disk cache off by default; cache-specific tests set their own.
                self._disk_cache = kw.get("disk_cache")

        return _TestableDataset(
            s3_client=s3_client,
            s3_bucket="test-bucket",
            s3_prefix="datasets/v1/images/train",
            local_labels_root=local_labels_root,
            split=split,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetImgFiles:
    def test_returns_s3_uris(self, mock_s3_client: MagicMock, tmp_path: Path) -> None:
        """get_img_files lists S3 objects and returns synthetic s3:// URIs."""
        dataset = _make_dataset(mock_s3_client, str(tmp_path))
        uris = dataset.get_img_files("ignored_path")

        assert len(uris) == 2
        assert all(u.startswith("s3://test-bucket/") for u in uris)
        assert "img001.jpg" in uris[0]
        assert "img002.jpg" in uris[1]

    def test_raises_when_no_images(self, tmp_path: Path) -> None:
        """FileNotFoundError is raised when no images exist at the prefix."""
        client = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": []}]
        client.get_paginator.return_value = paginator

        dataset = _make_dataset(client, str(tmp_path))
        with pytest.raises(FileNotFoundError, match="No images found"):
            dataset.get_img_files("ignored")


class TestImg2LabelPaths:
    def test_label_paths_resolved_correctly(
        self, mock_s3_client: MagicMock, tmp_path: Path
    ) -> None:
        """Label paths are resolved from synthetic URIs to local .txt files."""
        dataset = _make_dataset(mock_s3_client, str(tmp_path), split="train")

        uris = [
            "s3://test-bucket/datasets/v1/images/train/img001.jpg",
            "s3://test-bucket/datasets/v1/images/train/img002.jpg",
        ]
        label_paths = dataset.img2label_paths(uris)

        assert len(label_paths) == 2
        assert label_paths[0].endswith("train/img001.txt")
        assert label_paths[1].endswith("train/img002.txt")


class TestLoadImage:
    def test_loads_image_from_s3(
        self,
        mock_s3_client: MagicMock,
        tiny_jpeg_bytes: bytes,
        tmp_path: Path,
    ) -> None:
        """load_image fetches bytes from S3 and returns a decoded numpy array."""
        body_mock = MagicMock()
        body_mock.read.return_value = tiny_jpeg_bytes
        mock_s3_client.get_object.return_value = {"Body": body_mock}

        dataset = _make_dataset(mock_s3_client, str(tmp_path))
        dataset.im_files = ["s3://test-bucket/datasets/v1/images/train/img001.jpg"]

        img, orig_hw, resized_hw = dataset.load_image(0)

        assert isinstance(img, np.ndarray)
        assert img.shape[2] == 3  # BGR channels
        # Resized to imgsz x imgsz (64x64 in our test fixture)
        assert resized_hw == (64, 64)

    def test_raises_on_s3_error(
        self, mock_s3_client: MagicMock, tmp_path: Path
    ) -> None:
        """OSError is raised when S3 get_object fails."""
        mock_s3_client.get_object.side_effect = Exception("S3 unavailable")

        dataset = _make_dataset(mock_s3_client, str(tmp_path))
        dataset.im_files = ["s3://test-bucket/datasets/v1/images/train/img001.jpg"]

        with pytest.raises(OSError, match="Failed to download image"):
            dataset.load_image(0)

    def test_raises_on_corrupt_image(
        self, mock_s3_client: MagicMock, tmp_path: Path
    ) -> None:
        """OSError is raised when image bytes cannot be decoded."""
        body_mock = MagicMock()
        body_mock.read.return_value = b"not-an-image"
        mock_s3_client.get_object.return_value = {"Body": body_mock}

        dataset = _make_dataset(mock_s3_client, str(tmp_path))
        dataset.im_files = ["s3://test-bucket/datasets/v1/images/train/img001.jpg"]

        with pytest.raises(OSError, match="cv2.imdecode returned None"):
            dataset.load_image(0)


class TestCacheEvictionResilience:
    """Issue #175 A1: an evicted/missing cached file must not crash load_image."""

    def test_evicted_cache_file_refetches_instead_of_crashing(
        self,
        mock_s3_client: MagicMock,
        tiny_jpeg_bytes: bytes,
        tmp_path: Path,
    ) -> None:
        from app.services.lru_disk_cache import LruDiskCache

        body_mock = MagicMock()
        body_mock.read.return_value = tiny_jpeg_bytes
        mock_s3_client.get_object.return_value = {"Body": body_mock}

        cache = LruDiskCache(tmp_path / "cache", max_bytes=10 * 1024**2)
        dataset = _make_dataset(mock_s3_client, str(tmp_path))
        dataset._disk_cache = cache
        dataset.im_files = ["s3://test-bucket/datasets/v1/images/train/img001.jpg"]

        # First load: cache miss -> S3 fetch -> cached on disk.
        dataset.load_image(0)
        assert mock_s3_client.get_object.call_count == 1

        # Simulate eviction by another worker: the index still points at a file
        # that no longer exists on disk.
        dataset.ims = [None]  # clear RAM cache to force the disk path
        key = dataset.im_files[0][len("s3://test-bucket/") :]
        cached = cache.get(key)
        assert cached is not None and cached.exists()
        cached.unlink()

        # Old code raised an unhandled FileNotFoundError here; the fix re-fetches.
        img, _, _ = dataset.load_image(0)
        assert img is not None
        assert mock_s3_client.get_object.call_count == 2


class TestForkSafeClient:
    """Issue #175 A2: a forked worker (new PID) must build its own boto3 client."""

    def test_get_s3_rebuilds_client_on_pid_change(
        self, mock_s3_client: MagicMock, tmp_path: Path
    ) -> None:
        built: list[MagicMock] = []

        def factory() -> MagicMock:
            c = MagicMock(name=f"client{len(built)}")
            built.append(c)
            return c

        dataset = _make_dataset(mock_s3_client, str(tmp_path))
        dataset._s3_client_factory = factory
        dataset._s3_client_cached = None
        dataset._s3_client_pid = None

        first = dataset._get_s3()
        again = dataset._get_s3()
        assert first is again, "same process must reuse one client"
        assert len(built) == 1

        # Simulate the fork: the cached client now belongs to a different PID.
        dataset._s3_client_pid = -1
        after_fork = dataset._get_s3()
        assert after_fork is not first, "forked worker must build its own client"
        assert len(built) == 2
