"""Custom PyTorch Dataset that streams YOLO images directly from S3.

Ultralytics builds its dataset internally via ``build_yolo_dataset``. To inject
S3 streaming we subclass ``ultralytics.data.YOLODataset`` and override the
image-loading method so that images are fetched from S3 on demand rather than
read from disk. Label files are expected to live on the local filesystem at
``{local_labels_root}/{split}/{stem}.txt`` — only image bytes are streamed.

Architecture note
-----------------
Ultralytics calls ``self.im_files`` (a list of file paths) to enumerate the
dataset and ``cv2.imread`` inside ``load_image`` to load each sample. We
substitute the file paths with synthetic "s3://" URIs and override ``load_image``
to retrieve the bytes from S3 and decode them with OpenCV in-memory. Labels are
resolved from the synthetic path by replacing the S3 URI with the local label
path following the same stem-matching convention.

An optional ``LruDiskCache`` bounds the local disk footprint so that frequently
accessed images (e.g. mosaic partners) are served from disk rather than
re-downloaded every time.
"""

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.services.lru_disk_cache import LruDiskCache

_logger = logging.getLogger(__name__)

# Ultralytics imports are deferred to keep this module importable in test
# environments where ultralytics may be mocked.
try:
    from ultralytics.data import YOLODataset as _UltralyticsYOLODataset
    from ultralytics.data.dataset import (
        DATASET_CACHE_VERSION,
        get_hash,
        load_dataset_cache_file,
        save_dataset_cache_file,
    )
    from ultralytics.data.utils import HELP_URL
    from ultralytics.utils import LOGGER, TQDM

    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False
    _UltralyticsYOLODataset = object  # type: ignore[assignment,misc]
    TQDM = None  # type: ignore[assignment]


class S3YoloDataset(_UltralyticsYOLODataset):  # type: ignore[misc]
    """YOLO dataset that fetches image bytes from S3 instead of local disk.

    Parameters
    ----------
    s3_client:
        A boto3 S3 client used for ``get_object`` calls.
    s3_bucket:
        Name of the S3 bucket holding the images.
    s3_prefix:
        Key prefix under which images are stored, e.g.
        ``"datasets/speedplus_yolo/v1/images/train/"``.
    local_labels_root:
        Local directory under which label ``.txt`` files are stored, structured
        as ``<local_labels_root>/<split>/<stem>.txt``.
    split:
        Dataset split name: ``"train"``, ``"val"``, or ``"test"``.
    cache_dir:
        Local directory for the LRU disk cache.  ``None`` disables caching.
    cache_max_bytes:
        Maximum disk budget for cached images (default 2 GiB).
    *args / **kwargs:
        Forwarded to the Ultralytics ``YOLODataset`` constructor.
    """

    def __init__(
        self,
        *args: Any,
        s3_client: Any,
        s3_bucket: str,
        s3_prefix: str,
        local_labels_root: str,
        split: str,
        s3_labels_prefix: str | None = None,
        cache_dir: str | None = None,
        cache_max_bytes: int = 2 * 1024**3,
        **kwargs: Any,
    ) -> None:
        if not _ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics is required to use S3YoloDataset. "
                "Install it with: pip install ultralytics"
            )

        self._s3_client = s3_client
        self._s3_bucket = s3_bucket
        # Normalise prefix: must end with "/"
        self._s3_prefix = s3_prefix.rstrip("/") + "/"
        self._local_labels_root = Path(local_labels_root)
        self._split = split
        # When set, labels are fetched from S3 instead of local disk
        self._s3_labels_prefix: str | None = (
            s3_labels_prefix.rstrip("/") + "/" if s3_labels_prefix else None
        )

        # Disk cache (None = disabled)
        self._disk_cache: LruDiskCache | None = None
        if cache_dir is not None:
            self._disk_cache = LruDiskCache(
                cache_dir=cache_dir,
                max_bytes=cache_max_bytes,
            )

        # Ultralytics needs a path argument pointing to the image directory.
        # We pass a synthetic sentinel; get_image_and_label is overridden below.
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Enumerate images from S3
    # ------------------------------------------------------------------

    def get_img_files(self, img_path: str) -> list[str]:
        """Return synthetic s3:// URIs for all images under the S3 prefix.

        Ultralytics calls this method (or reads ``self.im_files``) to build the
        file list. We list the bucket prefix and return each object key wrapped
        in an ``s3://`` URI so the rest of the pipeline has a unique identifier
        per sample.
        """
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        uris: list[str] = []

        paginator = self._s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self._s3_bucket, Prefix=self._s3_prefix)

        for page in pages:
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if Path(key).suffix.lower() in supported_exts:
                    uris.append(f"s3://{self._s3_bucket}/{key}")

        if not uris:
            raise FileNotFoundError(
                f"No images found at s3://{self._s3_bucket}/{self._s3_prefix}"
            )

        _logger.debug(
            "S3YoloDataset (%s): found %d images at s3://%s/%s",
            self._split,
            len(uris),
            self._s3_bucket,
            self._s3_prefix,
        )
        return uris

    # ------------------------------------------------------------------
    # Resolve label path from synthetic S3 URI
    # ------------------------------------------------------------------

    def img2label_paths(self, img_paths: list[str]) -> list[str]:
        """Map synthetic s3:// image URIs to label paths.

        When ``s3_labels_prefix`` is set (manifest-only mode), returns
        synthetic local paths used as cache keys — the actual label data
        is fetched from S3 in ``cache_labels()``.

        Otherwise returns the path ``<local_labels_root>/<split>/<stem>.txt``
        for each image URI, which is where ``dataset_loading`` placed the
        label files.
        """
        label_paths: list[str] = []
        for uri in img_paths:
            # uri = "s3://<bucket>/<prefix>/<stem>.<ext>"
            stem = Path(uri).stem
            label_path = self._local_labels_root / self._split / f"{stem}.txt"
            label_paths.append(str(label_path))
        return label_paths

    # ------------------------------------------------------------------
    # Label loading — override to use local label paths, not S3-derived
    # ------------------------------------------------------------------

    def get_labels(self) -> list[dict]:
        """Return label dicts using local label files instead of S3-derived paths.

        The parent ``YOLODataset.get_labels()`` calls the standalone
        ``img2label_paths()`` function which replaces ``/images/`` with
        ``/labels/`` in S3 URIs — producing nonsensical paths.  We override
        to use ``self.img2label_paths()`` which maps to local label files.
        """
        self.label_files = self.img2label_paths(self.im_files)

        if not self.label_files:
            raise RuntimeError(
                f"No label files found for S3 dataset (split={self._split}). "
                "Check that S3 paginator returned images."
            )

        # Build a local cache path from the label directory
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")

        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, exists = self.cache_labels(cache_path), False

        nf, nm, ne, nc, n = cache.pop("results")
        if exists:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))

        [cache.pop(k) for k in ("hash", "version", "msgs")]
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(
                f"No valid labels found in {cache_path}. {HELP_URL}"
            )
        self.im_files = [lb["im_file"] for lb in labels]
        return labels

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict:
        """Build label cache from label files without opening S3 images.

        When ``_s3_labels_prefix`` is set (manifest-only mode), labels are
        fetched from S3 via ``get_object`` and parsed in-memory — no local
        label files required.

        Otherwise, the parent implementation's ``verify_image_label()`` is
        skipped and local label ``.txt`` files are read directly.

        Image shapes are set to a placeholder; Ultralytics replaces them with
        the actual shape from ``load_image()`` at training time.
        """
        x: dict[str, Any] = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)

        nkpt, ndim = self.data.get("kpt_shape", (0, 0))

        pbar = TQDM(
            zip(self.im_files, self.label_files),
            desc=desc,
            total=total,
        )
        for im_file, lb_file in pbar:
            try:
                lb_text = self._read_label(lb_file)

                if lb_text is None:
                    nm += 1
                    continue

                lb_lines = [
                    line.split()
                    for line in lb_text.strip().splitlines()
                    if len(line)
                ]

                if not lb_lines:
                    ne += 1
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": (self.imgsz, self.imgsz),
                            "cls": np.zeros((0, 1), dtype=np.float32),
                            "bboxes": np.zeros((0, 4), dtype=np.float32),
                            "segments": [],
                            "keypoints": None,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                    continue

                lb = np.array(lb_lines, dtype=np.float32)
                nl = len(lb)

                if self.use_keypoints and nl:
                    keypoints = lb[:, 5:].reshape(nl, nkpt, ndim)
                else:
                    keypoints = None

                nf += 1
                x["labels"].append(
                    {
                        "im_file": im_file,
                        "shape": (self.imgsz, self.imgsz),
                        "cls": lb[:, 0:1],
                        "bboxes": lb[:, 1:5],
                        "segments": [],
                        "keypoints": keypoints,
                        "normalized": True,
                        "bbox_format": "xywh",
                    }
                )

            except Exception as exc:
                nc += 1
                msgs.append(f"{self.prefix}{lb_file}: {exc}")

            pbar.desc = (
                f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            )
        pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(
                f"{self.prefix}No labels found in {path}. {HELP_URL}"
            )

        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs
        x["version"] = DATASET_CACHE_VERSION
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    # ------------------------------------------------------------------
    # Label reading — local or S3
    # ------------------------------------------------------------------

    def _read_label(self, lb_file: str) -> str | None:
        """Read label text from local file or S3.

        In S3-label mode (``_s3_labels_prefix`` is set), the corresponding
        S3 key is derived from the label file stem and fetched via
        ``get_object``. No files are written to disk.

        Returns the label file content as a string, or ``None`` if not found.
        """
        if self._s3_labels_prefix is not None:
            stem = Path(lb_file).stem
            s3_key = f"{self._s3_labels_prefix}{stem}.txt"
            try:
                response = self._s3_client.get_object(
                    Bucket=self._s3_bucket, Key=s3_key
                )
                return response["Body"].read().decode("utf-8")
            except self._s3_client.exceptions.NoSuchKey:
                return None
            except Exception as exc:
                _logger.warning(
                    "Failed to fetch label from s3://%s/%s: %s",
                    self._s3_bucket,
                    s3_key,
                    exc,
                )
                return None

        # Local file mode
        if not os.path.isfile(lb_file):
            return None

        with open(lb_file, encoding="utf-8") as f:
            return f.read()

    # ------------------------------------------------------------------
    # Load image bytes from S3 (with optional disk cache)
    # ------------------------------------------------------------------

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Fetch image *i* from S3 and decode it as an OpenCV BGR array.

        When a disk cache is configured the flow is:
        1. Check cache — on hit, read bytes from the local file.
        2. On miss: ``s3_client.get_object()``, then ``cache.put(key, bytes)``.
        3. ``cv2.imdecode()`` + resize.

        Maintains ``self.buffer`` so that Mosaic augmentation can pick random
        partner images (the parent ``load_image`` populates this buffer; without
        it ``Mosaic.get_indexes()`` fails on the empty list).

        Returns the same ``(image, original_hw, resized_hw)`` tuple as the
        standard Ultralytics ``load_image`` implementation.
        """
        # Return RAM-cached image if available
        im = self.ims[i]
        if im is not None:
            return im, self.im_hw0[i], self.im_hw[i]

        uri: str = self.im_files[i]
        # Parse the key from the synthetic URI
        # uri format: "s3://<bucket>/<key>"
        key = uri[len(f"s3://{self._s3_bucket}/"):]

        raw_bytes: bytes | None = None

        # -- Disk cache lookup --
        if self._disk_cache is not None:
            cached_path = self._disk_cache.get(key)
            if cached_path is not None:
                raw_bytes = cached_path.read_bytes()

        # -- S3 fetch on miss --
        if raw_bytes is None:
            try:
                response = self._s3_client.get_object(Bucket=self._s3_bucket, Key=key)
                raw_bytes = response["Body"].read()
            except Exception as exc:
                raise OSError(
                    f"Failed to download image from s3://{self._s3_bucket}/{key}: {exc}"
                ) from exc

            # Store in cache
            if self._disk_cache is not None:
                self._disk_cache.put(key, raw_bytes)

        img_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        im = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if im is None:
            raise OSError(
                f"cv2.imdecode returned None for s3://{self._s3_bucket}/{key}"
            )

        h0, w0 = im.shape[:2]

        # Resize — match parent behaviour (aspect-preserving or stretch)
        if rect_mode:
            r = self.imgsz / max(h0, w0)
            if r != 1:
                w = min(int(round(w0 * r)), self.imgsz)
                h = min(int(round(h0 * r)), self.imgsz)
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz):
            im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        # Populate buffer for Mosaic augmentation (mirrors parent logic)
        if self.augment:
            self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
            self.buffer.append(i)
            if len(self.buffer) >= self.max_buffer_length:
                j = self.buffer.pop(0)
                if self.cache != "ram":
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

        return im, (h0, w0), im.shape[:2]

    # ------------------------------------------------------------------
    # Override cache_images — we handle caching ourselves
    # ------------------------------------------------------------------

    def cache_images(self, cache: str = "ram") -> None:
        """No-op: S3YoloDataset manages its own disk cache."""
        pass

    # ------------------------------------------------------------------
    # Cache metrics for MLflow logging
    # ------------------------------------------------------------------

    @property
    def cache_metrics(self) -> tuple[int, int, int]:
        """Return ``(hits, misses, evictions)`` and reset counters."""
        if self._disk_cache is not None:
            return self._disk_cache.reset_metrics()
        return (0, 0, 0)

    @property
    def cache_size_bytes(self) -> int:
        """Current disk cache size in bytes."""
        if self._disk_cache is not None:
            return self._disk_cache.current_bytes
        return 0
