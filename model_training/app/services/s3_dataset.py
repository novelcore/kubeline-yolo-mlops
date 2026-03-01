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
"""

import io
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_logger = logging.getLogger(__name__)

# Ultralytics imports are deferred to keep this module importable in test
# environments where ultralytics may be mocked.
try:
    from ultralytics.data import YOLODataset as _UltralyticsYOLODataset
    from ultralytics.utils import TQDM

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
        """Map synthetic s3:// image URIs to local label file paths.

        Returns the path ``<local_labels_root>/<split>/<stem>.txt`` for each
        image URI, which is where ``dataset_loading`` placed the label files.
        """
        label_paths: list[str] = []
        for uri in img_paths:
            # uri = "s3://<bucket>/<prefix>/<stem>.<ext>"
            stem = Path(uri).stem
            label_path = self._local_labels_root / self._split / f"{stem}.txt"
            label_paths.append(str(label_path))
        return label_paths

    # ------------------------------------------------------------------
    # Load image bytes from S3
    # ------------------------------------------------------------------

    def load_image(self, i: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Fetch image *i* from S3 and decode it as an OpenCV BGR array.

        Returns the same (image, original_hw, resized_hw) tuple as the
        standard Ultralytics ``load_image`` implementation.
        """
        uri: str = self.im_files[i]
        # Parse the key from the synthetic URI
        # uri format: "s3://<bucket>/<key>"
        key = uri[len(f"s3://{self._s3_bucket}/"):]

        try:
            response = self._s3_client.get_object(Bucket=self._s3_bucket, Key=key)
            raw_bytes = response["Body"].read()
        except Exception as exc:
            raise OSError(
                f"Failed to download image from s3://{self._s3_bucket}/{key}: {exc}"
            ) from exc

        img_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(
                f"cv2.imdecode returned None for s3://{self._s3_bucket}/{key}"
            )

        original_hw = (img.shape[0], img.shape[1])

        # Resize to the target image size (self.imgsz is set by Ultralytics)
        target = self.imgsz
        img = cv2.resize(img, (target, target))
        resized_hw = (img.shape[0], img.shape[1])

        return img, original_hw, resized_hw
