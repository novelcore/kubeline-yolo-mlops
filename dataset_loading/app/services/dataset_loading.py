"""YOLO dataset loading service.

Responsibilities
----------------
1. Download a YOLO pose-estimation dataset from S3 or LakeFS to a local directory.
2. Generate (or copy) ``data.yaml`` with the correct runtime ``path`` field.
3. Validate the downloaded dataset for structural and semantic correctness.
4. Optionally sub-sample the dataset to a fixed number of image+label pairs per split.
5. Write a ``dataset_stats.json`` artifact to the output directory.
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import yaml

if TYPE_CHECKING:
    import boto3 as boto3_type  # noqa: F401  (type-checking only)

from app.models.dataset import YoloDatasetParams, YoloDatasetStats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLITS = ("train", "val", "test")
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# YOLO pose label: class + 4 bbox values + 11 keypoints × 3 values = 38 tokens
_EXPECTED_TOKENS_PER_LINE = 1 + 4 + 11 * 3  # 38

# Number of label lines spot-checked per split during validation
_SPOT_CHECK_LINES = 10

# Default S3 bucket and key prefix for the canonical dataset location
_DEFAULT_BUCKET = "temp-mlops"
_DEFAULT_PREFIX_TEMPLATE = "datasets/speedplus_yolo/{version}/"

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DatasetLoadingError(Exception):
    """Raised when the dataset download or validation step fails."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DatasetLoadingService:
    """Downloads, validates, and optionally samples a YOLO pose dataset.

    Parameters
    ----------
    s3_client:
        A pre-constructed ``boto3`` S3 client.  Injected so the service is
        unit-testable without real AWS calls.
    config:
        Optional step-level config (only used for ``max_retries`` / logging;
        the service is self-contained for the core logic).
    """

    def __init__(self, s3_client: Any, max_retries: int = 3) -> None:
        self._s3 = s3_client
        self._max_retries = max_retries
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, params: YoloDatasetParams) -> YoloDatasetStats:
        """Execute the full dataset loading pipeline.

        Parameters
        ----------
        params:
            All user-supplied parameters for this run.

        Returns
        -------
        YoloDatasetStats
            Statistics about the downloaded (and optionally sampled) dataset.

        Raises
        ------
        DatasetLoadingError
            On any download, validation, or sampling failure.
        """
        output_path = Path(params.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        bucket, prefix = self._resolve_source(params)

        self._logger.info(
            "Starting dataset download | source=%s bucket=%s prefix=%s -> %s",
            params.source,
            bucket,
            prefix,
            output_path,
        )

        self._download(bucket, prefix, output_path)
        self._write_data_yaml(output_path, params.output_dir)
        self._validate(output_path)

        sampled = False
        if params.sample_size is not None:
            self._sample(output_path, params.sample_size, params.seed)
            sampled = True

        split_counts = self._count_splits(output_path)

        stats = YoloDatasetStats(
            version=params.version,
            source=params.source,
            train_images=split_counts["train"],
            val_images=split_counts["val"],
            test_images=split_counts["test"],
            sampled=sampled,
            sample_size=params.sample_size,
            seed=params.seed,
        )

        self._write_stats(output_path, stats)

        self._logger.info(
            "Dataset loading complete | train=%d val=%d test=%d sampled=%s",
            stats.train_images,
            stats.val_images,
            stats.test_images,
            stats.sampled,
        )
        return stats

    # ------------------------------------------------------------------
    # Source resolution
    # ------------------------------------------------------------------

    def _resolve_source(self, params: YoloDatasetParams) -> tuple[str, str]:
        """Return (bucket, key_prefix) based on the user's params.

        When ``path_override`` is provided it must be a valid ``s3://`` URI.
        Otherwise the canonical convention is used.
        """
        if params.path_override:
            uri = params.path_override.strip()
            match = re.match(r"^s3://([^/]+)/(.*)$", uri)
            if not match:
                raise DatasetLoadingError(
                    f"path_override must be a valid s3:// URI, got: {uri!r}"
                )
            bucket = match.group(1)
            prefix = match.group(2)
            if not prefix.endswith("/"):
                prefix += "/"
            return bucket, prefix

        bucket = _DEFAULT_BUCKET
        prefix = _DEFAULT_PREFIX_TEMPLATE.format(version=params.version)
        return bucket, prefix

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download(self, bucket: str, prefix: str, output_path: Path) -> None:
        """List all objects under *prefix* and download them to *output_path*.

        Preserves the relative directory structure from the S3 prefix.
        """
        paginator = self._s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        downloaded = 0
        for page in pages:
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                # Strip the common prefix to get the relative path
                relative = key[len(prefix) :]
                if not relative:
                    # Skip the "directory" placeholder object itself
                    continue

                local_file = output_path / relative
                local_file.parent.mkdir(parents=True, exist_ok=True)

                self._logger.debug(
                    "Downloading s3://%s/%s -> %s", bucket, key, local_file
                )
                self._s3.download_file(bucket, key, str(local_file))
                downloaded += 1

        if downloaded == 0:
            raise DatasetLoadingError(
                f"No objects found at s3://{bucket}/{prefix} — "
                "check the bucket name, prefix, and credentials."
            )

        self._logger.info(
            "Downloaded %d files from s3://%s/%s", downloaded, bucket, prefix
        )

    # ------------------------------------------------------------------
    # data.yaml generation
    # ------------------------------------------------------------------

    def _write_data_yaml(self, output_path: Path, runtime_root: str) -> None:
        """Write (or overwrite) ``data.yaml`` at the dataset root.

        If a ``data.yaml`` was downloaded from S3 it is used as a base;
        otherwise a canonical template is generated.  In either case the
        ``path`` field is set to the absolute ``runtime_root`` so YOLO
        training tools resolve splits correctly.
        """
        yaml_path = output_path / "data.yaml"

        if yaml_path.exists():
            with yaml_path.open() as fh:
                content: dict[str, Any] = yaml.safe_load(fh) or {}
        else:
            content = {
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "kpt_shape": [11, 3],
                "flip_idx": [],
                "names": {0: "spacecraft"},
            }

        content["path"] = str(Path(runtime_root).resolve())

        with yaml_path.open("w") as fh:
            yaml.dump(content, fh, default_flow_style=False, sort_keys=False)

        self._logger.info("Wrote data.yaml to %s", yaml_path)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, output_path: Path) -> None:
        """Validate the YOLO dataset rooted at *output_path*.

        Checks
        ------
        - ``data.yaml`` exists and contains required keys.
        - Each split directory exists and contains at least one image.
        - Every image has a corresponding ``labels/<split>/<stem>.txt`` file.
        - Every label file is non-empty.
        - Spot-check: label lines have exactly 38 tokens.
        """
        self._logger.info("Validating dataset at %s", output_path)

        self._validate_data_yaml(output_path)

        for split in SPLITS:
            images_dir = output_path / "images" / split
            labels_dir = output_path / "labels" / split

            if not images_dir.is_dir():
                raise DatasetLoadingError(f"Missing split directory: {images_dir}")

            image_files = [
                f
                for f in images_dir.iterdir()
                if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ]
            if not image_files:
                raise DatasetLoadingError(
                    f"Split '{split}' contains no images in {images_dir}"
                )

            for img_file in image_files:
                label_file = labels_dir / (img_file.stem + ".txt")
                if not label_file.exists():
                    raise DatasetLoadingError(
                        f"Missing label file for image: {img_file} "
                        f"(expected {label_file})"
                    )
                if label_file.stat().st_size == 0:
                    raise DatasetLoadingError(f"Label file is empty: {label_file}")

            self._spot_check_labels(labels_dir, split)

        self._logger.info("Validation passed")

    def _validate_data_yaml(self, output_path: Path) -> None:
        """Check that ``data.yaml`` exists and has all required keys."""
        yaml_path = output_path / "data.yaml"
        if not yaml_path.exists():
            raise DatasetLoadingError(f"data.yaml not found at {yaml_path}")

        with yaml_path.open() as fh:
            content = yaml.safe_load(fh) or {}

        required_keys = {"path", "train", "val", "test", "kpt_shape", "names"}
        missing = required_keys - set(content.keys())
        if missing:
            raise DatasetLoadingError(
                f"data.yaml is missing required keys: {sorted(missing)}"
            )

    def _spot_check_labels(self, labels_dir: Path, split: str) -> None:
        """Spot-check up to ``_SPOT_CHECK_LINES`` label files in *labels_dir*.

        Each non-empty line must contain exactly 38 whitespace-separated tokens:
        ``<class> <cx> <cy> <w> <h> <kp1x> <kp1y> <kp1vis> ... <kp11x> <kp11y> <kp11vis>``
        """
        if not labels_dir.is_dir():
            return

        label_files = sorted(labels_dir.glob("*.txt"))[:_SPOT_CHECK_LINES]
        for label_path in label_files:
            with label_path.open() as fh:
                for lineno, raw_line in enumerate(fh, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    tokens = line.split()
                    if len(tokens) != _EXPECTED_TOKENS_PER_LINE:
                        raise DatasetLoadingError(
                            f"Label format error in {label_path} line {lineno}: "
                            f"expected {_EXPECTED_TOKENS_PER_LINE} tokens, "
                            f"got {len(tokens)}"
                        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample(self, output_path: Path, sample_size: int, seed: int) -> None:
        """Keep *sample_size* image+label pairs per split; delete the rest.

        If a split has fewer than *sample_size* images a warning is logged
        and all images in that split are kept.
        """
        rng = random.Random(seed)

        for split in SPLITS:
            images_dir = output_path / "images" / split
            labels_dir = output_path / "labels" / split

            image_files = sorted(
                f
                for f in images_dir.iterdir()
                if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            )
            total = len(image_files)

            if total <= sample_size:
                self._logger.warning(
                    "Split '%s' has only %d images (<= sample_size=%d); keeping all.",
                    split,
                    total,
                    sample_size,
                )
                continue

            keep = set(rng.sample(image_files, sample_size))
            removed = 0
            for img_file in image_files:
                if img_file not in keep:
                    img_file.unlink()
                    label_file = labels_dir / (img_file.stem + ".txt")
                    if label_file.exists():
                        label_file.unlink()
                    removed += 1

            self._logger.info(
                "Split '%s': kept %d / %d images (removed %d).",
                split,
                sample_size,
                total,
                removed,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_splits(self, output_path: Path) -> dict[str, int]:
        """Return a mapping of split name -> image count after all operations."""
        counts: dict[str, int] = {}
        for split in SPLITS:
            images_dir = output_path / "images" / split
            if images_dir.is_dir():
                counts[split] = sum(
                    1
                    for f in images_dir.iterdir()
                    if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                )
            else:
                counts[split] = 0
        return counts

    def _write_stats(self, output_path: Path, stats: YoloDatasetStats) -> None:
        """Serialize *stats* to ``{output_path}/dataset_stats.json``."""
        stats_path = output_path / "dataset_stats.json"
        with stats_path.open("w") as fh:
            json.dump(stats.model_dump(), fh, indent=2)
        self._logger.info("Wrote dataset stats to %s", stats_path)
