"""PoseTrainer subclass that builds S3YoloDataset instances.

Ultralytics' ``model.train(trainer=...)`` accepts a custom trainer class.  We
subclass ``PoseTrainer`` and override ``build_dataset()`` to return an
``S3YoloDataset`` that streams images from S3 with an LRU disk cache, instead
of the default ``YOLODataset`` that reads from local disk.

Configuration is injected via the :func:`make_s3_pose_trainer` factory, which
bakes S3 connection details and cache parameters into a dynamically created
subclass — avoiding any global state.
"""

import logging
from typing import Any

_logger = logging.getLogger(__name__)

try:
    from ultralytics.models.yolo.pose import PoseTrainer as _PoseTrainer

    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False
    _PoseTrainer = object  # type: ignore[assignment,misc]


class S3PoseTrainer(_PoseTrainer):  # type: ignore[misc]
    """PoseTrainer that builds S3-backed datasets.

    Class attributes ``_s3_client``, ``_s3_bucket``, ``_s3_prefix``,
    ``_local_labels_root``, ``_cache_dir``, and ``_cache_max_bytes`` must be
    set on the *class* (not the instance) before construction.  Use
    :func:`make_s3_pose_trainer` to create a properly configured subclass.
    """

    # Filled by the factory
    _s3_client: Any = None
    _s3_client_factory: Any = None
    _s3_bucket: str = ""
    _s3_prefix: str = ""
    _local_labels_root: str = ""
    _s3_labels_prefix: str | None = None
    _cache_dir: str | None = None
    _cache_max_bytes: int = 2 * 1024**3

    # One disk cache shared across the train and val datasets (single LRU
    # budget over a single directory).  Created lazily on first build_dataset.
    _shared_disk_cache: Any = None

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None) -> Any:
        """Override to return an :class:`S3YoloDataset`.

        Parameters
        ----------
        img_path:
            Path string Ultralytics passes (e.g. ``"images/train"``).
            We use it only to extract the split name.
        mode:
            ``"train"`` or ``"val"``.
        batch:
            Batch size (forwarded to the parent for augmentation settings).
        """
        from app.services.s3_dataset import S3YoloDataset

        # Determine split from img_path first (more reliable), then mode
        split = "train"
        for candidate in ("train", "val", "test"):
            if candidate in img_path:
                split = candidate
                break
        else:
            if mode in ("train", "val", "test"):
                split = mode

        # Build the S3 prefix for this split's images
        base_prefix = self._s3_prefix.rstrip("/")
        split_prefix = f"{base_prefix}/images/{split}/"

        gs = self.args  # Ultralytics training args namespace

        # Build (once) a single disk cache shared by the train and val datasets
        # so they don't run two independent LRU indexes over the same directory
        # (which would double the budget and cross-evict each other's files).
        if self._cache_dir is not None and self._shared_disk_cache is None:
            from app.services.lru_disk_cache import LruDiskCache

            self._shared_disk_cache = LruDiskCache(
                cache_dir=self._cache_dir,
                max_bytes=self._cache_max_bytes,
            )

        dataset = S3YoloDataset(
            img_path=img_path,
            imgsz=gs.imgsz,
            batch_size=batch or gs.batch,
            augment=mode == "train",
            hyp=gs,
            rect=gs.rect if mode == "val" else False,
            cache=False,  # we handle caching ourselves
            single_cls=gs.single_cls,
            stride=int(max(gs.stride if hasattr(gs, "stride") else 32, 32)),
            pad=0.5 if mode == "val" else 0.0,
            prefix=f"{mode}: ",
            task=gs.task,
            classes=gs.classes,
            data=self.data,
            fraction=gs.fraction if mode == "train" else 1.0,
            # S3-specific kwargs
            s3_client=self._s3_client,
            s3_client_factory=self._s3_client_factory,
            s3_bucket=self._s3_bucket,
            s3_prefix=split_prefix,
            local_labels_root=self._local_labels_root,
            split=split,
            s3_labels_prefix=(
                f"{base_prefix}/labels/{split}/"
                if self._s3_labels_prefix
                else None
            ),
            # Shared LRU instance (one budget across train+val).
            disk_cache=self._shared_disk_cache,
        )

        _logger.info(
            "Built S3YoloDataset for split=%s with %d images (cache=%s)",
            split,
            len(dataset),
            "enabled" if self._cache_dir else "disabled",
        )

        return dataset


def make_s3_pose_trainer(
    s3_client: Any = None,
    s3_bucket: str = "",
    s3_prefix: str = "",
    local_labels_root: str = "",
    s3_labels_prefix: str | None = None,
    cache_dir: str | None = None,
    cache_max_bytes: int = 2 * 1024**3,
    s3_client_factory: Any = None,
) -> type:
    """Create a configured :class:`S3PoseTrainer` subclass.

    Returns a *class* (not an instance) suitable for passing to
    ``model.train(trainer=...)``.

    Pass ``s3_client_factory`` (a zero-arg callable returning a fresh boto3
    client) so that forked DataLoader workers each build their own client;
    botocore clients are not fork-safe.  ``s3_client`` remains accepted for
    single-process / test use.
    """
    if not _ULTRALYTICS_AVAILABLE:
        raise ImportError(
            "ultralytics is required for S3PoseTrainer. "
            "Install it with: pip install ultralytics"
        )

    class _Configured(S3PoseTrainer):
        _s3_client = s3_client
        _s3_client_factory = s3_client_factory
        _s3_bucket = s3_bucket
        _s3_prefix = s3_prefix
        _local_labels_root = local_labels_root
        _s3_labels_prefix = s3_labels_prefix
        _cache_dir = cache_dir
        _cache_max_bytes = cache_max_bytes

    _Configured.__name__ = "S3PoseTrainer"
    _Configured.__qualname__ = f"S3PoseTrainer[{s3_bucket}/{s3_prefix}]"
    return _Configured
