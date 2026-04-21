"""PoseValidator subclass that builds S3YoloDataset instances.

Mirrors :mod:`app.services.s3_pose_trainer` for the validation path.  When
``model.val(split=...)`` is called with ``validator=S3PoseValidator``,
Ultralytics instantiates this subclass and uses it to build the dataset.
Overriding :meth:`build_dataset` lets us stream images (and optionally
labels) from S3 for the requested split, reusing the same
:class:`S3YoloDataset` used during training.

Configuration is injected via the :func:`make_s3_pose_validator` factory,
which bakes S3 connection details and cache parameters into a dynamically
created subclass — avoiding any global state.
"""

import logging
from typing import Any

_logger = logging.getLogger(__name__)

try:
    from ultralytics.models.yolo.pose import PoseValidator as _PoseValidator

    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False
    _PoseValidator = object  # type: ignore[assignment,misc]


class S3PoseValidator(_PoseValidator):  # type: ignore[misc]
    """PoseValidator that builds S3-backed datasets.

    Class attributes ``_s3_client``, ``_s3_bucket``, ``_s3_prefix``,
    ``_local_labels_root``, ``_cache_dir``, and ``_cache_max_bytes`` must
    be set on the *class* (not the instance) before construction.  Use
    :func:`make_s3_pose_validator` to create a properly configured
    subclass.
    """

    # Filled by the factory
    _s3_client: Any = None
    _s3_bucket: str = ""
    _s3_prefix: str = ""
    _local_labels_root: str = ""
    _s3_labels_prefix: str | None = None
    _cache_dir: str | None = None
    _cache_max_bytes: int = 2 * 1024**3

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None) -> Any:
        """Override to return an :class:`S3YoloDataset` for validation.

        Parameters
        ----------
        img_path:
            Path string Ultralytics passes (e.g. ``"images/test"``).  Used
            only to extract the split name.
        mode:
            Always ``"val"`` for validation.
        batch:
            Batch size (forwarded to the parent for bookkeeping).
        """
        from app.services.s3_dataset import S3YoloDataset

        # Determine split from img_path first (more reliable), then mode
        split = "val"
        for candidate in ("train", "val", "test"):
            if candidate in img_path:
                split = candidate
                break

        base_prefix = self._s3_prefix.rstrip("/")
        split_prefix = f"{base_prefix}/images/{split}/"

        gs = self.args  # Ultralytics args namespace

        dataset = S3YoloDataset(
            img_path=img_path,
            imgsz=gs.imgsz,
            batch_size=batch or gs.batch,
            augment=False,
            hyp=gs,
            rect=gs.rect,
            cache=False,  # we handle caching ourselves
            single_cls=gs.single_cls,
            stride=int(max(gs.stride if hasattr(gs, "stride") else 32, 32)),
            pad=0.5,
            prefix=f"{mode}: ",
            task=gs.task,
            classes=gs.classes,
            data=self.data,
            fraction=1.0,
            # S3-specific kwargs
            s3_client=self._s3_client,
            s3_bucket=self._s3_bucket,
            s3_prefix=split_prefix,
            local_labels_root=self._local_labels_root,
            split=split,
            s3_labels_prefix=(
                f"{base_prefix}/labels/{split}/"
                if self._s3_labels_prefix
                else None
            ),
            cache_dir=self._cache_dir,
            cache_max_bytes=self._cache_max_bytes,
        )

        _logger.info(
            "Built S3YoloDataset for split=%s with %d images (cache=%s)",
            split,
            len(dataset),
            "enabled" if self._cache_dir else "disabled",
        )

        return dataset


def make_s3_pose_validator(
    s3_client: Any,
    s3_bucket: str,
    s3_prefix: str,
    local_labels_root: str,
    s3_labels_prefix: str | None = None,
    cache_dir: str | None = None,
    cache_max_bytes: int = 2 * 1024**3,
) -> type:
    """Create a configured :class:`S3PoseValidator` subclass.

    Returns a *class* (not an instance) suitable for passing to
    ``model.val(validator=...)``.
    """
    if not _ULTRALYTICS_AVAILABLE:
        raise ImportError(
            "ultralytics is required for S3PoseValidator. "
            "Install it with: pip install ultralytics"
        )

    class _Configured(S3PoseValidator):
        _s3_client = s3_client
        _s3_bucket = s3_bucket
        _s3_prefix = s3_prefix
        _local_labels_root = local_labels_root
        _s3_labels_prefix = s3_labels_prefix
        _cache_dir = cache_dir
        _cache_max_bytes = cache_max_bytes

    _Configured.__name__ = "S3PoseValidator"
    _Configured.__qualname__ = f"S3PoseValidator[{s3_bucket}/{s3_prefix}]"
    return _Configured
