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

    def init_metrics(self, model: Any) -> None:
        """Initialise pose metrics, optionally overriding OKS sigmas from data.yaml.

        OKS (Object Keypoint Similarity) is the keypoint analogue of box IoU: it
        scores how close predicted keypoints are to ground truth, and pose-mAP is
        built by thresholding it. Each keypoint has a ``sigma`` (its tolerance —
        how much positional error is acceptable). Ultralytics' ``PoseValidator``
        defaults to COCO's 17 hand-tuned sigmas when there are 17 keypoints, else
        a uniform ``np.ones(nkpt)/nkpt``. For a non-17 custom keypoint set a
        uniform sigma treats every keypoint as equally easy to localise.

        If ``data.yaml`` provides a ``kpt_sigmas`` list (one value per keypoint),
        we use it so pose-mAP reflects per-keypoint tolerance. When absent or the
        length does not match the keypoint count, the Ultralytics default is kept
        (no behaviour change). Note: this affects *measurement only*, not training.
        """
        super().init_metrics(model)
        sigmas = (self.data or {}).get("kpt_sigmas")
        if not sigmas:
            return
        import numpy as np  # noqa: PLC0415

        try:
            arr = np.array(sigmas, dtype=float)
        except (TypeError, ValueError):
            _logger.warning(
                "data.yaml 'kpt_sigmas' is not numeric — keeping default OKS sigmas"
            )
            return
        if arr.shape == np.shape(self.sigma):
            self.sigma = arr
            _logger.info(
                "OKS sigmas overridden from data.yaml kpt_sigmas (%d values)", arr.size
            )
        else:
            _logger.warning(
                "data.yaml 'kpt_sigmas' has %d values but the model has %d "
                "keypoints — keeping default OKS sigmas",
                arr.size,
                np.size(self.sigma),
            )

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
