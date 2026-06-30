"""Pytest bootstrap shims.

Several ``opencv-python`` builds ship a ``cv2/typing/__init__.py`` that, when
imported *before* OpenCV finishes its lazy submodule bootstrap, raises
``AttributeError: module 'cv2.gapi.wip.draw' has no attribute 'Text'`` (and a
similar ``cv2.dnn.DictValue`` error).  Standalone this never happens because a
normal ``import cv2`` completes the bootstrap first; under pytest, collection
machinery can trigger ``import cv2.typing`` mid-bootstrap.

Importing cv2 (and its typing module) here — before any test module is
collected — forces the bootstrap to complete up front, so the streaming-dataset
tests that ``import cv2`` collect cleanly.  This only affects the test harness;
runtime training imports cv2 normally.
"""

try:  # pragma: no cover - import-time guard only
    import cv2  # noqa: F401

    import cv2.typing  # noqa: F401
except Exception:  # noqa: BLE001 - never let the shim break collection
    pass
