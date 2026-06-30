"""Thread-safe bounded LRU disk cache for S3 image streaming.

Images fetched from S3 are written to a local directory and tracked with an
``OrderedDict`` for O(1) LRU eviction.  When the total size on disk exceeds
``max_bytes``, the least-recently-used entries are evicted until the budget is
satisfied.

Design notes
------------
* **Digest-keyed index.** Entries are indexed by the md5 hex digest of the
  cache key (not the raw key). The on-disk filename is ``<digest><suffix>``, so
  a file left behind by a previous run can be re-associated with its key on the
  next ``get(key)`` — restart reuse actually works, and re-fetching an existing
  key writes to the same path without double-counting bytes.
* **Atomic writes.** Bytes are written to a temporary file and ``os.replace``-d
  into place. A crash mid-write leaves only a ``*.tmp`` file (skipped on scan),
  never a truncated image at the real path.
"""

import hashlib
import logging
import os
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple

_logger = logging.getLogger(__name__)

# Suffix used for in-progress (pre-rename) cache writes.
_TMP_SUFFIX = ".tmp"


class _CacheEntry(NamedTuple):
    size: int
    path: Path


class LruDiskCache:
    """Bounded LRU disk cache backed by an ``OrderedDict``.

    Parameters
    ----------
    cache_dir:
        Directory where cached files are stored.  Created if absent.
    max_bytes:
        Maximum total size of cached files on disk.  Defaults to 2 GiB.
    """

    def __init__(self, cache_dir: str | Path, max_bytes: int = 2 * 1024**3) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_bytes

        # OrderedDict: digest -> _CacheEntry (MRU at the end)
        self._index: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._current_bytes = 0
        self._lock = threading.Lock()

        # Metrics (cumulative, reset on read)
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        self._scan_existing()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Path | None:
        """Return the local path for *key* on cache hit, or ``None`` on miss.

        On a hit the entry is promoted to most-recently-used.
        """
        digest = self._key_to_digest(key)
        with self._lock:
            entry = self._index.get(digest)
            if entry is not None:
                self._index.move_to_end(digest)
                self._hits += 1
                return entry.path
            self._misses += 1
            return None

    def put(self, key: str, data: bytes) -> Path:
        """Write *data* to disk under *key*, evicting LRU entries if needed.

        Returns the local ``Path`` to the written file.  The write is atomic:
        a crash mid-write never leaves a truncated file at the returned path.
        """
        digest = self._key_to_digest(key)
        file_path = self._key_to_path(key)
        size = len(data)

        with self._lock:
            # If the digest already exists, drop the old size first so re-fetching
            # the same key does not double-count bytes.
            old = self._index.pop(digest, None)
            if old is not None:
                self._current_bytes -= old.size

            # Evict LRU entries until there is room.
            while self._current_bytes + size > self._max_bytes and self._index:
                self._evict_lru()

            self._atomic_write(file_path, data)

            self._index[digest] = _CacheEntry(size=size, path=file_path)
            self._current_bytes += size

        return file_path

    def reset_metrics(self) -> tuple[int, int, int]:
        """Atomically read and clear ``(hits, misses, evictions)``."""
        with self._lock:
            result = (self._hits, self._misses, self._evictions)
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            return result

    @property
    def current_bytes(self) -> int:
        """Total bytes currently held in the cache."""
        with self._lock:
            return self._current_bytes

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _key_to_digest(key: str) -> str:
        """Return the md5 hex digest used to index *key*."""
        return hashlib.md5(key.encode()).hexdigest()  # noqa: S324

    def _key_to_path(self, key: str) -> Path:
        """Derive a filesystem path from a cache key.

        The filename is ``<md5-digest><suffix>``; the original extension is
        preserved for debugging convenience and so OpenCV-friendly suffixes
        survive a round-trip.
        """
        suffix = Path(key).suffix
        return self._cache_dir / f"{self._key_to_digest(key)}{suffix}"

    def _atomic_write(self, file_path: Path, data: bytes) -> None:
        """Write *data* to *file_path* atomically via a temp file + ``os.replace``.

        Caller must hold ``_lock``.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(file_path.parent),
            prefix=f"{file_path.name}.",
            suffix=_TMP_SUFFIX,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp_path, file_path)  # atomic on the same filesystem
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def _evict_lru(self) -> None:
        """Evict the least-recently-used entry.  Caller must hold ``_lock``."""
        _digest, entry = self._index.popitem(last=False)
        try:
            entry.path.unlink(missing_ok=True)
        except OSError:
            pass
        self._current_bytes -= entry.size
        self._evictions += 1

    def _scan_existing(self) -> None:
        """Populate the index from files already present in ``cache_dir``.

        Files are indexed by their stem (which equals the md5 digest for files
        this cache wrote), so a re-requested key hits the leftover file.  Files
        are added in modification-time order (oldest = LRU).  Leftover ``*.tmp``
        partial writes from a crashed run are removed rather than indexed.
        """
        try:
            entries = list(self._cache_dir.iterdir())
        except OSError:
            return

        files = sorted(entries, key=lambda p: p.stat().st_mtime)
        for f in files:
            if not f.is_file():
                continue
            # Discard partial writes left by a crash mid-``put``.
            if f.name.endswith(_TMP_SUFFIX):
                f.unlink(missing_ok=True)
                continue
            size = f.stat().st_size
            digest = f.stem  # == md5 digest for files written by this cache
            self._index[digest] = _CacheEntry(size=size, path=f)
            self._current_bytes += size

        # Enforce budget for pre-existing files
        while self._current_bytes > self._max_bytes and self._index:
            self._evict_lru()

        if self._index:
            _logger.debug(
                "LruDiskCache: scanned %d existing files (%d bytes) in %s",
                len(self._index),
                self._current_bytes,
                self._cache_dir,
            )
