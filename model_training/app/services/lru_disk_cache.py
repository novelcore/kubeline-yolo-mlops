"""Thread-safe bounded LRU disk cache for S3 image streaming.

Images fetched from S3 are written to a local directory and tracked with an
``OrderedDict`` for O(1) LRU eviction.  When the total size on disk exceeds
``max_bytes``, the least-recently-used entries are evicted until the budget is
satisfied.

The cache directory is scanned on construction so that a restarted process can
reuse files left behind by a previous run.
"""

import hashlib
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple

_logger = logging.getLogger(__name__)


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

        # OrderedDict: key -> _CacheEntry (MRU at the end)
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
        with self._lock:
            if key in self._index:
                self._index.move_to_end(key)
                self._hits += 1
                return self._index[key].path
            self._misses += 1
            return None

    def put(self, key: str, data: bytes) -> Path:
        """Write *data* to disk under *key*, evicting LRU entries if needed.

        Returns the local ``Path`` to the written file.
        """
        file_path = self._key_to_path(key)
        size = len(data)

        with self._lock:
            # If key already exists, remove old entry first
            if key in self._index:
                self._current_bytes -= self._index.pop(key).size

            # Evict LRU entries until there is room
            while self._current_bytes + size > self._max_bytes and self._index:
                self._evict_lru()

            # Write file
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(data)

            # Track
            self._index[key] = _CacheEntry(size=size, path=file_path)
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

    def _key_to_path(self, key: str) -> Path:
        """Derive a filesystem path from a cache key.

        Uses an MD5 hash to avoid issues with long or special-character keys.
        The original extension is preserved for debugging convenience.
        """
        suffix = Path(key).suffix
        digest = hashlib.md5(key.encode()).hexdigest()  # noqa: S324
        return self._cache_dir / f"{digest}{suffix}"

    def _evict_lru(self) -> None:
        """Evict the least-recently-used entry.  Caller must hold ``_lock``."""
        key, entry = self._index.popitem(last=False)
        try:
            entry.path.unlink(missing_ok=True)
        except OSError:
            pass
        self._current_bytes -= entry.size
        self._evictions += 1

    def _scan_existing(self) -> None:
        """Populate the index from files already present in ``cache_dir``.

        Files are added in modification-time order (oldest = LRU).
        """
        files = sorted(self._cache_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        for f in files:
            if not f.is_file():
                continue
            size = f.stat().st_size
            # Use a pseudo-key for scanned files (the original key is unknown).
            # The actual path is stored in the entry so eviction works correctly.
            pseudo_key = f"__scanned__{f.name}"
            self._index[pseudo_key] = _CacheEntry(size=size, path=f)
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
