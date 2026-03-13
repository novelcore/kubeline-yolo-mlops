"""Tests for the LRU disk cache."""

from pathlib import Path

import pytest

from app.services.lru_disk_cache import LruDiskCache


class TestBasicPutGet:
    def test_put_then_get_returns_path(self, tmp_path: Path) -> None:
        cache = LruDiskCache(tmp_path / "cache", max_bytes=1024)
        cache.put("a.jpg", b"abc")
        result = cache.get("a.jpg")
        assert result is not None
        assert result.read_bytes() == b"abc"

    def test_get_miss_returns_none(self, tmp_path: Path) -> None:
        cache = LruDiskCache(tmp_path / "cache", max_bytes=1024)
        assert cache.get("nonexistent") is None

    def test_put_overwrites_existing_key(self, tmp_path: Path) -> None:
        cache = LruDiskCache(tmp_path / "cache", max_bytes=1024)
        cache.put("a.jpg", b"old")
        cache.put("a.jpg", b"new")
        path = cache.get("a.jpg")
        assert path is not None
        assert path.read_bytes() == b"new"

    def test_current_bytes_tracks_size(self, tmp_path: Path) -> None:
        cache = LruDiskCache(tmp_path / "cache", max_bytes=1024)
        assert cache.current_bytes == 0
        cache.put("a.jpg", b"12345")
        assert cache.current_bytes == 5
        cache.put("b.jpg", b"67890")
        assert cache.current_bytes == 10


class TestLruEviction:
    def test_evicts_lru_when_over_budget(self, tmp_path: Path) -> None:
        """Oldest entry is evicted first when the cache is full."""
        cache = LruDiskCache(tmp_path / "cache", max_bytes=10)
        cache.put("a.jpg", b"12345")  # 5 bytes
        cache.put("b.jpg", b"12345")  # 5 bytes — now at 10
        cache.put("c.jpg", b"12345")  # 5 bytes — must evict a

        assert cache.get("a.jpg") is None  # evicted
        assert cache.get("b.jpg") is not None
        assert cache.get("c.jpg") is not None

    def test_access_promotes_to_mru(self, tmp_path: Path) -> None:
        """Accessing an entry makes it most-recently-used (not evicted next)."""
        cache = LruDiskCache(tmp_path / "cache", max_bytes=15)
        cache.put("a.jpg", b"12345")  # 5
        cache.put("b.jpg", b"12345")  # 5
        cache.put("c.jpg", b"12345")  # 5 — full at 15

        # Access "a" to promote it
        cache.get("a.jpg")

        # Adding "d" should evict "b" (LRU), not "a"
        cache.put("d.jpg", b"12345")

        assert cache.get("a.jpg") is not None
        assert cache.get("b.jpg") is None  # evicted
        assert cache.get("c.jpg") is not None
        assert cache.get("d.jpg") is not None

    def test_budget_enforced_exact(self, tmp_path: Path) -> None:
        """Cache never exceeds max_bytes."""
        cache = LruDiskCache(tmp_path / "cache", max_bytes=20)
        for i in range(10):
            cache.put(f"file_{i}.jpg", b"X" * 5)  # 5 bytes each
            assert cache.current_bytes <= 20


class TestScanExisting:
    def test_scan_on_init_picks_up_files(self, tmp_path: Path) -> None:
        """Files already on disk are tracked and count against the budget."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "existing.jpg").write_bytes(b"hello")

        cache = LruDiskCache(cache_dir, max_bytes=1024)
        assert cache.current_bytes == 5

    def test_scan_evicts_if_over_budget(self, tmp_path: Path) -> None:
        """If existing files exceed the budget, the oldest are evicted on init."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        import time

        # Write files with distinct mtimes
        (cache_dir / "old.jpg").write_bytes(b"X" * 10)
        time.sleep(0.05)
        (cache_dir / "new.jpg").write_bytes(b"Y" * 10)

        cache = LruDiskCache(cache_dir, max_bytes=15)
        # "old.jpg" should have been evicted
        assert not (cache_dir / "old.jpg").exists()
        assert (cache_dir / "new.jpg").exists()
        assert cache.current_bytes == 10


class TestMetrics:
    def test_hit_miss_eviction_counts(self, tmp_path: Path) -> None:
        cache = LruDiskCache(tmp_path / "cache", max_bytes=10)
        cache.put("a.jpg", b"12345")
        cache.get("a.jpg")  # hit
        cache.get("miss1")  # miss
        cache.get("miss2")  # miss
        cache.put("b.jpg", b"12345")
        cache.put("c.jpg", b"12345")  # evicts a

        hits, misses, evictions = cache.reset_metrics()
        assert hits == 1
        assert misses == 2
        assert evictions == 1

    def test_reset_clears_metrics(self, tmp_path: Path) -> None:
        cache = LruDiskCache(tmp_path / "cache", max_bytes=1024)
        cache.put("a.jpg", b"data")
        cache.get("a.jpg")
        cache.get("nope")
        cache.reset_metrics()

        hits, misses, evictions = cache.reset_metrics()
        assert hits == 0
        assert misses == 0
        assert evictions == 0
