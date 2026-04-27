"""LRU prediction cache with TTL-based expiration.

Caches model predictions keyed by a hash of the input features.
Avoids redundant computation when the same input is seen repeatedly.
Uses an ordered dict for O(1) LRU eviction and per-entry TTL tracking.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np


@dataclass
class CacheEntry:
    """A single cached prediction.

    Attributes:
        key: Hash key of the input.
        value: Cached prediction (numpy array).
        created_at: When the entry was cached.
        ttl_s: Time-to-live in seconds.
        hits: Number of times this entry has been served.
    """
    key: str
    value: np.ndarray
    created_at: float = field(default_factory=time.time)
    ttl_s: float = 300.0
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if this entry has exceeded its TTL."""
        return (time.time() - self.created_at) > self.ttl_s


@dataclass
class CacheStats:
    """Aggregate cache statistics.

    Attributes:
        hits: Total cache hits.
        misses: Total cache misses.
        evictions: Total LRU evictions.
        expirations: Total TTL expirations.
        current_size: Current number of entries.
    """
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    current_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class PredictionCache:
    """LRU cache for model predictions with TTL expiration.

    Keys are SHA-256 hashes of the input numpy array bytes.
    The cache maintains insertion order for LRU eviction and
    checks TTL on every access.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_s: float = 300.0,
    ) -> None:
        """Initialize the prediction cache.

        Args:
            max_size: Maximum number of entries before LRU eviction.
            default_ttl_s: Default time-to-live for entries in seconds.
        """
        self._max_size = max_size
        self._default_ttl_s = default_ttl_s
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()

    @staticmethod
    def _hash_input(inputs: np.ndarray) -> str:
        """Compute a cache key from input features.

        Args:
            inputs: Input numpy array.

        Returns:
            Hex-encoded SHA-256 hash of the array bytes.
        """
        return hashlib.sha256(inputs.tobytes()).hexdigest()

    def get(self, inputs: np.ndarray) -> Optional[np.ndarray]:
        """Look up a cached prediction.

        Returns the cached value if present and not expired.
        Moves the entry to the end (most recently used) on hit.

        Args:
            inputs: Input feature array.

        Returns:
            Cached prediction array, or None on miss/expiration.
        """
        key = self._hash_input(inputs)

        if key not in self._cache:
            self._stats.misses += 1
            return None

        entry = self._cache[key]

        if entry.is_expired:
            del self._cache[key]
            self._stats.expirations += 1
            self._stats.misses += 1
            self._stats.current_size = len(self._cache)
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.hits += 1
        self._stats.hits += 1
        return entry.value

    def put(
        self,
        inputs: np.ndarray,
        prediction: np.ndarray,
        ttl_s: Optional[float] = None,
    ) -> str:
        """Cache a prediction result.

        If the cache is full, evicts the least recently used entry.

        Args:
            inputs: Input feature array (used to compute the key).
            prediction: Model prediction to cache.
            ttl_s: Optional TTL override for this entry.

        Returns:
            The cache key for this entry.
        """
        key = self._hash_input(inputs)

        if key in self._cache:
            # Update existing entry
            self._cache.move_to_end(key)
            self._cache[key] = CacheEntry(
                key=key,
                value=prediction,
                ttl_s=ttl_s or self._default_ttl_s,
            )
            self._stats.current_size = len(self._cache)
            return key

        # Evict LRU if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
            self._stats.evictions += 1

        self._cache[key] = CacheEntry(
            key=key,
            value=prediction,
            ttl_s=ttl_s or self._default_ttl_s,
        )
        self._stats.current_size = len(self._cache)
        return key

    def invalidate(self, inputs: np.ndarray) -> bool:
        """Remove a specific entry from the cache.

        Args:
            inputs: Input feature array.

        Returns:
            True if the entry was found and removed.
        """
        key = self._hash_input(inputs)
        if key in self._cache:
            del self._cache[key]
            self._stats.current_size = len(self._cache)
            return True
        return False

    def clear(self) -> int:
        """Remove all entries from the cache.

        Returns:
            Number of entries that were cleared.
        """
        count = len(self._cache)
        self._cache.clear()
        self._stats.current_size = 0
        return count

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        expired_keys = [
            k for k, v in self._cache.items() if v.is_expired
        ]
        for k in expired_keys:
            del self._cache[k]
            self._stats.expirations += 1
        self._stats.current_size = len(self._cache)
        return len(expired_keys)

    @property
    def stats(self) -> CacheStats:
        """Current cache statistics."""
        self._stats.current_size = len(self._cache)
        return self._stats


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    cache = PredictionCache(max_size=5, default_ttl_s=2.0)

    # Populate cache
    inputs_list = [rng.standard_normal(10) for _ in range(5)]
    for i, inp in enumerate(inputs_list):
        pred = rng.standard_normal(3)
        cache.put(inp, pred)
        print(f"Cached input {i}: key={cache._hash_input(inp)[:12]}...")

    # Hit existing entries
    for i, inp in enumerate(inputs_list):
        result = cache.get(inp)
        status = "HIT" if result is not None else "MISS"
        print(f"Lookup input {i}: {status}")

    # Miss on new input
    result = cache.get(rng.standard_normal(10))
    print(f"New input lookup: {'HIT' if result is not None else 'MISS'}")

    # Test LRU eviction
    for i in range(3):
        cache.put(rng.standard_normal(10), rng.standard_normal(3))
    print(f"\nAfter adding 3 more (max_size=5):")
    print(f"  cache size: {cache.stats.current_size}")
    print(f"  evictions: {cache.stats.evictions}")

    # Stats summary
    s = cache.stats
    print(f"\nCache stats:")
    print(f"  hits={s.hits}, misses={s.misses}, "
          f"hit_rate={s.hit_rate:.1%}, evictions={s.evictions}")
