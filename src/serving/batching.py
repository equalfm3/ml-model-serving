"""Dynamic request batching with configurable max size and timeout.

Collects incoming inference requests into batches, triggering execution
when either the batch is full or a deadline expires. This amortizes
per-request overhead — especially effective on GPUs where the cost of
a forward pass is nearly constant across batch sizes.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Generic, List, Optional, TypeVar

import numpy as np

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchConfig:
    """Configuration for the dynamic batcher.

    Attributes:
        max_batch_size: Maximum number of requests per batch.
        max_wait_ms: Maximum time to wait for a full batch (milliseconds).
        adaptive: If True, adjust batch size based on throughput.
    """
    max_batch_size: int = 32
    max_wait_ms: float = 50.0
    adaptive: bool = False


@dataclass
class BatchItem:
    """A single item in the batch queue.

    Attributes:
        data: The request payload (typically a numpy array).
        request_id: Unique identifier for tracking.
        arrived_at: Timestamp when the item entered the queue.
    """
    data: np.ndarray
    request_id: str = ""
    arrived_at: float = field(default_factory=time.time)


@dataclass
class BatchResult:
    """Result of processing a batch.

    Attributes:
        items: Individual results corresponding to each input.
        batch_size: Number of items in the batch.
        wait_time_ms: Time spent waiting for the batch to fill.
        inference_time_ms: Time spent on model inference.
    """
    items: List[np.ndarray]
    batch_size: int
    wait_time_ms: float
    inference_time_ms: float


class DynamicBatcher:
    """Collects requests into batches with deadline-aware scheduling.

    The batcher maintains a queue of incoming requests and triggers
    batch execution when either condition is met:
      - The queue reaches max_batch_size
      - The oldest request has waited max_wait_ms

    This ensures no individual request waits longer than its SLA.
    """

    def __init__(
        self,
        process_fn: Callable[[np.ndarray], np.ndarray],
        config: Optional[BatchConfig] = None,
    ) -> None:
        """Initialize the dynamic batcher.

        Args:
            process_fn: Function that processes a batch (numpy array)
                        and returns a batch of results.
            config: Batching configuration.
        """
        self._process_fn = process_fn
        self._config = config or BatchConfig()
        self._queue: Deque[BatchItem] = deque()
        self._lock = threading.Lock()
        self._stats = {"batches_processed": 0, "items_processed": 0}

    @property
    def config(self) -> BatchConfig:
        """Current batch configuration."""
        return self._config

    @property
    def queue_size(self) -> int:
        """Number of items currently waiting in the queue."""
        return len(self._queue)

    def add(self, item: BatchItem) -> None:
        """Add a request to the batch queue.

        Args:
            item: The batch item to enqueue.
        """
        with self._lock:
            self._queue.append(item)

    def should_trigger(self) -> bool:
        """Check if a batch should be triggered.

        Returns True if the queue is full or the oldest item has
        exceeded the maximum wait time.

        Returns:
            True if batch processing should be triggered.
        """
        if not self._queue:
            return False

        if len(self._queue) >= self._config.max_batch_size:
            return True

        oldest = self._queue[0]
        wait_ms = (time.time() - oldest.arrived_at) * 1000
        return wait_ms >= self._config.max_wait_ms

    def _collect_batch(self) -> List[BatchItem]:
        """Collect up to max_batch_size items from the queue.

        Returns:
            List of BatchItems to process.
        """
        with self._lock:
            batch_size = min(len(self._queue), self._config.max_batch_size)
            return [self._queue.popleft() for _ in range(batch_size)]

    def process_batch(self) -> Optional[BatchResult]:
        """Process the current batch if trigger conditions are met.

        Collects items from the queue, stacks them into a single
        numpy array, runs the process function, and splits results.

        Returns:
            BatchResult if a batch was processed, None otherwise.
        """
        if not self.should_trigger():
            return None

        items = self._collect_batch()
        if not items:
            return None

        wait_time_ms = (time.time() - items[0].arrived_at) * 1000
        batch_input = np.stack([item.data for item in items])

        start = time.perf_counter()
        batch_output = self._process_fn(batch_input)
        inference_ms = (time.perf_counter() - start) * 1000

        results = [batch_output[i] for i in range(len(items))]

        self._stats["batches_processed"] += 1
        self._stats["items_processed"] += len(items)

        return BatchResult(
            items=results,
            batch_size=len(items),
            wait_time_ms=wait_time_ms,
            inference_time_ms=inference_ms,
        )

    def flush(self) -> List[BatchResult]:
        """Process all remaining items in the queue.

        Drains the queue by processing batches until empty.

        Returns:
            List of BatchResults from all processed batches.
        """
        results: List[BatchResult] = []
        while self._queue:
            items = self._collect_batch()
            if not items:
                break
            batch_input = np.stack([item.data for item in items])
            start = time.perf_counter()
            batch_output = self._process_fn(batch_input)
            inference_ms = (time.perf_counter() - start) * 1000

            results.append(BatchResult(
                items=[batch_output[i] for i in range(len(items))],
                batch_size=len(items),
                wait_time_ms=(time.time() - items[0].arrived_at) * 1000,
                inference_time_ms=inference_ms,
            ))
            self._stats["batches_processed"] += 1
            self._stats["items_processed"] += len(items)
        return results

    def get_stats(self) -> dict:
        """Return batching statistics.

        Returns:
            Dict with batches_processed and items_processed counts.
        """
        return dict(self._stats)


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Simulate a model: simple matrix multiply
    W = rng.standard_normal((8, 3))

    def model_fn(batch: np.ndarray) -> np.ndarray:
        """Simulated model inference."""
        time.sleep(0.01)  # Simulate compute
        return batch @ W

    config = BatchConfig(max_batch_size=8, max_wait_ms=100.0)
    batcher = DynamicBatcher(model_fn, config)

    # Add requests one at a time
    for i in range(20):
        item = BatchItem(
            data=rng.standard_normal(8),
            request_id=f"req-{i:03d}",
        )
        batcher.add(item)

        result = batcher.process_batch()
        if result:
            print(f"Batch triggered: size={result.batch_size}, "
                  f"wait={result.wait_time_ms:.1f}ms, "
                  f"inference={result.inference_time_ms:.1f}ms")

    # Flush remaining
    remaining = batcher.flush()
    for r in remaining:
        print(f"Flushed batch: size={r.batch_size}")

    print(f"\nStats: {batcher.get_stats()}")
