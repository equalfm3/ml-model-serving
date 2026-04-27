"""Real-time metrics: latency percentiles, throughput, and error rate.

Collects and aggregates inference metrics using sliding time windows.
Tracks latency distributions (p50, p95, p99), request throughput,
error rates, and prediction value distributions.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


@dataclass
class MetricPoint:
    """A single metric observation.

    Attributes:
        value: The metric value.
        timestamp: When the observation was recorded.
        tags: Optional key-value tags for grouping.
    """
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Computed latency statistics.

    Attributes:
        p50: Median latency in milliseconds.
        p95: 95th percentile latency.
        p99: 99th percentile latency.
        mean: Mean latency.
        count: Number of observations.
    """
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    mean: float = 0.0
    count: int = 0


class MetricsCollector:
    """Collects and aggregates real-time inference metrics.

    Uses sliding time windows to compute statistics over recent
    observations. Supports multiple named metric streams.
    """

    def __init__(self, window_seconds: float = 300.0) -> None:
        """Initialize the metrics collector.

        Args:
            window_seconds: Size of the sliding window in seconds.
        """
        self._window_s = window_seconds
        self._streams: Dict[str, Deque[MetricPoint]] = {}

    def _get_stream(self, name: str) -> Deque[MetricPoint]:
        """Get or create a metric stream.

        Args:
            name: Stream name.

        Returns:
            The deque for this stream.
        """
        if name not in self._streams:
            self._streams[name] = deque()
        return self._streams[name]

    def _prune(self, stream: Deque[MetricPoint]) -> None:
        """Remove observations outside the sliding window.

        Args:
            stream: The metric stream to prune.
        """
        cutoff = time.time() - self._window_s
        while stream and stream[0].timestamp < cutoff:
            stream.popleft()

    def record(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric observation.

        Args:
            name: Metric name (e.g. 'latency_ms', 'error').
            value: The metric value.
            tags: Optional tags for grouping.
        """
        stream = self._get_stream(name)
        stream.append(MetricPoint(value=value, tags=tags or {}))

    def record_request(
        self,
        model_name: str,
        version: int,
        latency_ms: float,
        is_error: bool = False,
    ) -> None:
        """Convenience: record a complete request observation.

        Records latency, error, and throughput metrics with model tags.

        Args:
            model_name: Name of the model.
            version: Model version.
            latency_ms: Request latency in milliseconds.
            is_error: Whether the request resulted in an error.
        """
        tags = {"model": model_name, "version": str(version)}
        self.record("latency_ms", latency_ms, tags)
        self.record("request_count", 1.0, tags)
        if is_error:
            self.record("error_count", 1.0, tags)

    def get_latency_stats(
        self,
        model_name: Optional[str] = None,
        version: Optional[int] = None,
    ) -> LatencyStats:
        """Compute latency percentiles over the sliding window.

        Args:
            model_name: Optional filter by model name.
            version: Optional filter by version.

        Returns:
            LatencyStats with p50, p95, p99, mean.
        """
        stream = self._get_stream("latency_ms")
        self._prune(stream)

        values = self._filter_values(stream, model_name, version)
        if not values:
            return LatencyStats()

        values.sort()
        n = len(values)
        return LatencyStats(
            p50=values[int(n * 0.50)],
            p95=values[int(n * 0.95)] if n > 1 else values[0],
            p99=values[int(n * 0.99)] if n > 1 else values[0],
            mean=sum(values) / n,
            count=n,
        )

    def get_throughput(
        self,
        model_name: Optional[str] = None,
        version: Optional[int] = None,
    ) -> float:
        """Compute requests per second over the sliding window.

        Args:
            model_name: Optional filter by model name.
            version: Optional filter by version.

        Returns:
            Requests per second.
        """
        stream = self._get_stream("request_count")
        self._prune(stream)

        values = self._filter_values(stream, model_name, version)
        if not values:
            return 0.0

        return len(values) / self._window_s

    def get_error_rate(
        self,
        model_name: Optional[str] = None,
        version: Optional[int] = None,
    ) -> float:
        """Compute error rate over the sliding window.

        Args:
            model_name: Optional filter by model name.
            version: Optional filter by version.

        Returns:
            Error rate as a fraction (0-1).
        """
        req_stream = self._get_stream("request_count")
        err_stream = self._get_stream("error_count")
        self._prune(req_stream)
        self._prune(err_stream)

        total = len(self._filter_values(req_stream, model_name, version))
        errors = len(self._filter_values(err_stream, model_name, version))

        return errors / total if total > 0 else 0.0

    def _filter_values(
        self,
        stream: Deque[MetricPoint],
        model_name: Optional[str],
        version: Optional[int],
    ) -> List[float]:
        """Filter metric points by model name and version.

        Args:
            stream: The metric stream.
            model_name: Optional model name filter.
            version: Optional version filter.

        Returns:
            List of matching metric values.
        """
        values = []
        for point in stream:
            if model_name and point.tags.get("model") != model_name:
                continue
            if version is not None and point.tags.get("version") != str(version):
                continue
            values.append(point.value)
        return values

    def get_summary(
        self,
        model_name: Optional[str] = None,
        version: Optional[int] = None,
    ) -> Dict:
        """Get a complete metrics summary.

        Args:
            model_name: Optional filter by model name.
            version: Optional filter by version.

        Returns:
            Dict with latency stats, throughput, and error rate.
        """
        latency = self.get_latency_stats(model_name, version)
        return {
            "latency": {
                "p50": latency.p50,
                "p95": latency.p95,
                "p99": latency.p99,
                "mean": latency.mean,
            },
            "throughput_rps": self.get_throughput(model_name, version),
            "error_rate": self.get_error_rate(model_name, version),
            "observation_count": latency.count,
        }


if __name__ == "__main__":
    import random

    collector = MetricsCollector(window_seconds=60.0)
    rng = random.Random(42)

    # Simulate requests for two model versions
    for _ in range(500):
        version = 1 if rng.random() < 0.9 else 2
        latency = rng.gauss(45 if version == 1 else 55, 10)
        is_error = rng.random() < (0.02 if version == 1 else 0.05)
        collector.record_request("sentiment", version, latency, is_error)

    # Print summary for each version
    for v in [1, 2]:
        summary = collector.get_summary("sentiment", v)
        print(f"Model sentiment v{v}:")
        print(f"  Latency: p50={summary['latency']['p50']:.1f}ms, "
              f"p95={summary['latency']['p95']:.1f}ms, "
              f"p99={summary['latency']['p99']:.1f}ms")
        print(f"  Throughput: {summary['throughput_rps']:.1f} rps")
        print(f"  Error rate: {summary['error_rate']:.1%}")
        print(f"  Observations: {summary['observation_count']}")
        print()
