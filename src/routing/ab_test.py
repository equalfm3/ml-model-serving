"""A/B test configuration, user bucketing, and metric collection.

Manages A/B experiments between model versions. Users are deterministically
assigned to groups using a hash of their user ID, ensuring consistent
experience across requests. Collects per-group metrics for comparison.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExperimentStatus(Enum):
    """Status of an A/B experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class Variant:
    """A variant (arm) in an A/B experiment.

    Attributes:
        name: Variant identifier (e.g. 'control', 'treatment').
        model_version: Model version to serve for this variant.
        traffic_pct: Percentage of traffic allocated (0-100).
    """
    name: str
    model_version: int
    traffic_pct: float


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment.

    Attributes:
        experiment_id: Unique experiment identifier.
        model_name: Name of the model being tested.
        variants: List of experiment variants.
        status: Current experiment status.
        created_at: When the experiment was created.
        min_sample_size: Minimum observations per variant before analysis.
    """
    experiment_id: str
    model_name: str
    variants: List[Variant]
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: float = field(default_factory=time.time)
    min_sample_size: int = 100


@dataclass
class VariantMetrics:
    """Collected metrics for a single variant.

    Attributes:
        observations: Number of requests served.
        successes: Number of successful predictions.
        total_latency_ms: Sum of latencies for mean calculation.
        latencies: Individual latency measurements for percentiles.
        predictions: Collected prediction values for distribution analysis.
    """
    observations: int = 0
    successes: int = 0
    total_latency_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)
    predictions: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Fraction of successful predictions."""
        return self.successes / self.observations if self.observations > 0 else 0.0

    @property
    def mean_latency_ms(self) -> float:
        """Mean latency in milliseconds."""
        return self.total_latency_ms / self.observations if self.observations > 0 else 0.0


class ABTestManager:
    """Manages A/B experiments between model versions.

    Handles user bucketing, metric collection, and basic statistical
    analysis to determine if one variant is significantly better.
    """

    def __init__(self) -> None:
        self._experiments: Dict[str, ExperimentConfig] = {}
        self._metrics: Dict[str, Dict[str, VariantMetrics]] = {}

    def create_experiment(self, config: ExperimentConfig) -> None:
        """Create a new A/B experiment.

        Args:
            config: Experiment configuration.

        Raises:
            ValueError: If traffic percentages don't sum to 100.
            ValueError: If experiment ID already exists.
        """
        total_pct = sum(v.traffic_pct for v in config.variants)
        if abs(total_pct - 100.0) > 0.01:
            raise ValueError(
                f"Traffic percentages must sum to 100, got {total_pct}"
            )
        if config.experiment_id in self._experiments:
            raise ValueError(
                f"Experiment {config.experiment_id} already exists"
            )

        self._experiments[config.experiment_id] = config
        self._metrics[config.experiment_id] = {
            v.name: VariantMetrics() for v in config.variants
        }

    def start_experiment(self, experiment_id: str) -> None:
        """Start a draft experiment.

        Args:
            experiment_id: The experiment to start.
        """
        exp = self._get_experiment(experiment_id)
        exp.status = ExperimentStatus.RUNNING

    def assign_variant(
        self, experiment_id: str, user_id: str
    ) -> Variant:
        """Assign a user to a variant using deterministic hashing.

        The same user_id always maps to the same variant within
        an experiment, ensuring consistent experience.

        Args:
            experiment_id: The experiment.
            user_id: The user to assign.

        Returns:
            The assigned Variant.
        """
        exp = self._get_experiment(experiment_id)
        bucket = self._hash_to_bucket(experiment_id, user_id)

        cumulative = 0.0
        for variant in exp.variants:
            cumulative += variant.traffic_pct
            if bucket < cumulative:
                return variant
        return exp.variants[-1]

    @staticmethod
    def _hash_to_bucket(experiment_id: str, user_id: str) -> float:
        """Hash experiment+user to a bucket in [0, 100).

        Args:
            experiment_id: Experiment identifier.
            user_id: User identifier.

        Returns:
            Float in [0, 100).
        """
        key = f"{experiment_id}:{user_id}"
        digest = hashlib.sha256(key.encode()).hexdigest()
        return int(digest[:8], 16) % 100

    def record_observation(
        self,
        experiment_id: str,
        variant_name: str,
        success: bool,
        latency_ms: float,
        prediction_value: Optional[float] = None,
    ) -> None:
        """Record an observation for a variant."""
        metrics = self._metrics[experiment_id][variant_name]
        metrics.observations += 1
        if success:
            metrics.successes += 1
        metrics.total_latency_ms += latency_ms
        metrics.latencies.append(latency_ms)
        if prediction_value is not None:
            metrics.predictions.append(prediction_value)

    def get_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get current experiment results with z-test for success rate difference."""
        exp = self._get_experiment(experiment_id)
        variant_results = {}

        for v in exp.variants:
            m = self._metrics[experiment_id][v.name]
            variant_results[v.name] = {
                "model_version": v.model_version,
                "observations": m.observations,
                "success_rate": m.success_rate,
                "mean_latency_ms": m.mean_latency_ms,
            }

        result: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "status": exp.status.value,
            "variants": variant_results,
        }

        # Z-test between first two variants if enough data
        if len(exp.variants) >= 2:
            m_a = self._metrics[experiment_id][exp.variants[0].name]
            m_b = self._metrics[experiment_id][exp.variants[1].name]
            if m_a.observations >= exp.min_sample_size and m_b.observations >= exp.min_sample_size:
                z, p = self._z_test(m_a, m_b)
                result["z_score"] = z
                result["p_value"] = p
                result["significant"] = p < 0.05

        return result

    @staticmethod
    def _z_test(a: VariantMetrics, b: VariantMetrics) -> tuple[float, float]:
        """Two-proportion z-test for success rates."""
        p_a = a.success_rate
        p_b = b.success_rate
        n_a = a.observations
        n_b = b.observations

        p_pool = (a.successes + b.successes) / (n_a + n_b)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b)) if p_pool > 0 else 1e-10

        z = (p_a - p_b) / se if se > 0 else 0.0
        # Approximate two-tailed p-value using normal CDF
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        return z, p_value

    def _get_experiment(self, experiment_id: str) -> ExperimentConfig:
        """Retrieve an experiment by ID or raise KeyError."""
        if experiment_id not in self._experiments:
            raise KeyError(f"Experiment {experiment_id} not found")
        return self._experiments[experiment_id]


if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser(description="A/B test manager")
    parser.add_argument("--model", default="sentiment")
    parser.add_argument("--version-a", type=int, default=1)
    parser.add_argument("--version-b", type=int, default=2)
    parser.add_argument("--split", default="80/20")
    args = parser.parse_args()

    split_a, split_b = map(float, args.split.split("/"))

    manager = ABTestManager()
    config = ExperimentConfig(
        experiment_id="exp-001",
        model_name=args.model,
        variants=[
            Variant("control", args.version_a, split_a),
            Variant("treatment", args.version_b, split_b),
        ],
        min_sample_size=50,
    )
    manager.create_experiment(config)
    manager.start_experiment("exp-001")

    # Simulate 500 requests
    rng = random.Random(42)
    for i in range(500):
        user_id = f"user-{i}"
        variant = manager.assign_variant("exp-001", user_id)
        # Treatment has slightly higher success rate
        success_prob = 0.85 if variant.name == "control" else 0.90
        success = rng.random() < success_prob
        latency = rng.gauss(50, 10)
        manager.record_observation("exp-001", variant.name, success, latency)

    results = manager.get_results("exp-001")
    print("A/B Test Results:")
    for name, data in results["variants"].items():
        print(f"  {name} (v{data['model_version']}): "
              f"n={data['observations']}, "
              f"success={data['success_rate']:.1%}, "
              f"latency={data['mean_latency_ms']:.1f}ms")

    if "z_score" in results:
        print(f"\nStatistical test:")
        print(f"  z-score: {results['z_score']:.3f}")
        print(f"  p-value: {results['p_value']:.4f}")
        print(f"  significant: {results['significant']}")
