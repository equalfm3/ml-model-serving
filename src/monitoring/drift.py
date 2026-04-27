"""Data drift detection: KS test and feature distribution comparison.

Detects when the distribution of incoming features diverges from the
training distribution. Uses the Kolmogorov-Smirnov test on a sliding
window of production data, with reservoir sampling to maintain a
fixed-size sample efficiently.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class DriftResult:
    """Result of a drift detection test for a single feature.

    Attributes:
        feature_name: Name of the feature tested.
        ks_statistic: KS test statistic (max CDF distance).
        p_value: p-value from the KS test.
        is_drifted: Whether drift was detected at the given threshold.
        reference_mean: Mean of the reference distribution.
        production_mean: Mean of the production sample.
        reference_std: Std dev of the reference distribution.
        production_std: Std dev of the production sample.
    """
    feature_name: str
    ks_statistic: float
    p_value: float
    is_drifted: bool
    reference_mean: float = 0.0
    production_mean: float = 0.0
    reference_std: float = 0.0
    production_std: float = 0.0


@dataclass
class DriftReport:
    """Aggregate drift report across all features.

    Attributes:
        model_name: Name of the model.
        version: Model version.
        timestamp: When the report was generated.
        feature_results: Per-feature drift results.
        overall_drifted: True if any feature has drifted.
        drift_score: Fraction of features that drifted.
    """
    model_name: str
    version: int
    timestamp: float = field(default_factory=time.time)
    feature_results: List[DriftResult] = field(default_factory=list)
    overall_drifted: bool = False
    drift_score: float = 0.0


class ReservoirSampler:
    """Reservoir sampling for maintaining a fixed-size sample of streaming data.

    Implements Algorithm R: each new item has a decreasing probability
    of being included, ensuring a uniform random sample of all items
    seen so far.
    """

    def __init__(self, capacity: int = 1000) -> None:
        """Initialize the reservoir.

        Args:
            capacity: Maximum number of samples to retain.
        """
        self._capacity = capacity
        self._reservoir: List[np.ndarray] = []
        self._count = 0
        self._rng = random.Random(42)

    @property
    def size(self) -> int:
        """Number of samples currently in the reservoir."""
        return len(self._reservoir)

    def add(self, sample: np.ndarray) -> None:
        """Add a sample to the reservoir.

        Uses reservoir sampling to maintain a uniform random sample.

        Args:
            sample: Feature vector to add.
        """
        self._count += 1
        if len(self._reservoir) < self._capacity:
            self._reservoir.append(sample.copy())
        else:
            j = self._rng.randint(0, self._count - 1)
            if j < self._capacity:
                self._reservoir[j] = sample.copy()

    def get_samples(self) -> np.ndarray:
        """Get all samples as a numpy array.

        Returns:
            Array of shape (n_samples, n_features).
        """
        if not self._reservoir:
            return np.array([])
        return np.stack(self._reservoir)

    def reset(self) -> None:
        """Clear the reservoir."""
        self._reservoir.clear()
        self._count = 0


class DriftDetector:
    """Detects data drift using the Kolmogorov-Smirnov test.

    Maintains a reference distribution (from training data) and compares
    it against a reservoir sample of production data. The KS test measures
    the maximum distance between the two empirical CDFs.
    """

    def __init__(
        self,
        feature_names: List[str],
        significance_level: float = 0.05,
        reservoir_capacity: int = 1000,
    ) -> None:
        """Initialize the drift detector.

        Args:
            feature_names: Names of the features to monitor.
            significance_level: p-value threshold for drift detection.
            reservoir_capacity: Size of the production data reservoir.
        """
        self._feature_names = feature_names
        self._significance = significance_level
        self._reference: Optional[np.ndarray] = None
        self._sampler = ReservoirSampler(reservoir_capacity)

    def set_reference(self, data: np.ndarray) -> None:
        """Set the reference distribution from training data.

        Args:
            data: Training data of shape (n_samples, n_features).

        Raises:
            ValueError: If feature count doesn't match.
        """
        if data.shape[1] != len(self._feature_names):
            raise ValueError(
                f"Expected {len(self._feature_names)} features, "
                f"got {data.shape[1]}"
            )
        self._reference = data.copy()

    def observe(self, features: np.ndarray) -> None:
        """Add a production observation to the reservoir.

        Args:
            features: Feature vector of shape (n_features,).
        """
        self._sampler.add(features)

    def detect(
        self,
        model_name: str = "",
        version: int = 0,
    ) -> DriftReport:
        """Run drift detection on all features.

        Compares the reference distribution against the production
        reservoir using the two-sample KS test.

        Args:
            model_name: Model name for the report.
            version: Model version for the report.

        Returns:
            DriftReport with per-feature results.

        Raises:
            ValueError: If reference data is not set.
            ValueError: If insufficient production data.
        """
        if self._reference is None:
            raise ValueError("Reference distribution not set")

        production = self._sampler.get_samples()
        if production.size == 0:
            raise ValueError("No production data observed")

        results: List[DriftResult] = []
        for i, name in enumerate(self._feature_names):
            ref_col = self._reference[:, i]
            prod_col = production[:, i]

            ks_stat, p_value = stats.ks_2samp(ref_col, prod_col)

            results.append(DriftResult(
                feature_name=name,
                ks_statistic=ks_stat,
                p_value=p_value,
                is_drifted=p_value < self._significance,
                reference_mean=float(np.mean(ref_col)),
                production_mean=float(np.mean(prod_col)),
                reference_std=float(np.std(ref_col)),
                production_std=float(np.std(prod_col)),
            ))

        n_drifted = sum(1 for r in results if r.is_drifted)
        return DriftReport(
            model_name=model_name,
            version=version,
            feature_results=results,
            overall_drifted=n_drifted > 0,
            drift_score=n_drifted / len(results) if results else 0.0,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drift detection")
    parser.add_argument("--model", default="sentiment")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--reference-data", default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    feature_names = ["word_count", "sentiment_score", "text_length",
                     "vocab_richness"]

    detector = DriftDetector(feature_names, significance_level=0.05)

    # Simulate reference data (training distribution)
    reference = rng.normal(loc=[50, 0.5, 200, 0.3], scale=[10, 0.2, 50, 0.1],
                           size=(500, 4))
    detector.set_reference(reference)

    # Simulate production data with drift on features 1 and 3
    for _ in range(300):
        sample = rng.normal(
            loc=[50, 0.7, 200, 0.5],  # Shifted sentiment_score and vocab_richness
            scale=[10, 0.2, 50, 0.1],
        )
        detector.observe(sample)

    report = detector.detect(args.model, args.version)

    print(f"Drift Report for {report.model_name} v{report.version}")
    print(f"Overall drifted: {report.overall_drifted}")
    print(f"Drift score: {report.drift_score:.0%} of features drifted\n")

    for r in report.feature_results:
        status = "DRIFTED" if r.is_drifted else "OK"
        print(f"  {r.feature_name}: KS={r.ks_statistic:.3f}, "
              f"p={r.p_value:.4f} [{status}]")
        print(f"    ref: mean={r.reference_mean:.2f}, std={r.reference_std:.2f}")
        print(f"    prod: mean={r.production_mean:.2f}, std={r.production_std:.2f}")
