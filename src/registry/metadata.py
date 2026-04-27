"""Model metadata: hyperparameters, evaluation metrics, and lineage tracking.

Each model version carries metadata describing how it was trained, what data
it was trained on, and how it performed during evaluation. This enables
reproducibility and informed deployment decisions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainingInfo:
    """Training provenance for a model version.

    Attributes:
        dataset_name: Name or identifier of the training dataset.
        dataset_hash: Hash of the training data for reproducibility.
        framework: ML framework used (e.g. 'pytorch', 'sklearn').
        training_duration_s: Wall-clock training time in seconds.
        hyperparameters: Dict of hyperparameter name → value.
    """
    dataset_name: str = ""
    dataset_hash: str = ""
    framework: str = "pytorch"
    training_duration_s: float = 0.0
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalMetrics:
    """Evaluation metrics from offline validation.

    Attributes:
        accuracy: Classification accuracy (0-1).
        precision: Weighted precision.
        recall: Weighted recall.
        f1_score: Weighted F1.
        custom: Additional domain-specific metrics.
    """
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    custom: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelMetadata:
    """Complete metadata record for a model version.

    Attributes:
        model_name: Name of the model.
        version: Version number.
        description: Human-readable description of this version.
        training: Training provenance information.
        eval_metrics: Offline evaluation metrics.
        tags: Arbitrary key-value tags for filtering.
        created_at: Unix timestamp of metadata creation.
        parent_version: Version this was derived from (for lineage).
    """
    model_name: str
    version: int
    description: str = ""
    training: TrainingInfo = field(default_factory=TrainingInfo)
    eval_metrics: EvalMetrics = field(default_factory=EvalMetrics)
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    parent_version: Optional[int] = None


class MetadataStore:
    """In-memory metadata store for model versions.

    Provides CRUD operations on model metadata with support for
    lineage queries (which version was derived from which).
    """

    def __init__(self) -> None:
        self._records: Dict[str, ModelMetadata] = {}

    @staticmethod
    def _key(name: str, version: int) -> str:
        """Build a lookup key from model name and version."""
        return f"{name}:{version}"

    def put(self, meta: ModelMetadata) -> None:
        """Store or update metadata for a model version.

        Args:
            meta: The metadata record to store.
        """
        key = self._key(meta.model_name, meta.version)
        self._records[key] = meta

    def get(self, name: str, version: int) -> ModelMetadata:
        """Retrieve metadata for a specific model version.

        Args:
            name: Model name.
            version: Version number.

        Returns:
            The ModelMetadata record.

        Raises:
            KeyError: If no metadata exists for this version.
        """
        key = self._key(name, version)
        if key not in self._records:
            raise KeyError(f"No metadata for {key}")
        return self._records[key]

    def list_versions(self, name: str) -> List[ModelMetadata]:
        """List all metadata records for a model, sorted by version.

        Args:
            name: Model name.

        Returns:
            Sorted list of ModelMetadata.
        """
        records = [r for r in self._records.values() if r.model_name == name]
        return sorted(records, key=lambda r: r.version)

    def get_lineage(self, name: str, version: int) -> List[ModelMetadata]:
        """Trace the lineage of a model version back to its root.

        Follows parent_version links to build the full derivation chain.

        Args:
            name: Model name.
            version: Starting version.

        Returns:
            List from the given version back to the root (no parent).
        """
        chain: List[ModelMetadata] = []
        current: Optional[int] = version
        visited: set[int] = set()

        while current is not None and current not in visited:
            visited.add(current)
            key = self._key(name, current)
            if key not in self._records:
                break
            meta = self._records[key]
            chain.append(meta)
            current = meta.parent_version

        return chain

    def find_by_tag(self, tag_key: str, tag_value: str) -> List[ModelMetadata]:
        """Find all model versions matching a tag.

        Args:
            tag_key: Tag key to match.
            tag_value: Tag value to match.

        Returns:
            List of matching ModelMetadata records.
        """
        return [
            r for r in self._records.values()
            if r.tags.get(tag_key) == tag_value
        ]

    def compare_versions(
        self, name: str, v1: int, v2: int
    ) -> Dict[str, Any]:
        """Compare evaluation metrics between two versions.

        Args:
            name: Model name.
            v1: First version number.
            v2: Second version number.

        Returns:
            Dict with metric deltas (v2 - v1).
        """
        m1 = self.get(name, v1)
        m2 = self.get(name, v2)
        return {
            "accuracy_delta": m2.eval_metrics.accuracy - m1.eval_metrics.accuracy,
            "precision_delta": m2.eval_metrics.precision - m1.eval_metrics.precision,
            "recall_delta": m2.eval_metrics.recall - m1.eval_metrics.recall,
            "f1_delta": m2.eval_metrics.f1_score - m1.eval_metrics.f1_score,
        }


if __name__ == "__main__":
    store = MetadataStore()

    # Register a lineage: v1 → v2 → v3
    for v in range(1, 4):
        meta = ModelMetadata(
            model_name="sentiment",
            version=v,
            description=f"Sentiment classifier v{v}",
            training=TrainingInfo(
                dataset_name="imdb-reviews",
                framework="pytorch",
                hyperparameters={"lr": 0.001 / v, "epochs": 10 * v},
            ),
            eval_metrics=EvalMetrics(
                accuracy=0.85 + 0.03 * v,
                f1_score=0.83 + 0.04 * v,
            ),
            tags={"team": "nlp", "experiment": f"exp-{v}"},
            parent_version=v - 1 if v > 1 else None,
        )
        store.put(meta)

    # Show lineage
    print("Lineage of sentiment v3:")
    for m in store.get_lineage("sentiment", 3):
        print(f"  v{m.version}: acc={m.eval_metrics.accuracy:.2f}, "
              f"parent={'v' + str(m.parent_version) if m.parent_version else 'root'}")

    # Compare versions
    delta = store.compare_versions("sentiment", 1, 3)
    print(f"\nv1 → v3 improvement:")
    for k, d in delta.items():
        print(f"  {k}: {d:+.4f}")

    # Find by tag
    nlp_models = store.find_by_tag("team", "nlp")
    print(f"\nModels tagged team=nlp: {len(nlp_models)}")
