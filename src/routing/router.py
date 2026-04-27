"""Traffic routing: weighted split and hash-based user assignment.

Routes incoming requests to model versions based on configurable
traffic splits. Supports weighted random routing and deterministic
hash-based assignment (so the same user always sees the same version).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RoutingStrategy(Enum):
    """Available traffic routing strategies."""
    WEIGHTED = "weighted"
    HASH_BASED = "hash_based"
    ROUND_ROBIN = "round_robin"


@dataclass
class RouteTarget:
    """A model version that can receive traffic.

    Attributes:
        model_name: Name of the model.
        version: Version number.
        weight: Traffic weight (0.0 to 1.0).
        is_shadow: If True, receives a copy of traffic but responses
                   are not returned to the client.
    """
    model_name: str
    version: int
    weight: float = 1.0
    is_shadow: bool = False


@dataclass
class RoutingDecision:
    """The result of a routing decision.

    Attributes:
        target: The selected model version.
        strategy: Which strategy was used.
        user_id: The user ID that was routed (if applicable).
        shadow_targets: Additional targets receiving shadow traffic.
        decided_at: Timestamp of the decision.
    """
    target: RouteTarget
    strategy: RoutingStrategy
    user_id: str = ""
    shadow_targets: List[RouteTarget] = field(default_factory=list)
    decided_at: float = field(default_factory=time.time)


class TrafficRouter:
    """Routes requests to model versions based on traffic rules.

    Supports three strategies:
    - WEIGHTED: Random selection proportional to weights.
    - HASH_BASED: Deterministic assignment using user ID hash.
    - ROUND_ROBIN: Cycle through targets in order.
    """

    def __init__(
        self,
        targets: Optional[List[RouteTarget]] = None,
        strategy: RoutingStrategy = RoutingStrategy.HASH_BASED,
    ) -> None:
        """Initialize the router.

        Args:
            targets: List of route targets with weights.
            strategy: Routing strategy to use.
        """
        self._targets: List[RouteTarget] = targets or []
        self._strategy = strategy
        self._rr_index = 0
        self._route_counts: Dict[str, int] = {}

    @property
    def strategy(self) -> RoutingStrategy:
        """Current routing strategy."""
        return self._strategy

    def set_targets(self, targets: List[RouteTarget]) -> None:
        """Update the routing targets.

        Normalizes weights so they sum to 1.0 (excluding shadow targets).

        Args:
            targets: New list of route targets.
        """
        self._targets = targets
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize non-shadow target weights to sum to 1.0."""
        active = [t for t in self._targets if not t.is_shadow]
        total = sum(t.weight for t in active)
        if total > 0:
            for t in active:
                t.weight = t.weight / total

    @staticmethod
    def _hash_user(user_id: str) -> int:
        """Hash a user ID to a bucket in [0, 100).

        Args:
            user_id: The user identifier.

        Returns:
            Integer in [0, 100).
        """
        digest = hashlib.md5(user_id.encode()).hexdigest()
        return int(digest, 16) % 100

    def route(
        self,
        user_id: str = "",
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Route a request to a model version.

        Args:
            user_id: User identifier (used for hash-based routing).
            request_metadata: Optional metadata for routing decisions.

        Returns:
            RoutingDecision with the selected target.

        Raises:
            ValueError: If no active targets are configured.
        """
        active = [t for t in self._targets if not t.is_shadow]
        shadows = [t for t in self._targets if t.is_shadow]

        if not active:
            raise ValueError("No active routing targets configured")

        if self._strategy == RoutingStrategy.HASH_BASED and user_id:
            target = self._route_hash(user_id, active)
        elif self._strategy == RoutingStrategy.ROUND_ROBIN:
            target = self._route_round_robin(active)
        else:
            target = self._route_weighted(active)

        key = f"{target.model_name}:{target.version}"
        self._route_counts[key] = self._route_counts.get(key, 0) + 1

        return RoutingDecision(
            target=target,
            strategy=self._strategy,
            user_id=user_id,
            shadow_targets=shadows,
        )

    def _route_hash(
        self, user_id: str, targets: List[RouteTarget]
    ) -> RouteTarget:
        """Deterministic routing based on user ID hash.

        Args:
            user_id: User identifier.
            targets: Active targets sorted by weight.

        Returns:
            The selected RouteTarget.
        """
        bucket = self._hash_user(user_id)
        cumulative = 0.0
        for target in targets:
            cumulative += target.weight * 100
            if bucket < cumulative:
                return target
        return targets[-1]

    def _route_weighted(self, targets: List[RouteTarget]) -> RouteTarget:
        """Random weighted routing.

        Args:
            targets: Active targets with normalized weights.

        Returns:
            The selected RouteTarget.
        """
        import random
        r = random.random()
        cumulative = 0.0
        for target in targets:
            cumulative += target.weight
            if r < cumulative:
                return target
        return targets[-1]

    def _route_round_robin(self, targets: List[RouteTarget]) -> RouteTarget:
        """Round-robin routing across targets.

        Args:
            targets: Active targets.

        Returns:
            The next RouteTarget in the cycle.
        """
        target = targets[self._rr_index % len(targets)]
        self._rr_index += 1
        return target

    def get_traffic_distribution(self) -> Dict[str, float]:
        """Get the observed traffic distribution across targets.

        Returns:
            Dict mapping model:version to fraction of total traffic.
        """
        total = sum(self._route_counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self._route_counts.items()}


if __name__ == "__main__":
    targets = [
        RouteTarget("sentiment", 1, weight=0.9),
        RouteTarget("sentiment", 2, weight=0.1),
        RouteTarget("sentiment", 3, is_shadow=True),
    ]

    router = TrafficRouter(targets, strategy=RoutingStrategy.HASH_BASED)

    # Route 1000 requests with different user IDs
    for i in range(1000):
        decision = router.route(user_id=f"user-{i}")

    dist = router.get_traffic_distribution()
    print("Traffic distribution (hash-based, 90/10 split):")
    for target, frac in sorted(dist.items()):
        print(f"  {target}: {frac:.1%}")

    # Verify determinism: same user always gets same version
    decisions = [router.route(user_id="user-42") for _ in range(10)]
    versions = {d.target.version for d in decisions}
    print(f"\nDeterminism check: user-42 always routed to v{decisions[0].target.version} "
          f"(unique versions seen: {len(versions)})")

    # Show shadow targets
    d = router.route(user_id="user-1")
    print(f"\nShadow targets: {[f'v{s.version}' for s in d.shadow_targets]}")
