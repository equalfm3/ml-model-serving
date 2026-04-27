"""Model lifecycle management: stage transitions with audit trail.

Enforces the deployment pipeline: registered → staging → production → archived.
Every transition is logged with a timestamp, initiator, and reason, creating
a full audit trail for model governance and compliance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Stage(Enum):
    """Lifecycle stages for a model version."""
    REGISTERED = "registered"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    ROLLBACK = "rollback"


# Valid transitions: from_stage → set of allowed to_stages
VALID_TRANSITIONS: Dict[Stage, set[Stage]] = {
    Stage.REGISTERED: {Stage.STAGING, Stage.ARCHIVED},
    Stage.STAGING: {Stage.PRODUCTION, Stage.ARCHIVED},
    Stage.PRODUCTION: {Stage.ARCHIVED, Stage.ROLLBACK},
    Stage.ROLLBACK: {Stage.PRODUCTION},
    Stage.ARCHIVED: set(),
}


@dataclass
class TransitionRecord:
    """Audit record for a lifecycle transition.

    Attributes:
        model_name: Name of the model.
        version: Version number.
        from_stage: Previous stage.
        to_stage: New stage.
        initiated_by: User or system that triggered the transition.
        reason: Human-readable reason for the transition.
        timestamp: Unix timestamp of the transition.
    """
    model_name: str
    version: int
    from_stage: Stage
    to_stage: Stage
    initiated_by: str
    reason: str
    timestamp: float = field(default_factory=time.time)


class LifecycleManager:
    """Manages model version lifecycle with enforced transition rules.

    Tracks the current stage of every model version and maintains a
    complete audit log of all transitions.
    """

    def __init__(self) -> None:
        self._stages: Dict[str, Stage] = {}
        self._audit_log: List[TransitionRecord] = []
        self._production_history: Dict[str, List[int]] = {}

    @staticmethod
    def _key(name: str, version: int) -> str:
        """Build a lookup key."""
        return f"{name}:{version}"

    def register(self, name: str, version: int) -> Stage:
        """Register a new model version in the lifecycle.

        Args:
            name: Model name.
            version: Version number.

        Returns:
            The initial stage (REGISTERED).

        Raises:
            ValueError: If the version is already registered.
        """
        key = self._key(name, version)
        if key in self._stages:
            raise ValueError(f"{key} is already registered")
        self._stages[key] = Stage.REGISTERED
        return Stage.REGISTERED

    def get_stage(self, name: str, version: int) -> Stage:
        """Get the current lifecycle stage of a model version.

        Args:
            name: Model name.
            version: Version number.

        Returns:
            Current Stage.

        Raises:
            KeyError: If the version is not registered.
        """
        key = self._key(name, version)
        if key not in self._stages:
            raise KeyError(f"{key} not found in lifecycle")
        return self._stages[key]

    def transition(
        self,
        name: str,
        version: int,
        to_stage: Stage,
        initiated_by: str = "system",
        reason: str = "",
    ) -> TransitionRecord:
        """Transition a model version to a new stage.

        Validates that the transition is allowed before applying it.

        Args:
            name: Model name.
            version: Version number.
            to_stage: Target stage.
            initiated_by: Who initiated the transition.
            reason: Why the transition is happening.

        Returns:
            The TransitionRecord for the audit log.

        Raises:
            KeyError: If the version is not registered.
            ValueError: If the transition is not allowed.
        """
        key = self._key(name, version)
        current = self.get_stage(name, version)

        allowed = VALID_TRANSITIONS.get(current, set())
        if to_stage not in allowed:
            raise ValueError(
                f"Cannot transition {key} from {current.value} to "
                f"{to_stage.value}. Allowed: {[s.value for s in allowed]}"
            )

        record = TransitionRecord(
            model_name=name,
            version=version,
            from_stage=current,
            to_stage=to_stage,
            initiated_by=initiated_by,
            reason=reason,
        )
        self._stages[key] = to_stage
        self._audit_log.append(record)

        if to_stage == Stage.PRODUCTION:
            self._production_history.setdefault(name, []).append(version)

        return record

    def promote_to_staging(
        self, name: str, version: int, initiated_by: str = "system"
    ) -> TransitionRecord:
        """Convenience: promote from registered to staging.

        Args:
            name: Model name.
            version: Version number.
            initiated_by: Who initiated the promotion.

        Returns:
            The TransitionRecord.
        """
        return self.transition(
            name, version, Stage.STAGING, initiated_by,
            reason="Promoted to staging for validation"
        )

    def promote_to_production(
        self, name: str, version: int, initiated_by: str = "system"
    ) -> TransitionRecord:
        """Convenience: promote from staging to production.

        Args:
            name: Model name.
            version: Version number.
            initiated_by: Who initiated the promotion.

        Returns:
            The TransitionRecord.
        """
        return self.transition(
            name, version, Stage.PRODUCTION, initiated_by,
            reason="Promoted to production after staging validation"
        )

    def rollback(
        self, name: str, version: int, reason: str = "Performance degradation"
    ) -> Optional[TransitionRecord]:
        """Roll back a production model and restore the previous version.

        Args:
            name: Model name.
            version: Current production version to roll back.
            reason: Why the rollback is happening.

        Returns:
            TransitionRecord for the rollback, or None if no previous version.
        """
        record = self.transition(
            name, version, Stage.ROLLBACK, "auto-rollback", reason
        )

        history = self._production_history.get(name, [])
        previous = [v for v in history if v != version]
        if previous:
            prev_version = previous[-1]
            prev_key = self._key(name, prev_version)
            if prev_key in self._stages:
                self._stages[prev_key] = Stage.PRODUCTION

        return record

    def get_audit_log(
        self, name: Optional[str] = None
    ) -> List[TransitionRecord]:
        """Retrieve the audit log, optionally filtered by model name.

        Args:
            name: If provided, filter to this model only.

        Returns:
            List of TransitionRecords in chronological order.
        """
        if name is None:
            return list(self._audit_log)
        return [r for r in self._audit_log if r.model_name == name]

    def get_production_version(self, name: str) -> Optional[int]:
        """Get the current production version of a model.

        Args:
            name: Model name.

        Returns:
            Version number, or None if no version is in production.
        """
        for key, stage in self._stages.items():
            if key.startswith(f"{name}:") and stage == Stage.PRODUCTION:
                return int(key.split(":")[1])
        return None


if __name__ == "__main__":
    lm = LifecycleManager()

    # Register and promote v1 through the pipeline
    lm.register("sentiment", 1)
    lm.promote_to_staging("sentiment", 1, initiated_by="alice")
    lm.promote_to_production("sentiment", 1, initiated_by="alice")
    print(f"v1 stage: {lm.get_stage('sentiment', 1).value}")

    # Register v2 and promote to staging
    lm.register("sentiment", 2)
    lm.promote_to_staging("sentiment", 2, initiated_by="bob")
    lm.promote_to_production("sentiment", 2, initiated_by="bob")

    # Simulate rollback of v2
    lm.rollback("sentiment", 2, reason="Accuracy dropped 5%")
    print(f"v2 stage after rollback: {lm.get_stage('sentiment', 2).value}")
    print(f"Production version: {lm.get_production_version('sentiment')}")

    # Show audit trail
    print("\nAudit log:")
    for rec in lm.get_audit_log("sentiment"):
        print(f"  v{rec.version}: {rec.from_stage.value} → {rec.to_stage.value} "
              f"by {rec.initiated_by} ({rec.reason})")

    # Demonstrate invalid transition
    lm.register("sentiment", 3)
    try:
        lm.promote_to_production("sentiment", 3)
    except ValueError as e:
        print(f"\nBlocked invalid transition: {e}")
