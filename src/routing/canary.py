"""Canary deployment: gradual rollout with automatic rollback on error.

Manages the progressive shift of traffic from a stable model version to
a new canary version. Monitors error rate and latency in real time using
a sliding window, and automatically rolls back if thresholds are breached.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple


class CanaryState(Enum):
    """State of a canary deployment."""
    PENDING = "pending"
    RAMPING = "ramping"
    FULL_ROLLOUT = "full_rollout"
    ROLLED_BACK = "rolled_back"
    PAUSED = "paused"


@dataclass
class CanaryConfig:
    """Configuration for a canary deployment.

    Attributes:
        model_name: Name of the model.
        stable_version: Current production version.
        canary_version: New version being tested.
        initial_pct: Starting traffic percentage for canary.
        ramp_step_pct: How much to increase per ramp step.
        ramp_interval_s: Seconds between ramp steps.
        error_threshold: Max error rate before rollback (0-1).
        latency_threshold_ms: Max p99 latency before rollback.
        window_size: Number of observations in the sliding window.
    """
    model_name: str
    stable_version: int
    canary_version: int
    initial_pct: float = 5.0
    ramp_step_pct: float = 10.0
    ramp_interval_s: float = 60.0
    error_threshold: float = 0.05
    latency_threshold_ms: float = 200.0
    window_size: int = 100


@dataclass
class CanaryObservation:
    """A single observation from the canary version.

    Attributes:
        is_error: Whether the request resulted in an error.
        latency_ms: Request latency in milliseconds.
        timestamp: When the observation was recorded.
    """
    is_error: bool
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class RollbackEvent:
    """Record of an automatic rollback.

    Attributes:
        reason: Why the rollback was triggered.
        error_rate: Error rate at time of rollback.
        p99_latency_ms: p99 latency at time of rollback.
        canary_pct: Traffic percentage at time of rollback.
        observations: Number of observations in the window.
        timestamp: When the rollback occurred.
    """
    reason: str
    error_rate: float
    p99_latency_ms: float
    canary_pct: float
    observations: int
    timestamp: float = field(default_factory=time.time)


class CanaryController:
    """Controls canary deployment with sliding-window health checks.

    Gradually increases traffic to the canary version while monitoring
    error rate and latency. Rolls back automatically if the canary
    shows sustained degradation (not just a single spike).
    """

    def __init__(self, config: CanaryConfig) -> None:
        """Initialize the canary controller.

        Args:
            config: Canary deployment configuration.
        """
        self._config = config
        self._state = CanaryState.PENDING
        self._canary_pct = 0.0
        self._window: Deque[CanaryObservation] = deque(
            maxlen=config.window_size
        )
        self._last_ramp_time = 0.0
        self._rollback_event: Optional[RollbackEvent] = None
        self._ramp_history: List[Tuple[float, float]] = []

    @property
    def state(self) -> CanaryState:
        """Current canary deployment state."""
        return self._state

    @property
    def canary_pct(self) -> float:
        """Current traffic percentage going to canary."""
        return self._canary_pct

    def start(self) -> None:
        """Start the canary deployment at the initial percentage."""
        self._state = CanaryState.RAMPING
        self._canary_pct = self._config.initial_pct
        self._last_ramp_time = time.time()
        self._ramp_history.append((time.time(), self._canary_pct))

    def record_observation(self, obs: CanaryObservation) -> None:
        """Record a canary observation."""
        self._window.append(obs)

    def check_health(self) -> Tuple[bool, str]:
        """Check canary health based on the sliding window."""
        if len(self._window) < 10:
            return True, "insufficient data"

        observations = list(self._window)
        error_rate = sum(1 for o in observations if o.is_error) / len(observations)
        latencies = sorted(o.latency_ms for o in observations)
        p99_idx = int(len(latencies) * 0.99)
        p99_latency = latencies[min(p99_idx, len(latencies) - 1)]

        if error_rate > self._config.error_threshold:
            return False, (
                f"error rate {error_rate:.1%} exceeds threshold "
                f"{self._config.error_threshold:.1%}"
            )

        if p99_latency > self._config.latency_threshold_ms:
            return False, (
                f"p99 latency {p99_latency:.0f}ms exceeds threshold "
                f"{self._config.latency_threshold_ms:.0f}ms"
            )

        return True, "healthy"

    def step(self) -> str:
        """Advance the canary deployment by one step.

        Checks health, ramps up if healthy, or rolls back if unhealthy.
        """
        if self._state not in (CanaryState.RAMPING, CanaryState.PENDING):
            return f"canary is {self._state.value}, no action taken"

        if self._state == CanaryState.PENDING:
            self.start()
            return f"started canary at {self._canary_pct:.0f}%"

        healthy, reason = self.check_health()

        if not healthy:
            self._trigger_rollback(reason)
            return f"rolled back: {reason}"

        elapsed = time.time() - self._last_ramp_time
        if elapsed >= self._config.ramp_interval_s:
            return self._ramp_up()

        return f"waiting ({elapsed:.0f}s / {self._config.ramp_interval_s:.0f}s)"

    def _ramp_up(self) -> str:
        """Increase canary traffic percentage."""
        new_pct = min(
            self._canary_pct + self._config.ramp_step_pct, 100.0
        )
        old_pct = self._canary_pct
        self._canary_pct = new_pct
        self._last_ramp_time = time.time()
        self._ramp_history.append((time.time(), new_pct))

        if new_pct >= 100.0:
            self._state = CanaryState.FULL_ROLLOUT
            return f"full rollout: canary at 100%"

        return f"ramped {old_pct:.0f}% → {new_pct:.0f}%"

    def _trigger_rollback(self, reason: str) -> None:
        """Execute automatic rollback."""
        observations = list(self._window)
        error_rate = sum(1 for o in observations if o.is_error) / max(len(observations), 1)
        latencies = sorted(o.latency_ms for o in observations)
        p99_idx = int(len(latencies) * 0.99)
        p99 = latencies[min(p99_idx, len(latencies) - 1)] if latencies else 0.0

        self._rollback_event = RollbackEvent(
            reason=reason,
            error_rate=error_rate,
            p99_latency_ms=p99,
            canary_pct=self._canary_pct,
            observations=len(observations),
        )
        self._state = CanaryState.ROLLED_BACK
        self._canary_pct = 0.0

    def get_status(self) -> Dict:
        """Get current canary deployment status."""
        observations = list(self._window)
        error_rate = (
            sum(1 for o in observations if o.is_error) / len(observations)
            if observations else 0.0
        )
        latencies = [o.latency_ms for o in observations]
        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

        status = {
            "state": self._state.value,
            "canary_pct": self._canary_pct,
            "stable_pct": 100.0 - self._canary_pct,
            "observations": len(observations),
            "error_rate": error_rate,
            "mean_latency_ms": mean_latency,
            "ramp_history": self._ramp_history,
        }
        if self._rollback_event:
            status["rollback"] = {
                "reason": self._rollback_event.reason,
                "error_rate": self._rollback_event.error_rate,
                "p99_latency_ms": self._rollback_event.p99_latency_ms,
            }
        return status


if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser(description="Canary deployment controller")
    parser.add_argument("--model", default="sentiment")
    parser.add_argument("--stable", type=int, default=1)
    parser.add_argument("--canary", type=int, default=2)
    parser.add_argument("--initial-pct", type=float, default=5.0)
    parser.add_argument("--error-threshold", type=float, default=0.05)
    args = parser.parse_args()

    config = CanaryConfig(
        model_name=args.model, stable_version=args.stable,
        canary_version=args.canary, initial_pct=args.initial_pct,
        ramp_step_pct=15.0, ramp_interval_s=0.0,
        error_threshold=args.error_threshold, window_size=50,
    )

    controller = CanaryController(config)
    rng = random.Random(42)

    print("=== Canary Deployment Simulation ===\n")

    # Simulate healthy canary ramping up
    for step_num in range(10):
        result = controller.step()
        print(f"Step {step_num + 1}: {result}")
        for _ in range(20):
            controller.record_observation(CanaryObservation(
                is_error=rng.random() < 0.02, latency_ms=rng.gauss(50, 10),
            ))

        if controller.state in (CanaryState.FULL_ROLLOUT, CanaryState.ROLLED_BACK):
            break

    print(f"\nFinal state: {controller.state.value}")
    print(f"Canary traffic: {controller.canary_pct:.0f}%")

    # Now simulate a bad canary that triggers rollback
    print("\n=== Bad Canary (auto-rollback) ===\n")
    bad_controller = CanaryController(CanaryConfig(
        model_name="sentiment", stable_version=1, canary_version=3,
        initial_pct=5.0, ramp_interval_s=0.0,
        error_threshold=0.05, window_size=50,
    ))

    for step_num in range(10):
        result = bad_controller.step()
        print(f"Step {step_num + 1}: {result}")
        for _ in range(20):
            bad_controller.record_observation(CanaryObservation(
                is_error=rng.random() < 0.10, latency_ms=rng.gauss(80, 30),
            ))
        if bad_controller.state == CanaryState.ROLLED_BACK:
            status = bad_controller.get_status()
            print(f"\nRollback: {status['rollback']['reason']}")
            break
