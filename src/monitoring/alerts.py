"""Alert rules: threshold-based triggers and notification dispatch.

Defines alert rules that fire when metrics exceed configurable thresholds.
Supports cooldown periods to prevent alert storms, severity levels, and
a pluggable notification dispatch system.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class Severity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Comparator(Enum):
    """Comparison operators for threshold checks."""
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    EQUAL = "eq"


@dataclass
class AlertRule:
    """A threshold-based alert rule.

    Attributes:
        rule_id: Unique identifier for this rule.
        name: Human-readable rule name.
        metric_name: Which metric to monitor.
        comparator: How to compare the metric value to the threshold.
        threshold: The threshold value.
        severity: Alert severity when triggered.
        cooldown_s: Minimum seconds between consecutive firings.
        description: Detailed description of what this alert means.
        model_name: Optional filter to a specific model.
        version: Optional filter to a specific version.
    """
    rule_id: str
    name: str
    metric_name: str
    comparator: Comparator
    threshold: float
    severity: Severity = Severity.WARNING
    cooldown_s: float = 300.0
    description: str = ""
    model_name: Optional[str] = None
    version: Optional[int] = None


@dataclass
class Alert:
    """A fired alert instance.

    Attributes:
        rule: The rule that triggered this alert.
        metric_value: The metric value that triggered the alert.
        message: Human-readable alert message.
        fired_at: When the alert was fired.
        acknowledged: Whether the alert has been acknowledged.
        acknowledged_by: Who acknowledged the alert.
    """
    rule: AlertRule
    metric_value: float
    message: str
    fired_at: float = field(default_factory=time.time)
    acknowledged: bool = False
    acknowledged_by: str = ""


class AlertManager:
    """Manages alert rules, evaluates metrics, and dispatches notifications.

    Evaluates metric values against configured rules, respects cooldown
    periods, and dispatches alerts through registered handlers.
    """

    def __init__(self) -> None:
        self._rules: Dict[str, AlertRule] = {}
        self._fired_alerts: List[Alert] = []
        self._last_fired: Dict[str, float] = {}
        self._handlers: List[Callable[[Alert], None]] = []

    def add_rule(self, rule: AlertRule) -> None:
        """Register an alert rule.

        Args:
            rule: The alert rule to add.
        """
        self._rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> None:
        """Remove an alert rule."""
        if rule_id not in self._rules:
            raise KeyError(f"Rule {rule_id} not found")
        del self._rules[rule_id]

    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register a notification handler called when alerts fire."""
        self._handlers.append(handler)

    def evaluate(
        self,
        metric_name: str,
        value: float,
        model_name: Optional[str] = None,
        version: Optional[int] = None,
    ) -> List[Alert]:
        """Evaluate a metric value against all matching rules.

        Args:
            metric_name: Name of the metric.
            value: Current metric value.
            model_name: Optional model name for filtering.
            version: Optional version for filtering.

        Returns:
            List of alerts that fired.
        """
        fired: List[Alert] = []

        for rule in self._rules.values():
            if rule.metric_name != metric_name:
                continue
            if rule.model_name and rule.model_name != model_name:
                continue
            if rule.version is not None and rule.version != version:
                continue

            if not self._check_threshold(value, rule.comparator, rule.threshold):
                continue

            if self._in_cooldown(rule.rule_id, rule.cooldown_s):
                continue

            alert = Alert(
                rule=rule,
                metric_value=value,
                message=(
                    f"[{rule.severity.value.upper()}] {rule.name}: "
                    f"{metric_name}={value:.4f} {rule.comparator.value} "
                    f"{rule.threshold}"
                ),
            )
            self._fired_alerts.append(alert)
            self._last_fired[rule.rule_id] = time.time()
            fired.append(alert)

            for handler in self._handlers:
                handler(alert)

        return fired

    @staticmethod
    def _check_threshold(
        value: float, comparator: Comparator, threshold: float
    ) -> bool:
        """Check if a value breaches a threshold.

        Args:
            value: The metric value.
            comparator: Comparison operator.
            threshold: The threshold value.

        Returns:
            True if the threshold is breached.
        """
        if comparator == Comparator.GREATER_THAN:
            return value > threshold
        elif comparator == Comparator.LESS_THAN:
            return value < threshold
        elif comparator == Comparator.GREATER_EQUAL:
            return value >= threshold
        elif comparator == Comparator.LESS_EQUAL:
            return value <= threshold
        elif comparator == Comparator.EQUAL:
            return abs(value - threshold) < 1e-9
        return False

    def _in_cooldown(self, rule_id: str, cooldown_s: float) -> bool:
        """Check if a rule is in its cooldown period.

        Args:
            rule_id: The rule to check.
            cooldown_s: Cooldown duration in seconds.

        Returns:
            True if the rule fired recently and is still cooling down.
        """
        last = self._last_fired.get(rule_id)
        if last is None:
            return False
        return (time.time() - last) < cooldown_s

    def acknowledge(self, alert_index: int, by: str = "operator") -> None:
        """Acknowledge a fired alert."""
        if 0 <= alert_index < len(self._fired_alerts):
            self._fired_alerts[alert_index].acknowledged = True
            self._fired_alerts[alert_index].acknowledged_by = by

    def get_active_alerts(self) -> List[Alert]:
        """Get all unacknowledged alerts."""
        return [a for a in self._fired_alerts if not a.acknowledged]

    def get_alert_history(self, severity: Optional[Severity] = None) -> List[Alert]:
        """Get alert history, optionally filtered by severity."""
        if severity is None:
            return list(self._fired_alerts)
        return [a for a in self._fired_alerts if a.rule.severity == severity]

    def list_rules(self) -> List[AlertRule]:
        """List all configured alert rules."""
        return list(self._rules.values())


if __name__ == "__main__":
    manager = AlertManager()

    # Set up alert rules
    manager.add_rule(AlertRule(
        rule_id="high-error-rate",
        name="High Error Rate",
        metric_name="error_rate",
        comparator=Comparator.GREATER_THAN,
        threshold=0.05,
        severity=Severity.CRITICAL,
        cooldown_s=0.1,  # Short for demo
        description="Error rate exceeds 5%",
    ))
    manager.add_rule(AlertRule(
        rule_id="high-latency",
        name="High P99 Latency",
        metric_name="p99_latency_ms",
        comparator=Comparator.GREATER_THAN,
        threshold=200.0,
        severity=Severity.WARNING,
        cooldown_s=0.1,
        description="P99 latency exceeds 200ms",
    ))
    manager.add_rule(AlertRule(
        rule_id="drift-detected",
        name="Data Drift Detected",
        metric_name="drift_score",
        comparator=Comparator.GREATER_THAN,
        threshold=0.3,
        severity=Severity.WARNING,
        cooldown_s=0.1,
        description="More than 30% of features have drifted",
    ))

    # Register a console handler
    notifications: List[str] = []
    manager.register_handler(lambda a: notifications.append(a.message))

    # Simulate metric evaluations
    print("Evaluating metrics:\n")

    # Normal operation — no alerts
    alerts = manager.evaluate("error_rate", 0.02)
    print(f"error_rate=0.02: {len(alerts)} alerts")

    # High error rate — should fire
    time.sleep(0.15)
    alerts = manager.evaluate("error_rate", 0.08)
    print(f"error_rate=0.08: {len(alerts)} alerts")
    for a in alerts:
        print(f"  → {a.message}")

    # High latency — should fire
    time.sleep(0.15)
    alerts = manager.evaluate("p99_latency_ms", 350.0)
    print(f"p99_latency=350ms: {len(alerts)} alerts")
    for a in alerts:
        print(f"  → {a.message}")

    # Drift detected
    time.sleep(0.15)
    alerts = manager.evaluate("drift_score", 0.5)
    print(f"drift_score=0.5: {len(alerts)} alerts")
    for a in alerts:
        print(f"  → {a.message}")

    # Summary
    print(f"\nTotal notifications dispatched: {len(notifications)}")
    print(f"Active (unacknowledged) alerts: {len(manager.get_active_alerts())}")

    # Acknowledge first alert
    manager.acknowledge(0, by="oncall-engineer")
    print(f"After acknowledging first: {len(manager.get_active_alerts())} active")
