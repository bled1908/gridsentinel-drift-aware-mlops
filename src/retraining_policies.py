"""Policy-based retraining decision logic for the GridSentinel pipeline."""
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Optional

import pandas as pd

from logger import get_logger

log = get_logger(__name__)


class RetrainingPolicy(ABC):
    """Abstract base class for all retraining policies."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.state: dict = {}

    @abstractmethod
    def should_retrain(
        self,
        current_time: pd.Timestamp,
        metrics: dict[str, float],
        drift_signals: dict,
    ) -> bool:
        """Return *True* if retraining should be triggered at *current_time*."""

    def reset(self) -> None:
        """Reset internal policy state (useful between experiments)."""
        self.state = {}


# ---------------------------------------------------------------------------
# Policy 0 — No Retraining
# ---------------------------------------------------------------------------

class Policy0_NoRetrain(RetrainingPolicy):
    """Static model: never retrain after initial training."""

    def should_retrain(
        self,
        current_time: pd.Timestamp,
        metrics: dict[str, float],
        drift_signals: dict,
    ) -> bool:
        return False


# ---------------------------------------------------------------------------
# Policy 1 — Fixed Periodic Retraining
# ---------------------------------------------------------------------------

class Policy1_PeriodicRetrain(RetrainingPolicy):
    """Retrain on a fixed calendar schedule (default: every 7 days)."""

    def should_retrain(
        self,
        current_time: pd.Timestamp,
        metrics: dict[str, float],
        drift_signals: dict,
    ) -> bool:
        interval_days: int = self.config.get("retrain_interval_days", 7)

        if "last_retrain_time" not in self.state:
            self.state["last_retrain_time"] = current_time
            log.info("Policy1: initial retrain triggered at %s", current_time)
            return True

        elapsed_days = (current_time - self.state["last_retrain_time"]).total_seconds() / 86400
        if elapsed_days >= interval_days:
            self.state["last_retrain_time"] = current_time
            log.info("Policy1: periodic retrain triggered at %s (%.1f days elapsed)", current_time, elapsed_days)
            return True
        return False


# ---------------------------------------------------------------------------
# Policy 2 — Performance-Triggered Retraining
# ---------------------------------------------------------------------------

class Policy2_PerformanceTriggered(RetrainingPolicy):
    """Retrain when MAPE exceeds a threshold multiple of the baseline."""

    def should_retrain(
        self,
        current_time: pd.Timestamp,
        metrics: dict[str, float],
        drift_signals: dict,
    ) -> bool:
        mape_threshold: float = self.config.get("mape_threshold", 1.30)
        baseline_mape: Optional[float] = self.config.get("baseline_mape")

        if baseline_mape is None:
            return False

        mape_ratio = metrics["MAPE"] / baseline_mape
        if mape_ratio > mape_threshold:
            log.info(
                "Policy2: performance retrain triggered at %s (ratio=%.2f > threshold=%.2f)",
                current_time,
                mape_ratio,
                mape_threshold,
            )
            return True
        return False


# ---------------------------------------------------------------------------
# Policy 3 — Hybrid (Performance + Drift + Time Guard)
# ---------------------------------------------------------------------------

class Policy3_Hybrid(RetrainingPolicy):
    """
    Retrain when (performance degraded OR drift detected) AND the minimum
    cooldown interval has elapsed since the last retrain.
    """

    def should_retrain(
        self,
        current_time: pd.Timestamp,
        metrics: dict[str, float],
        drift_signals: dict,
    ) -> bool:
        mape_threshold: float = self.config.get("mape_threshold", 1.30)
        baseline_mape: Optional[float] = self.config.get("baseline_mape")
        min_interval_hours: int = self.config.get("min_retrain_interval_hours", 168)
        use_drift: bool = self.config.get("use_drift_signals", True)

        if baseline_mape is None:
            return False

        # Time-guard: enforce minimum cooldown
        if "last_retrain_time" in self.state:
            hours_since = (current_time - self.state["last_retrain_time"]).total_seconds() / 3600
            if hours_since < min_interval_hours:
                return False

        performance_degraded = metrics["MAPE"] / baseline_mape > mape_threshold
        drift_detected = use_drift and drift_signals.get("overall_drift", False)

        if performance_degraded or drift_detected:
            self.state["last_retrain_time"] = current_time
            log.info(
                "Policy3: hybrid retrain triggered at %s (perf=%s, drift=%s)",
                current_time,
                performance_degraded,
                drift_detected,
            )
            return True
        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_POLICY_REGISTRY: dict[str, type[RetrainingPolicy]] = {
    "policy0": Policy0_NoRetrain,
    "policy1": Policy1_PeriodicRetrain,
    "policy2": Policy2_PerformanceTriggered,
    "policy3": Policy3_Hybrid,
}


def get_policy(policy_name: str, config: dict) -> RetrainingPolicy:
    """
    Instantiate and return a :class:`RetrainingPolicy` by name.

    Raises:
        ValueError: If *policy_name* is not registered.
    """
    if policy_name not in _POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy '{policy_name}'. Valid options: {list(_POLICY_REGISTRY)}"
        )
    return _POLICY_REGISTRY[policy_name](config)