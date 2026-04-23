"""Tests for all four retraining policies."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retraining_policies import (
    Policy0_NoRetrain,
    Policy1_PeriodicRetrain,
    Policy2_PerformanceTriggered,
    Policy3_Hybrid,
    get_policy,
)

T0 = pd.Timestamp("2024-01-01 00:00")
T7D = pd.Timestamp("2024-01-08 00:00")
T14D = pd.Timestamp("2024-01-15 00:00")

_METRICS_OK = {"MAPE": 10.0, "RMSE": 0.5, "R2": 0.95}
_METRICS_DEGRADED = {"MAPE": 16.0, "RMSE": 1.2, "R2": 0.70}
_DRIFT_NONE = {"overall_drift": False}
_DRIFT_YES = {"overall_drift": True}
_BASE_CONFIG = {"baseline_mape": 10.0, "mape_threshold": 1.30}


# ---------------------------------------------------------------------------
# Policy 0
# ---------------------------------------------------------------------------

class TestPolicy0:
    def test_never_retrains(self):
        policy = Policy0_NoRetrain({})
        for _ in range(10):
            assert policy.should_retrain(T0, _METRICS_DEGRADED, _DRIFT_YES) is False


# ---------------------------------------------------------------------------
# Policy 1
# ---------------------------------------------------------------------------

class TestPolicy1:
    def test_retrains_on_first_call(self):
        policy = Policy1_PeriodicRetrain({"retrain_interval_days": 7})
        assert policy.should_retrain(T0, _METRICS_OK, _DRIFT_NONE) is True

    def test_no_retrain_before_interval(self):
        policy = Policy1_PeriodicRetrain({"retrain_interval_days": 7})
        policy.should_retrain(T0, _METRICS_OK, _DRIFT_NONE)  # first call sets state
        # 3 days later — should not retrain
        t3d = pd.Timestamp("2024-01-04 00:00")
        assert policy.should_retrain(t3d, _METRICS_OK, _DRIFT_NONE) is False

    def test_retrains_after_interval(self):
        policy = Policy1_PeriodicRetrain({"retrain_interval_days": 7})
        policy.should_retrain(T0, _METRICS_OK, _DRIFT_NONE)
        assert policy.should_retrain(T7D, _METRICS_OK, _DRIFT_NONE) is True

    def test_reset_clears_state(self):
        policy = Policy1_PeriodicRetrain({"retrain_interval_days": 7})
        policy.should_retrain(T0, _METRICS_OK, _DRIFT_NONE)
        policy.reset()
        assert "last_retrain_time" not in policy.state


# ---------------------------------------------------------------------------
# Policy 2
# ---------------------------------------------------------------------------

class TestPolicy2:
    def test_no_retrain_when_performance_ok(self):
        policy = Policy2_PerformanceTriggered(_BASE_CONFIG)
        assert policy.should_retrain(T0, _METRICS_OK, _DRIFT_NONE) is False

    def test_retrains_when_mape_exceeds_threshold(self):
        policy = Policy2_PerformanceTriggered(_BASE_CONFIG)
        assert policy.should_retrain(T0, _METRICS_DEGRADED, _DRIFT_NONE) is True

    def test_no_retrain_when_baseline_mape_missing(self):
        policy = Policy2_PerformanceTriggered({"mape_threshold": 1.30})
        assert policy.should_retrain(T0, _METRICS_DEGRADED, _DRIFT_NONE) is False


# ---------------------------------------------------------------------------
# Policy 3
# ---------------------------------------------------------------------------

class TestPolicy3:
    def test_retrains_on_performance_degradation(self):
        policy = Policy3_Hybrid(_BASE_CONFIG)
        assert policy.should_retrain(T0, _METRICS_DEGRADED, _DRIFT_NONE) is True

    def test_retrains_on_drift_signal(self):
        policy = Policy3_Hybrid(_BASE_CONFIG)
        assert policy.should_retrain(T0, _METRICS_OK, _DRIFT_YES) is True

    def test_no_retrain_when_all_ok(self):
        policy = Policy3_Hybrid(_BASE_CONFIG)
        assert policy.should_retrain(T0, _METRICS_OK, _DRIFT_NONE) is False

    def test_time_guard_prevents_immediate_second_retrain(self):
        policy = Policy3_Hybrid({**_BASE_CONFIG, "min_retrain_interval_hours": 168})
        policy.should_retrain(T0, _METRICS_DEGRADED, _DRIFT_NONE)  # first retrain
        # 1 hour later — time guard should block
        t1h = pd.Timestamp("2024-01-01 01:00")
        assert policy.should_retrain(t1h, _METRICS_DEGRADED, _DRIFT_YES) is False

    def test_retrains_after_cooldown(self):
        policy = Policy3_Hybrid({**_BASE_CONFIG, "min_retrain_interval_hours": 168})
        policy.should_retrain(T0, _METRICS_DEGRADED, _DRIFT_NONE)
        assert policy.should_retrain(T7D, _METRICS_DEGRADED, _DRIFT_NONE) is True

    def test_no_retrain_when_baseline_mape_missing(self):
        policy = Policy3_Hybrid({"mape_threshold": 1.30})
        assert policy.should_retrain(T0, _METRICS_DEGRADED, _DRIFT_YES) is False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestGetPolicy:
    @pytest.mark.parametrize("name,cls", [
        ("policy0", Policy0_NoRetrain),
        ("policy1", Policy1_PeriodicRetrain),
        ("policy2", Policy2_PerformanceTriggered),
        ("policy3", Policy3_Hybrid),
    ])
    def test_factory_returns_correct_class(self, name, cls):
        policy = get_policy(name, _BASE_CONFIG)
        assert isinstance(policy, cls)

    def test_factory_raises_on_unknown_policy(self):
        with pytest.raises(ValueError, match="Unknown policy"):
            get_policy("policy99", {})
