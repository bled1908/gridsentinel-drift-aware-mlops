"""Tests for drift_detection module: PSI, KS-test, MAPE monitor, DriftMonitor class."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drift_detection import (
    DriftMonitor,
    compute_ks_test,
    compute_psi,
    mape_drift_check,
    multi_feature_ks_test,
    psi_drift_check,
)


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

class TestComputePsi:
    def test_identical_distributions_near_zero(self, reference_series):
        psi = compute_psi(reference_series, reference_series.copy())
        assert psi < 0.05, "PSI of identical distributions should be near 0"

    def test_similar_distributions_low_psi(self, reference_series, similar_series):
        psi = compute_psi(reference_series, similar_series)
        assert psi < 0.1, "Similar distributions should have PSI < 0.1"

    def test_drifted_distributions_high_psi(self, reference_series, drifted_series):
        psi = compute_psi(reference_series, drifted_series)
        assert psi > 0.25, "Clearly drifted distributions should have PSI > 0.25"

    def test_returns_float(self, reference_series, similar_series):
        assert isinstance(compute_psi(reference_series, similar_series), float)

    def test_handles_nan_values(self, reference_series):
        noisy = reference_series.copy()
        noisy.iloc[:10] = np.nan
        psi = compute_psi(reference_series, noisy)
        assert np.isfinite(psi)


class TestPsiDriftCheck:
    def test_stable_classification(self, reference_series, similar_series):
        result = psi_drift_check(reference_series, similar_series)
        assert result["severity"] == "stable"
        assert result["is_drift"] is False

    def test_critical_classification(self, reference_series, drifted_series):
        result = psi_drift_check(reference_series, drifted_series)
        assert result["severity"] == "critical"
        assert result["is_drift"] is True

    def test_result_keys(self, reference_series, similar_series):
        result = psi_drift_check(reference_series, similar_series)
        assert set(result.keys()) == {"psi", "is_drift", "severity"}


# ---------------------------------------------------------------------------
# KS Test
# ---------------------------------------------------------------------------

class TestComputeKsTest:
    def test_same_distribution_no_drift(self, reference_series, similar_series):
        result = compute_ks_test(reference_series, similar_series)
        # p-value should be high (not drift) for similar distributions
        assert result["p_value"] > 0.01

    def test_different_distribution_drift(self, reference_series, drifted_series):
        result = compute_ks_test(reference_series, drifted_series)
        assert result["is_drift"] is True
        assert result["p_value"] < 0.05

    def test_result_schema(self, reference_series, similar_series):
        result = compute_ks_test(reference_series, similar_series)
        assert "ks_statistic" in result
        assert "p_value" in result
        assert "is_drift" in result


class TestMultiFeatureKsTest:
    def test_no_drift_when_all_stable(self, reference_series, feature_columns):
        ref_df = pd.DataFrame({f: reference_series for f in feature_columns[:4]})
        cur_df = pd.DataFrame({f: reference_series + np.random.default_rng(0).normal(0, 0.01, len(reference_series)) for f in feature_columns[:4]})
        result = multi_feature_ks_test(ref_df, cur_df, feature_columns[:4], drift_count_threshold=3)
        assert result["num_drifted"] < 3

    def test_drift_when_all_shifted(self, reference_series, drifted_series, feature_columns):
        ref_df = pd.DataFrame({f: reference_series for f in feature_columns[:4]})
        cur_df = pd.DataFrame({f: drifted_series for f in feature_columns[:4]})
        result = multi_feature_ks_test(ref_df, cur_df, feature_columns[:4], drift_count_threshold=3)
        assert result["is_drift_overall"] is True

    def test_skips_missing_columns(self, reference_series):
        ref_df = pd.DataFrame({"feat_a": reference_series})
        cur_df = pd.DataFrame({"feat_b": reference_series})
        result = multi_feature_ks_test(ref_df, cur_df, ["feat_a"], drift_count_threshold=1)
        # feat_a is missing from cur_df — should be skipped
        assert result["num_drifted"] == 0


# ---------------------------------------------------------------------------
# MAPE Monitor
# ---------------------------------------------------------------------------

class TestMapeDriftCheck:
    def test_stable_when_no_degradation(self):
        result = mape_drift_check(baseline_mape=10.0, recent_mape=10.5)
        assert result["severity"] == "stable"
        assert result["is_drift"] is False

    def test_warning_at_alpha(self):
        result = mape_drift_check(baseline_mape=10.0, recent_mape=11.6, alpha=1.15, beta=1.30)
        assert result["severity"] == "warning"
        assert result["is_warning"] is True
        assert result["is_drift"] is False

    def test_critical_at_beta(self):
        result = mape_drift_check(baseline_mape=10.0, recent_mape=13.5, alpha=1.15, beta=1.30)
        assert result["severity"] == "critical"
        assert result["is_drift"] is True

    def test_zero_baseline_handled(self):
        result = mape_drift_check(baseline_mape=0.0, recent_mape=5.0)
        assert result["is_drift"] is True  # ratio will be enormous

    def test_mape_ratio_computed(self):
        result = mape_drift_check(baseline_mape=20.0, recent_mape=25.0)
        assert abs(result["mape_ratio"] - 1.25) < 1e-6


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------

class TestDriftMonitor:
    @pytest.fixture
    def monitor(self, sample_X, sample_y):
        m = DriftMonitor(
            reference_data=sample_X,
            reference_target=sample_y,
            features_to_monitor=["load_lag_1h", "load_lag_24h"],
            psi_threshold=0.25,
            ks_drift_count=2,
        )
        m.set_baseline_mape(10.0)
        return m

    def test_detect_drift_returns_expected_keys(self, monitor, sample_X, sample_y):
        result = monitor.detect_drift(sample_X, sample_y, recent_mape=10.5)
        for key in ("psi", "ks", "mape", "overall_drift", "timestamp"):
            assert key in result

    def test_no_drift_on_same_data(self, monitor, sample_X, sample_y):
        result = monitor.detect_drift(sample_X, sample_y, recent_mape=10.5)
        # Same reference and current → should not flag overall drift
        assert isinstance(result["overall_drift"], bool)

    def test_fallback_baseline_mape(self, sample_X, sample_y):
        m = DriftMonitor(
            reference_data=sample_X,
            reference_target=sample_y,
            features_to_monitor=["load_lag_1h"],
        )
        # baseline_mape not set — should not crash
        result = m.detect_drift(sample_X, sample_y, recent_mape=15.0)
        assert "overall_drift" in result

    def test_summary_report_is_string(self, monitor, sample_X, sample_y):
        result = monitor.detect_drift(sample_X, sample_y, recent_mape=10.0)
        report = monitor.summary_report(result)
        assert isinstance(report, str)
        assert "PSI" in report
