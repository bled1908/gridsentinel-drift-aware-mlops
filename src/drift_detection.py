"""Drift detection utilities: PSI, KS-test, and MAPE-based monitors."""
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Population Stability Index (PSI)
# ---------------------------------------------------------------------------

def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Compute Population Stability Index between two distributions.

    Args:
        reference: Reference (training) distribution.
        current:   Current (production) distribution.
        bins:      Number of quantile bins.

    Returns:
        Float PSI value. Interpretation: <0.1 stable, 0.1–0.25 warning, >0.25 drift.
    """
    reference = reference.dropna()
    current = current.dropna()

    breakpoints = np.unique(np.percentile(reference, np.linspace(0, 100, bins + 1)))
    ref_binned = np.digitize(reference, breakpoints, right=True)
    cur_binned = np.digitize(current, breakpoints, right=True)

    ref_counts = np.bincount(ref_binned, minlength=len(breakpoints))
    cur_counts = np.bincount(cur_binned, minlength=len(breakpoints))

    eps = 1e-10
    ref_props = ref_counts / len(reference) + eps
    cur_props = cur_counts / len(current) + eps

    return float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))


def psi_drift_check(
    reference: pd.Series,
    current: pd.Series,
    threshold: float = 0.25,
) -> dict:
    """
    Check whether PSI exceeds the drift threshold.

    Returns:
        dict with keys ``psi``, ``is_drift``, ``severity``.
    """
    psi_value = compute_psi(reference, current)

    if psi_value < 0.1:
        severity, is_drift = "stable", False
    elif psi_value < 0.25:
        severity, is_drift = "warning", False
    else:
        severity, is_drift = "critical", True

    if psi_value >= threshold:
        is_drift = True

    return {"psi": psi_value, "is_drift": is_drift, "severity": severity}


# ---------------------------------------------------------------------------
# 2. Kolmogorov-Smirnov Test
# ---------------------------------------------------------------------------

def compute_ks_test(reference: pd.Series, current: pd.Series) -> dict:
    """
    Two-sample KS test for distribution equality.

    Returns:
        dict with keys ``ks_statistic``, ``p_value``, ``is_drift``.
    """
    reference = reference.dropna()
    current = current.dropna()
    ks_stat, p_value = ks_2samp(reference, current)
    return {
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "is_drift": bool(p_value < 0.05),
    }


def multi_feature_ks_test(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    drift_count_threshold: int = 3,
) -> dict:
    """
    Run KS test on multiple features and aggregate results.

    Returns:
        dict with per-feature results, list of drifted features, count, and overall flag.
    """
    per_feature_results: dict = {}
    drifted_features: list[str] = []

    for feature in features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        result = compute_ks_test(reference_df[feature], current_df[feature])
        per_feature_results[feature] = result
        if result["is_drift"]:
            drifted_features.append(feature)

    num_drifted = len(drifted_features)
    return {
        "per_feature": per_feature_results,
        "drifted_features": drifted_features,
        "num_drifted": num_drifted,
        "is_drift_overall": num_drifted >= drift_count_threshold,
    }


# ---------------------------------------------------------------------------
# 3. MAPE Performance Monitor
# ---------------------------------------------------------------------------

def mape_drift_check(
    baseline_mape: float,
    recent_mape: float,
    alpha: float = 1.15,
    beta: float = 1.30,
) -> dict:
    """
    Two-threshold MAPE degradation check.

    Args:
        baseline_mape: Validation-period MAPE used as reference.
        recent_mape:   Current window MAPE.
        alpha:         Warning multiplier (default 1.15 = 15% worse).
        beta:          Critical multiplier (default 1.30 = 30% worse).

    Returns:
        dict with keys ``baseline_mape``, ``recent_mape``, ``mape_ratio``,
        ``is_warning``, ``is_drift``, ``severity``.
    """
    if baseline_mape == 0:
        baseline_mape = 1e-10

    mape_ratio = recent_mape / baseline_mape
    is_drift = mape_ratio >= beta
    is_warning = mape_ratio >= alpha

    if is_drift:
        severity = "critical"
    elif is_warning:
        severity = "warning"
    else:
        severity = "stable"

    return {
        "baseline_mape": baseline_mape,
        "recent_mape": recent_mape,
        "mape_ratio": float(mape_ratio),
        "is_warning": is_warning,
        "is_drift": is_drift,
        "severity": severity,
    }


# ---------------------------------------------------------------------------
# 4. Unified DriftMonitor
# ---------------------------------------------------------------------------

class DriftMonitor:
    """
    Unified drift detector combining PSI, KS-test, and MAPE monitors.

    All three detectors run independently; ``overall_drift`` is set when
    *any* detector flags a critical event.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        reference_target: pd.Series,
        features_to_monitor: list[str],
        psi_threshold: float = 0.25,
        ks_drift_count: int = 3,
        mape_alpha: float = 1.15,
        mape_beta: float = 1.30,
    ) -> None:
        self.reference_data = reference_data
        self.reference_target = reference_target
        self.features_to_monitor = features_to_monitor
        self.psi_threshold = psi_threshold
        self.ks_drift_count = ks_drift_count
        self.mape_alpha = mape_alpha
        self.mape_beta = mape_beta
        self.baseline_mape: Optional[float] = None

    def set_baseline_mape(self, baseline_mape: float) -> None:
        """Store the validation-period MAPE used as the performance baseline."""
        self.baseline_mape = baseline_mape
        log.info("Baseline MAPE set to %.4f%%", baseline_mape)

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        current_target: pd.Series,
        recent_mape: float,
    ) -> dict:
        """
        Run all three drift detectors on the current window.

        Returns:
            dict with keys ``psi``, ``ks``, ``mape``, ``overall_drift``, ``timestamp``.
        """
        if self.baseline_mape is None:
            log.warning("Baseline MAPE not set — using current MAPE as temporary baseline.")
            self.baseline_mape = recent_mape

        psi_result = psi_drift_check(self.reference_target, current_target, self.psi_threshold)
        ks_result = multi_feature_ks_test(
            self.reference_data, current_data, self.features_to_monitor, self.ks_drift_count
        )
        mape_result = mape_drift_check(self.baseline_mape, recent_mape, self.mape_alpha, self.mape_beta)

        overall_drift = (
            psi_result["is_drift"] or ks_result["is_drift_overall"] or mape_result["is_drift"]
        )

        if overall_drift:
            log.warning(
                "Drift detected | PSI=%.4f (%s) | KS drifted=%d | MAPE ratio=%.2f (%s)",
                psi_result["psi"],
                psi_result["severity"],
                ks_result["num_drifted"],
                mape_result["mape_ratio"],
                mape_result["severity"],
            )

        return {
            "psi": psi_result,
            "ks": ks_result,
            "mape": mape_result,
            "overall_drift": overall_drift,
            "timestamp": pd.Timestamp.now(),
        }

    def summary_report(self, drift_result: dict) -> str:
        """Return a human-readable summary of a drift detection result."""
        lines = [
            "=== Drift Detection Summary ===",
            f"Timestamp: {drift_result['timestamp']}",
            f"Overall Drift Detected: {drift_result['overall_drift']}",
        ]
        psi = drift_result["psi"]
        lines.append(f"[PSI] Value: {psi['psi']:.4f}, Severity: {psi['severity']}, Drift: {psi['is_drift']}")
        ks = drift_result["ks"]
        lines.append(f"[KS Test] Drifted features: {ks['num_drifted']}, Drift: {ks['is_drift_overall']}")
        if ks["num_drifted"] > 0:
            lines.append(f"  Features: {ks['drifted_features']}")
        mape = drift_result["mape"]
        lines.append(f"[MAPE Monitor] Ratio: {mape['mape_ratio']:.2f}x baseline")
        lines.append(f"  Base: {mape['baseline_mape']:.2f}%, Recent: {mape['recent_mape']:.2f}%")
        lines.append(f"  Severity: {mape['severity']}, Drift: {mape['is_drift']}")
        return "\n".join(lines)