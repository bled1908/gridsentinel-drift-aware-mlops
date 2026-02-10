import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# --- 1. POPULATION STABILITY INDEX (PSI) ---
def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between two distributions [cite: 1085-1093].
    """
    # Remove NaNs
    reference = reference.dropna()
    current = current.dropna()
    
    # Create quantile bins based on reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints) # Remove duplicates
    
    # Bin both distributions
    ref_binned = np.digitize(reference, breakpoints, right=True)
    cur_binned = np.digitize(current, breakpoints, right=True)
    
    # Compute proportions
    ref_counts = np.bincount(ref_binned, minlength=len(breakpoints))
    cur_counts = np.bincount(cur_binned, minlength=len(breakpoints))
    
    ref_props = ref_counts / len(reference)
    cur_props = cur_counts / len(current)
    
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    ref_props = ref_props + epsilon
    cur_props = cur_props + epsilon
    
    # PSI formula
    psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    return psi

def psi_drift_check(reference: pd.Series, current: pd.Series, threshold: float = 0.25) -> dict:
    """Check if PSI exceeds drift threshold [cite: 1096-1100]."""
    psi_value = compute_psi(reference, current)
    
    if psi_value < 0.1:
        severity = 'stable'
        is_drift = False
    elif psi_value < 0.25:
        severity = 'warning'
        is_drift = False
    else:
        severity = 'critical'
        is_drift = True
        
    # Override based on strict threshold provided
    if psi_value >= threshold:
        is_drift = True
        
    return {
        'psi': psi_value,
        'is_drift': is_drift,
        'severity': severity
    }

# --- 2. KOLMOGOROV-SMIRNOV (KS) TEST ---
def compute_ks_test(reference: pd.Series, current: pd.Series) -> dict:
    """Perform KS test for distribution equality [cite: 1106-1114]."""
    reference = reference.dropna()
    current = current.dropna()
    
    ks_stat, p_value = ks_2samp(reference, current)
    
    # Drift if p-value < 0.05 (reject null hypothesis of same distribution)
    is_drift = (p_value < 0.05)
    
    return {
        'ks_statistic': ks_stat,
        'p_value': p_value,
        'is_drift': is_drift
    }

def multi_feature_ks_test(reference_df: pd.DataFrame, current_df: pd.DataFrame, 
                          features: list, drift_count_threshold: int = 3) -> dict:
    """Run KS test on multiple features [cite: 1121-1129]."""
    per_feature_results = {}
    drifted_features = []
    
    for feature in features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
            
        ks_result = compute_ks_test(reference_df[feature], current_df[feature])
        per_feature_results[feature] = ks_result
        
        if ks_result['is_drift']:
            drifted_features.append(feature)
            
    num_drifted = len(drifted_features)
    is_drift_overall = (num_drifted >= drift_count_threshold)
    
    return {
        'per_feature': per_feature_results,
        'drifted_features': drifted_features,
        'num_drifted': num_drifted,
        'is_drift_overall': is_drift_overall
    }

# --- 3. PERFORMANCE MONITORING (MAPE) ---
def mape_drift_check(baseline_mape: float, recent_mape: float, 
                     alpha: float = 1.15, beta: float = 1.30) -> dict:
    """
    Two-threshold MAPE degradation check [cite: 1139-1156].
    alpha: Warning threshold (e.g. 1.15 = 15% worse)
    beta: Critical threshold (e.g. 1.30 = 30% worse)
    """
    if baseline_mape == 0:
        baseline_mape = 1e-10
        
    mape_ratio = recent_mape / baseline_mape
    
    is_drift = (mape_ratio >= beta)
    is_warning = (mape_ratio >= alpha)
    
    if is_drift:
        severity = 'critical'
    elif is_warning:
        severity = 'warning'
    else:
        severity = 'stable'
        
    return {
        'baseline_mape': baseline_mape,
        'recent_mape': recent_mape,
        'mape_ratio': mape_ratio,
        'is_warning': is_warning,
        'is_drift': is_drift,
        'severity': severity
    }

# --- 4. UNIFIED DRIFT MONITOR CLASS ---
class DriftMonitor:
    """Unified drift detector combining PSI, KS, and performance monitors [cite: 1170-1233]."""
    
    def __init__(self, reference_data: pd.DataFrame, reference_target: pd.Series,
                 features_to_monitor: list,
                 psi_threshold: float = 0.25,
                 ks_drift_count: int = 3,
                 mape_alpha: float = 1.15,
                 mape_beta: float = 1.30):
        
        self.reference_data = reference_data
        self.reference_target = reference_target
        self.features_to_monitor = features_to_monitor
        self.psi_threshold = psi_threshold
        self.ks_drift_count = ks_drift_count
        self.mape_alpha = mape_alpha
        self.mape_beta = mape_beta
        self.baseline_mape = None

    def set_baseline_mape(self, baseline_mape: float):
        """Set baseline MAPE from validation period."""
        self.baseline_mape = baseline_mape
        print(f"Baseline MAPE set to: {baseline_mape:.2f}%")

    def detect_drift(self, current_data: pd.DataFrame, current_target: pd.Series, 
                     recent_mape: float) -> dict:
        """Run all drift detectors on current window."""
        
        if self.baseline_mape is None:
            # Fallback if user forgot to set baseline (prevents crash)
            self.baseline_mape = recent_mape 
            print("Warning: Baseline MAPE was not set. Using current MAPE as temporary baseline.")

        # 1. PSI on target distribution (Load)
        psi_result = psi_drift_check(self.reference_target, current_target, self.psi_threshold)
        
        # 2. KS test on features
        ks_result = multi_feature_ks_test(self.reference_data, current_data, 
                                          self.features_to_monitor, self.ks_drift_count)
        
        # 3. MAPE degradation
        mape_result = mape_drift_check(self.baseline_mape, recent_mape, 
                                       self.mape_alpha, self.mape_beta)
        
        # 4. Overall drift flag (True if ANY detector flags critical drift)
        overall_drift = (psi_result['is_drift'] or 
                         ks_result['is_drift_overall'] or 
                         mape_result['is_drift'])
        
        return {
            'psi': psi_result,
            'ks': ks_result,
            'mape': mape_result,
            'overall_drift': overall_drift,
            'timestamp': pd.Timestamp.now()
        }

    def summary_report(self, drift_result: dict) -> str:
        """Generate human-readable summary of drift detection [cite: 1492-1517]."""
        lines = []
        lines.append("=== Drift Detection Summary ===")
        lines.append(f"Timestamp: {drift_result['timestamp']}")
        lines.append(f"Overall Drift Detected: {drift_result['overall_drift']}")
        
        # PSI
        psi = drift_result['psi']
        lines.append(f"[PSI] Value: {psi['psi']:.4f}, Severity: {psi['severity']}, Drift: {psi['is_drift']}")
        
        # KS
        ks = drift_result['ks']
        lines.append(f"[KS Test] Drifted features: {ks['num_drifted']}, Drift: {ks['is_drift_overall']}")
        if ks['num_drifted'] > 0:
            lines.append(f"  Features: {ks['drifted_features']}")
            
        # MAPE
        mape = drift_result['mape']
        lines.append(f"[MAPE Monitor] Ratio: {mape['mape_ratio']:.2f}x baseline")
        lines.append(f"  Base: {mape['baseline_mape']:.2f}%, Recent: {mape['recent_mape']:.2f}%")
        lines.append(f"  Severity: {mape['severity']}, Drift: {mape['is_drift']}")
        
        return "\n".join(lines)