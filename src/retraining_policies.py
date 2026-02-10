from abc import ABC, abstractmethod
import pandas as pd
from datetime import timedelta

class RetrainingPolicy(ABC):
    """Abstract base class for retraining policies."""
    def __init__(self, config: dict):
        self.config = config
        self.state = {} # Internal state (e.g., last retrain time)

    @abstractmethod
    def should_retrain(self, current_time: pd.Timestamp, metrics: dict, drift_signals: dict) -> bool:
        """Decide whether to trigger retraining."""
        pass

    def reset(self):
        """Reset policy state."""
        self.state = {}

# --- POLICY 0: No Retraining ---
class Policy0_NoRetrain(RetrainingPolicy):
    """Policy 0: Static model, never retrain."""
    def should_retrain(self, current_time, metrics, drift_signals):
        return False

# --- POLICY 1: Fixed Periodic Retraining ---
class Policy1_PeriodicRetrain(RetrainingPolicy):
    """Policy 1: Retrain every N days (fixed schedule)."""
    def should_retrain(self, current_time, metrics, drift_signals):
        interval = self.config.get('retrain_interval_days', 7)
        
        if 'last_retrain_time' not in self.state:
            # First retrain: trigger immediately or at start
            self.state['last_retrain_time'] = current_time
            return True
            
        last_retrain = self.state['last_retrain_time']
        time_since_retrain = (current_time - last_retrain).total_seconds() / 86400
        
        if time_since_retrain >= interval:
            self.state['last_retrain_time'] = current_time
            return True
        return False

# --- POLICY 2: Performance-Only Triggered Retraining ---
class Policy2_PerformanceTriggered(RetrainingPolicy):
    """Policy 2: Retrain when MAPE exceeds threshold."""
    def should_retrain(self, current_time, metrics, drift_signals):
        mape_threshold = self.config.get('mape_threshold', 1.30)
        baseline_mape = self.config.get('baseline_mape')
        
        if baseline_mape is None:
            # Fallback for safety if config is missing baseline
            return False 
            
        recent_mape = metrics['MAPE']
        mape_ratio = recent_mape / baseline_mape
        
        if mape_ratio > mape_threshold:
            return True
        return False

# --- POLICY 3: Hybrid (Performance + Drift + Time Guard) ---
class Policy3_Hybrid(RetrainingPolicy):
    """Policy 3: Retrain if (Performance degraded OR drift detected) AND time guard satisfied."""
    def should_retrain(self, current_time, metrics, drift_signals):
        mape_threshold = self.config.get('mape_threshold', 1.30)
        baseline_mape = self.config.get('baseline_mape')
        min_interval_hours = self.config.get('min_retrain_interval_hours', 168) # 7 days
        use_drift = self.config.get('use_drift_signals', True)
        
        if baseline_mape is None:
            return False

        # 1. Check time guard
        if 'last_retrain_time' in self.state:
            last_retrain = self.state['last_retrain_time']
            hours_since_retrain = (current_time - last_retrain).total_seconds() / 3600
            if hours_since_retrain < min_interval_hours:
                return False # Too soon to retrain
        
        # 2. Check performance degradation
        recent_mape = metrics['MAPE']
        mape_ratio = recent_mape / baseline_mape
        performance_degraded = (mape_ratio > mape_threshold)
        
        # 3. Check drift signals
        drift_detected = False
        if use_drift:
            drift_detected = drift_signals.get('overall_drift', False)
            
        # 4. Trigger if either condition met
        if performance_degraded or drift_detected:
            self.state['last_retrain_time'] = current_time
            return True
            
        return False

# --- POLICY FACTORY ---
def get_policy(policy_name: str, config: dict) -> RetrainingPolicy:
    policies = {
        'policy0': Policy0_NoRetrain,
        'policy1': Policy1_PeriodicRetrain,
        'policy2': Policy2_PerformanceTriggered,
        'policy3': Policy3_Hybrid
    }
    
    if policy_name not in policies:
        raise ValueError(f"Unknown policy: {policy_name}")
        
    return policies[policy_name](config)