import pandas as pd
import numpy as np
import warnings
from drift_detection import DriftMonitor
from forecasting_model import LoadForecaster

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data():
    print("Loading datasets...")
    # Load val and test (Role 1 outputs)
    val_df = pd.read_csv('data/processed/val.csv', index_col=0, parse_dates=True)
    test_df = pd.read_csv('data/processed/test.csv', index_col=0, parse_dates=True)
    return val_df, test_df

def split_X_y(df):
    """Helper to separate features and target."""
    feature_cols = [c for c in df.columns if c not in ['load', 'scenario']]
    X = df[feature_cols]
    y = df['load']
    return X, y

def run_validation():
    # 1. Setup
    val_df, test_df = load_data()
    
    # Load the trained model from Role 2
    print("Loading Forecasting Model...")
    forecaster = LoadForecaster()
    forecaster.load_model('models/xgboost_baseline.json')
    
    # 2. Establish Baseline (using 1st half of validation)
    mid_point = len(val_df) // 2
    val_ref = val_df.iloc[:mid_point]
    val_cur = val_df.iloc[mid_point:]
    
    X_ref, y_ref = split_X_y(val_ref)
    
    # Get Baseline MAPE
    ref_metrics = forecaster.evaluate(X_ref, y_ref)
    baseline_mape = ref_metrics['MAPE']
    print(f"\n--- Establishing Baseline ---")
    print(f"Reference Data: {len(val_ref)} samples")
    print(f"Baseline MAPE: {baseline_mape:.2f}%")
    
    # 3. Initialize Monitor
    features_to_check = ['load_lag_24h', 'load_lag_168h', 'hour_of_day', 'load_roll_mean_24h']
    
    monitor = DriftMonitor(
        reference_data=X_ref,
        reference_target=y_ref,
        features_to_monitor=features_to_check,
        psi_threshold=0.25,      # Standard PSI threshold
        ks_drift_count=2,        # Flag drift if 2+ features change distribution
        mape_alpha=1.10,         # Warn if error increases by 10%
        mape_beta=1.25           # Critical if error increases by 25%
    )
    monitor.set_baseline_mape(baseline_mape)
    
    # 4. Scenario A: Stability Check (2nd half of Validation)
    print(f"\n--- Scenario A: Stability Check (Validation Data) ---")
    X_cur, y_cur = split_X_y(val_cur)
    cur_metrics = forecaster.evaluate(X_cur, y_cur)
    
    drift_result = monitor.detect_drift(
        current_data=X_cur,
        current_target=y_cur,
        recent_mape=cur_metrics['MAPE']
    )
    print(monitor.summary_report(drift_result))
    
    # 5. Scenario B: Drift Check (Test Set - Long Term Drift)
    # We look at the 'long_term_drift' period defined by Role 1
    print(f"\n--- Scenario B: Real Drift Check (Test Data - Long Term) ---")
    
    # Filter for the long term drift scenario created in Role 1
    drift_df = test_df[test_df['scenario'] == 'long_term_drift']
    
    if len(drift_df) == 0:
        print("Warning: No 'long_term_drift' labels found. Using last 1000 rows of test.")
        drift_df = test_df.tail(1000)
        
    X_drift, y_drift = split_X_y(drift_df)
    drift_metrics = forecaster.evaluate(X_drift, y_drift)
    
    drift_result = monitor.detect_drift(
        current_data=X_drift,
        current_target=y_drift,
        recent_mape=drift_metrics['MAPE']
    )
    print(monitor.summary_report(drift_result))

if __name__ == "__main__":
    run_validation()