import pandas as pd
import numpy as np
from pathlib import Path
import time
import yaml
from datetime import timedelta
import warnings

# Import modules from previous roles
from forecasting_model import LoadForecaster
from drift_detection import DriftMonitor
from retraining_policies import get_policy

warnings.filterwarnings('ignore')

class ForecastingPipeline:
    """Main MLOps pipeline for drift-aware retraining experiments."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load Data
        print("Loading data...")
        self.train_df = pd.read_csv(self.config['data']['train_path'], index_col=0, parse_dates=True)
        self.val_df = pd.read_csv(self.config['data']['val_path'], index_col=0, parse_dates=True)
        self.test_df = pd.read_csv(self.config['data']['test_path'], index_col=0, parse_dates=True)
        
        # Initialize Forecaster
        print("Initializing Initial Model...")
        self.forecaster = LoadForecaster(model_params=self.config['model']['hyperparameters'])
        
        # Train Initial Model
        X_train, y_train = self._split_X_y(self.train_df)
        self.forecaster.fit(X_train, y_train)
        
        # Initialize Drift Monitor
        X_val, y_val = self._split_X_y(self.val_df)
        baseline_metrics = self.forecaster.evaluate(X_val, y_val)
        
        self.drift_monitor = DriftMonitor(
            reference_data=X_train,
            reference_target=y_train,
            features_to_monitor=self.config['drift']['features_to_monitor'],
            psi_threshold=self.config['drift']['psi_threshold'],
            ks_drift_count=self.config['drift']['ks_drift_count'],
            mape_alpha=self.config['drift']['mape_alpha'],
            mape_beta=self.config['drift']['mape_beta']
        )
        self.drift_monitor.set_baseline_mape(baseline_metrics['MAPE'])
        
        # Initialize Policy
        policy_config = self.config['policy']
        # Inject the dynamic baseline MAPE into policy config
        policy_config['baseline_mape'] = baseline_metrics['MAPE']
        self.policy = get_policy(policy_config['name'], policy_config)
        
        # Logging
        self.event_log = []
        self.metrics_log = []

    def _split_X_y(self, df):
        feature_cols = [c for c in df.columns if c not in ['load', 'scenario']]
        X = df[feature_cols]
        y = df['load']
        return X, y

    def run(self, scenario_filter: str = None):
        """Run pipeline on test data."""
        
        # Filter test data by scenario
        if scenario_filter:
            if 'scenario' in self.test_df.columns:
                 # Check if scenario exists
                if scenario_filter not in self.test_df['scenario'].unique():
                     print(f"Warning: Scenario '{scenario_filter}' not found in test data.")
                     return [], []
                test_data = self.test_df[self.test_df['scenario'] == scenario_filter].copy()
            else:
                print("Warning: 'scenario' column not found in test data.")
                test_data = self.test_df.copy()
        else:
            test_data = self.test_df.copy()
            
        print(f"Starting Pipeline on {len(test_data)} test samples...")
        
        forecast_window_hours = self.config['pipeline']['forecast_window_hours']
        retrain_window_days = self.config['pipeline']['retrain_window_days']
        step_size_hours = self.config['pipeline']['step_size_hours']
        
        start_idx = 0
        num_retrains = 0
        
        # Sliding Window Loop
        while start_idx < len(test_data):
            end_idx = min(start_idx + forecast_window_hours, len(test_data))
            current_window = test_data.iloc[start_idx:end_idx]
            
            if len(current_window) == 0:
                break
                
            current_time = current_window.index[0]
            
            # 1. Generate Forecast
            X_current, y_current = self._split_X_y(current_window)
            y_pred = self.forecaster.predict(X_current)
            
            # 2. Evaluate Performance
            current_metrics = self.forecaster.evaluate(X_current, y_current)
            
            # 3. Detect Drift
            drift_result = self.drift_monitor.detect_drift(
                current_data=X_current,
                current_target=y_current,
                recent_mape=current_metrics['MAPE']
            )
            
            # 4. Log Metrics
            self.metrics_log.append({
                'timestamp': current_time,
                'mape': current_metrics['MAPE'],
                'rmse': current_metrics['RMSE'],
                'psi': drift_result['psi']['psi'],
                'ks_drifted': drift_result['ks']['num_drifted'],
                'overall_drift': drift_result['overall_drift']
            })
            
            # 5. Check Retraining Policy
            should_retrain = self.policy.should_retrain(
                current_time=current_time,
                metrics=current_metrics,
                drift_signals=drift_result
            )
            
            if should_retrain:
                print(f"[{current_time}] RETRAIN TRIGGERED (Policy: {self.config['policy']['name']})")
                
                retrain_start_time = time.time()
                
                # Get recent data for retraining (lookback from current time)
                retrain_end = current_time
                retrain_start = retrain_end - timedelta(days=retrain_window_days)
                
                # Combine train + available test data up to retrain_end
                # Note: In real prod, this would be a database query. 
                # Here we simulate by concatenating and slicing.
                full_history = pd.concat([self.train_df, self.test_df.loc[:retrain_end]])
                retrain_data = full_history.loc[retrain_start:retrain_end]
                
                if len(retrain_data) > 100: # Safety check
                    X_retrain, y_retrain = self._split_X_y(retrain_data)
                    self.forecaster.fit(X_retrain, y_retrain)
                    
                    retrain_elapsed = time.time() - retrain_start_time
                    num_retrains += 1
                    
                    # Log Event
                    self.event_log.append({
                        'timestamp': current_time,
                        'event': 'retrain',
                        'retrain_time_seconds': retrain_elapsed,
                        'trigger_reason': 'policy_trigger'
                    })
                else:
                    print("Skipping retrain: Insufficient history data.")

            # Move to next window
            start_idx += int(step_size_hours) # step forward
            
        print(f"Pipeline complete. Total retrains: {num_retrains}")
        return self.metrics_log, self.event_log

    def save_results(self, output_dir: str):
        """Save metrics and events to CSV [cite: 2030-2040]."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        policy_name = self.config['policy']['name']
        
        # Save Metrics
        if self.metrics_log:
            metrics_df = pd.DataFrame(self.metrics_log)
            metrics_df.to_csv(output_path / f"{policy_name}_metrics.csv", index=False)
            
        # Save Events
        if self.event_log:
            events_df = pd.DataFrame(self.event_log)
            events_df.to_csv(output_path / f"{policy_name}_events.csv", index=False)
        else:
            # Create empty event file if no retrains occurred (for consistency)
            pd.DataFrame(columns=['timestamp', 'event']).to_csv(output_path / f"{policy_name}_events.csv", index=False)
            
        print(f"Results saved to {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--scenario', type=str, default=None, help='Test scenario filter')
    parser.add_argument('--output', type=str, default='results/experiments', help='Output dir')
    args = parser.parse_args()
    
    pipeline = ForecastingPipeline(args.config)
    pipeline.run(scenario_filter=args.scenario)
    pipeline.save_results(args.output)