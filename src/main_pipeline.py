"""Main MLOps pipeline orchestration."""
import argparse
import random
import time
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from drift_detection import DriftMonitor
from forecasting_model import LoadForecaster
from logger import get_logger
from retraining_policies import get_policy

warnings.filterwarnings("ignore")
log = get_logger(__name__)


class ForecastingPipeline:
    """
    End-to-end MLOps pipeline for drift-aware load forecasting.

    Orchestrates data loading, initial model training, sliding-window
    inference, drift detection, policy evaluation, and optional retraining.
    """

    def __init__(self, config_path: str, random_seed: int = 42) -> None:
        """
        Args:
            config_path:  Path to a policy YAML configuration file.
            random_seed:  Seed for XGBoost and all numpy/random operations.
        """
        with open(config_path) as fh:
            self.config: dict = yaml.safe_load(fh)

        self.random_seed = random_seed

        # Data
        log.info("Loading processed datasets…")
        self.train_df = pd.read_csv(self.config["data"]["train_path"], index_col=0, parse_dates=True)
        self.val_df = pd.read_csv(self.config["data"]["val_path"], index_col=0, parse_dates=True)
        self.test_df = pd.read_csv(self.config["data"]["test_path"], index_col=0, parse_dates=True)

        # Initial model
        log.info("Training initial model (seed=%d)…", random_seed)
        self.forecaster = LoadForecaster(
            model_params=self.config["model"]["hyperparameters"],
            random_seed=random_seed,
        )
        X_train, y_train = self._split_X_y(self.train_df)
        self.forecaster.fit(X_train, y_train)

        # Drift monitor
        X_val, y_val = self._split_X_y(self.val_df)
        baseline_metrics = self.forecaster.evaluate(X_val, y_val)
        log.info("Validation MAPE (baseline): %.4f%%", baseline_metrics["MAPE"])

        self.drift_monitor = DriftMonitor(
            reference_data=X_train,
            reference_target=y_train,
            features_to_monitor=self.config["drift"]["features_to_monitor"],
            psi_threshold=self.config["drift"]["psi_threshold"],
            ks_drift_count=self.config["drift"]["ks_drift_count"],
            mape_alpha=self.config["drift"]["mape_alpha"],
            mape_beta=self.config["drift"]["mape_beta"],
        )
        self.drift_monitor.set_baseline_mape(baseline_metrics["MAPE"])

        # Policy
        policy_config = dict(self.config["policy"])
        policy_config["baseline_mape"] = baseline_metrics["MAPE"]
        self.policy = get_policy(policy_config["name"], policy_config)
        log.info("Policy initialised: %s", policy_config["name"])

        # Logging buffers
        self.event_log: list[dict] = []
        self.metrics_log: list[dict] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_X_y(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Split a DataFrame into feature matrix X and target series y."""
        feature_cols = [c for c in df.columns if c not in ("load", "scenario")]
        return df[feature_cols], df["load"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, scenario_filter: Optional[str] = None) -> tuple[list[dict], list[dict]]:
        """
        Execute the sliding-window forecasting loop over the test set.

        Args:
            scenario_filter: If provided, restrict inference to rows whose
                             ``scenario`` column matches this value.

        Returns:
            Tuple of (metrics_log, event_log) — lists of per-window dicts.
        """
        if scenario_filter:
            if "scenario" not in self.test_df.columns:
                log.warning("'scenario' column not found — running on full test set.")
                test_data = self.test_df.copy()
            elif scenario_filter not in self.test_df["scenario"].unique():
                log.warning("Scenario '%s' not found in test data.", scenario_filter)
                return [], []
            else:
                test_data = self.test_df[self.test_df["scenario"] == scenario_filter].copy()
        else:
            test_data = self.test_df.copy()

        log.info("Starting inference loop: %d samples, scenario=%s", len(test_data), scenario_filter)

        window_hours: int = self.config["pipeline"]["forecast_window_hours"]
        retrain_days: int = self.config["pipeline"]["retrain_window_days"]
        step_hours: int = int(self.config["pipeline"]["step_size_hours"])
        num_retrains = 0
        start_idx = 0

        while start_idx < len(test_data):
            end_idx = min(start_idx + window_hours, len(test_data))
            window = test_data.iloc[start_idx:end_idx]
            if len(window) == 0:
                break

            current_time: pd.Timestamp = window.index[0]
            X_w, y_w = self._split_X_y(window)

            y_pred = self.forecaster.predict(X_w)
            metrics = self.forecaster.evaluate(X_w, y_w)
            drift_result = self.drift_monitor.detect_drift(X_w, y_w, metrics["MAPE"])

            self.metrics_log.append({
                "timestamp": current_time,
                "mape": metrics["MAPE"],
                "rmse": metrics["RMSE"],
                "psi": drift_result["psi"]["psi"],
                "ks_drifted": drift_result["ks"]["num_drifted"],
                "overall_drift": drift_result["overall_drift"],
            })

            if self.policy.should_retrain(current_time, metrics, drift_result):
                log.info("[%s] Retrain triggered by %s", current_time, self.config["policy"]["name"])
                t0 = time.time()

                retrain_end = current_time
                retrain_start = retrain_end - timedelta(days=retrain_days)
                full_history = pd.concat([self.train_df, self.test_df.loc[:retrain_end]])
                retrain_data = full_history.loc[retrain_start:retrain_end]

                if len(retrain_data) > 100:
                    X_r, y_r = self._split_X_y(retrain_data)
                    self.forecaster = LoadForecaster(
                        model_params=self.config["model"]["hyperparameters"],
                        random_seed=self.random_seed,
                    )
                    self.forecaster.fit(X_r, y_r)
                    elapsed = time.time() - t0
                    num_retrains += 1
                    self.event_log.append({
                        "timestamp": current_time,
                        "event": "retrain",
                        "retrain_time_seconds": elapsed,
                        "trigger_reason": "policy_trigger",
                    })
                    log.info("Retrain complete in %.2fs (total retrains: %d)", elapsed, num_retrains)
                else:
                    log.warning("Skipping retrain: insufficient history (%d rows)", len(retrain_data))

            start_idx += step_hours

        log.info("Pipeline complete. Total retrains: %d", num_retrains)
        return self.metrics_log, self.event_log

    def save_results(self, output_dir: str) -> None:
        """
        Persist metrics and event logs as CSV files under *output_dir*.

        Args:
            output_dir: Directory path (created if it does not exist).
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        policy_name = self.config["policy"]["name"]

        if self.metrics_log:
            pd.DataFrame(self.metrics_log).to_csv(out / f"{policy_name}_metrics.csv", index=False)

        events_df = pd.DataFrame(self.event_log) if self.event_log else pd.DataFrame(columns=["timestamp", "event"])
        events_df.to_csv(out / f"{policy_name}_events.csv", index=False)
        log.info("Results saved to %s/", output_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridSentinel forecasting pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to policy YAML config")
    parser.add_argument("--scenario", type=str, default=None, help="Test scenario filter")
    parser.add_argument("--output", type=str, default="results/experiments", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    pipeline = ForecastingPipeline(args.config, random_seed=args.seed)
    pipeline.run(scenario_filter=args.scenario)
    pipeline.save_results(args.output)