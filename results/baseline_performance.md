# Baseline Model Performance Report
**Model:** XGBoost Regressor (Tuned)
**Training Time:** 0.19 seconds

## Validation Metrics
| Metric | Value | Target | Status |
| :--- | :--- | :--- | :--- |
| **MAPE** | 43.70% | < 5% | FAIL |
| **RMSE** | 0.5271 | - | - |
| **R²** | 0.6121 | > 0.9 | WARN |

## Top 5 Features
| feature           |   importance |
|:------------------|-------------:|
| load_lag_1h       |    0.475769  |
| load_lag_168h     |    0.130188  |
| load_roll_std_24h |    0.0938207 |
| hour_of_day       |    0.0818402 |
| load_lag_24h      |    0.0573095 |
