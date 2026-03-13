# GridSentinel — Drift-Aware MLOps Pipeline

> An end-to-end MLOps pipeline for household electricity load forecasting with automated concept drift detection and policy-based model retraining.

---

## Overview

GridSentinel trains an XGBoost model on the [UCI Household Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) dataset and continuously monitors for distribution and performance drift. When drift is detected, a configurable retraining policy determines whether — and how — the model is updated.

**Key finding:** Policy 0 (no retraining) achieves the best accuracy (avg. MAPE ~42%) across all drift scenarios, with retraining policies adding computational cost without improving forecasting performance.

---

## Project Structure

```
gridsentinel/
├── configs/                  # Policy configuration files (YAML)
│   ├── policy0_config.yaml   #   No retraining (baseline)
│   ├── policy1_config.yaml   #   Periodic retraining
│   ├── policy2_config.yaml   #   Performance-triggered retraining
│   └── policy3_config.yaml   #   Hybrid (PSI + performance)
│
├── data/
│   ├── raw/                  # UCI source data
│   ├── processed/            # Train / val / test splits
│   └── weather/              # Weather covariates
│
├── models/                   # Saved XGBoost model artefacts
│
├── results/
│   ├── figures/              # Evaluation plots
│   └── tables/               # LaTeX performance tables
│
├── src/                      # Core pipeline modules
│   ├── data_processing.py    # Feature engineering & splitting
│   ├── drift_detection.py    # PSI, KS-test, MAPE monitors
│   ├── evaluation.py         # MAPE, RMSE, MAE metrics
│   ├── forecasting_model.py  # XGBoost wrapper
│   ├── main_pipeline.py      # Orchestration entry point
│   └── retraining_policies.py# Policy 0-3 logic
│
├── requirements.txt
└── README.md
```

---

## Drift Scenarios

| Scenario | Description |
|---|---|
| `baseline` | No injected drift — stable distribution |
| `holiday_drift` | Anomalous consumption during public holidays |
| `long_term_drift` | Gradual trend shift over several months |
| `seasonal_drift_summer` | Consumption pattern shift in summer |
| `seasonal_drift_winter` | Consumption pattern shift in winter |

---

## Retraining Policies

| Policy | Trigger | Description |
|---|---|---|
| **0** | None | Model never retrained after initial training |
| **1** | Periodic | Retrain on a fixed weekly schedule |
| **2** | Performance | Retrain when MAPE exceeds threshold |
| **3** | Hybrid | Retrain on PSI drift signal OR performance drop |

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline for a specific policy and scenario
python src/main_pipeline.py \
    --config configs/policy0_config.yaml \
    --scenario baseline \
    --output results/
```

---

## Results Summary

| Policy | Avg. MAPE (%) | Avg. Retrains |
|---|---|---|
| Policy 0 (No Retrain) | **42.11 ± 0.19** | 0 |
| Policy 1 (Periodic) | 52.32 ± 0.73 | ~9 |
| Policy 2 (Performance) | 43.24 ± 1.02 | ~2 |
| Policy 3 (Hybrid) | 52.29 ± 0.64 | ~9 |

Results are averaged over 10 independent random seeds across all 5 drift scenarios.

---

## Requirements

- Python 3.10+
- `xgboost`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `pyyaml`, `matplotlib`

See `requirements.txt` for pinned versions.
