# Data Engineering Notes (Role 1)

## 1. Data Source
* **Dataset:** Individual Household Electric Power Consumption (UCI Machine Learning Repository).
* **Location:** Sceaux, France (7km from Paris).
* **Original Range:** Dec 2006 – Nov 2010.
* **Target Variable:** `load` (derived from `Global_active_power` in kW).

## 2. Preprocessing & Cleaning
* **Resampling:** Raw 1-minute data was downsampled to **Hourly Mean**.
* **Missing Values:** Small gaps (<1.25% of data) were filled using **Linear Interpolation**.
* **Outliers:** Values exceeding $1.5 \times 99^{th}$ percentile were capped.
* **Weather:** Not included (dataset contained power metrics only).

## 3. Feature Engineering
The following features were generated for the `load` target:
* **Lags:** `load_lag_1h`, `load_lag_24h`, `load_lag_168h` (1 week).
* **Rolling Window (24h):** `load_roll_mean_24h`, `load_roll_std_24h`.
* **Calendar:** `hour_of_day`, `day_of_week`, `day_of_month`, `month`, `is_weekend`.
* **Context:** `season` (mapped to Winter/Spring/Summer/Autumn), `is_holiday` (mapped to major French public holidays).

## 4. Train / Val / Test Splits
Data was split chronologically to preventing data leakage:

| Split | Date Range | Description |
| :--- | :--- | :--- |
| **Train** | Start – Dec 31, 2008 | Baseline period for model fitting (approx 2 years). |
| **Validation** | Jan 01, 2009 – Jun 30, 2009 | Recent history for hyperparameter tuning. |
| **Test** | Jul 01, 2009 – End | Future unseen data for drift evaluation. |

## 5. Drift Scenarios (Test Set)
The **Test Set** includes a `scenario` column to evaluate model robustness under specific conditions:

* **`baseline`**: Standard behavior (default).
* **`seasonal_drift_winter`**: Jan–Feb 2010 (Test model performance during peak winter load).
* **`seasonal_drift_summer`**: Jul–Aug 2010 (Test model performance during summer vacation usage).
* **`holiday_drift`**: Dec 20–31, 2009 (Christmas/New Year period).
* **`long_term_drift`**: Sep 2010 onwards (Data furthest from the training set).