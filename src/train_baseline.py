import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from forecasting_model import LoadForecaster
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import xgboost as xgb

# --- Configuration ---
DATA_DIR = 'data/processed'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
FIGURES_DIR = 'results/figures'

def load_data():
    """Load train/val/test CSVs."""
    print("Loading data...")
    train = pd.read_csv(f'{DATA_DIR}/train.csv', index_col=0, parse_dates=True)
    val = pd.read_csv(f'{DATA_DIR}/val.csv', index_col=0, parse_dates=True)
    return train, val

def split_features_target(df):
    """Separates X (features) and y (target)."""
    drop_cols = ['load', 'scenario']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['load']
    return X, y

def tune_hyperparameters(X_train, y_train):
    """
    Performs Randomized Search to find better XGBoost parameters.
    """
    print("\n--- Starting Hyperparameter Tuning ---")
    
    # 1. Define the search space
    param_grid = {
        'n_estimators': [200, 400, 600],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    # 2. Setup TimeSeriesSplit (to respect temporal order during validation)
    tscv = TimeSeriesSplit(n_splits=3)
    
    # 3. Setup XGBRegressor
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)
    
    # 4. Randomized Search
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=15,  # Try 15 different combinations
        scoring='neg_mean_absolute_percentage_error',
        cv=tscv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    start_search = time.time()
    search.fit(X_train, y_train)
    print(f"Tuning finished in {time.time() - start_search:.2f}s")
    print(f"Best Params: {search.best_params_}")
    
    return search.best_params_

def main():
    # 1. Setup
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 2. Data Loading
    train_df, val_df = load_data()
    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)
    
    print(f"Training on {len(X_train)} samples.")

    # 3. Hyperparameter Tuning
    # We tune on X_train to find the best config
    best_params = tune_hyperparameters(X_train, y_train)

    # 4. Final Training with Best Params
    print("\nTraining Final Model with Tuned Parameters...")
    forecaster = LoadForecaster(model_params=best_params)
    
    start_time = time.time()
    forecaster.fit(X_train, y_train)
    duration = time.time() - start_time
    print(f"Final training completed in {duration:.2f} seconds.")

    # 5. Evaluation
    print("\nEvaluating on Validation Set...")
    metrics = forecaster.evaluate(X_val, y_val)
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"RMSE: {metrics['RMSE']:.4f} kW")
    print(f"R2:   {metrics['R2']:.4f}")

    # 6. Feature Importance
    print("\nExtracting Feature Importance...")
    imp_df = forecaster.get_feature_importance()
    top_10 = imp_df.head(10)
    print(top_10)
    
    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    plt.barh(top_10['feature'], top_10['importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/feature_importance.png')
    plt.close()

    # 7. Residual Analysis
    y_pred = forecaster.predict(X_val)
    residuals = y_val - y_pred
    
    # Plot Residuals
    plt.figure(figsize=(12, 4))
    plt.plot(y_val.index, residuals, alpha=0.5, label='Residuals')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals on Validation Set')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/residuals.png')
    plt.close()

    # 8. Save Model
    model_path = f'{MODELS_DIR}/xgboost_baseline.json'
    forecaster.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    # 9. Generate Report (Safe from ImportError now)
    try:
        top_5_table = top_10.head(5).to_markdown(index=False)
    except ImportError:
        top_5_table = top_10.head(5).to_string(index=False)

    report_content = f"""# Baseline Model Performance Report
**Model:** XGBoost Regressor (Tuned)
**Training Time:** {duration:.2f} seconds

## Validation Metrics
| Metric | Value | Target | Status |
| :--- | :--- | :--- | :--- |
| **MAPE** | {metrics['MAPE']:.2f}% | < 5% | {'PASS' if metrics['MAPE'] < 5 else 'FAIL'} |
| **RMSE** | {metrics['RMSE']:.4f} | - | - |
| **R²** | {metrics['R2']:.4f} | > 0.9 | {'PASS' if metrics['R2'] > 0.9 else 'WARN'} |

## Top 5 Features
{top_5_table}
"""
    with open(f'{RESULTS_DIR}/baseline_performance.md', 'w') as f:
        f.write(report_content)
    print(f"Report saved to {RESULTS_DIR}/baseline_performance.md")

if __name__ == "__main__":
    main()