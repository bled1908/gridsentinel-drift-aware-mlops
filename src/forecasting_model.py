import numpy as np
import pandas as pd
import xgboost as xgb
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class LoadForecaster:
    """
    Wrapper for XGBoost Regressor for hourly load forecasting.
    Follows the Role 2 contract for clean fit/predict/evaluate API.
    """
    
    def __init__(self, model_params: dict = None):
        """
        Args:
            model_params (dict): XGBoost hyperparameters.
        """
        self.model_params = model_params or self._default_params()
        self.model = None
        self.feature_names = None
        
    def _default_params(self) -> dict:
        """Return sensible default hyperparameters [cite: 93-102]."""
        return {
            'n_estimators': 250,      # Slightly increased for better fit
            'max_depth': 6,           # Moderate depth to prevent overfitting
            'learning_rate': 0.05,    # Low LR for stability
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model on given features and target [cite: 104-112].
        """
        self.feature_names = list(X.columns)
        self.model = xgb.XGBRegressor(**self.model_params)
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions on test features [cite: 114-124].
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Ensure columns match training order
        X_aligned = X[self.feature_names]
        return self.model.predict(X_aligned)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Evaluate model on test data and return metrics [cite: 126-146].
        """
        y_pred = self.predict(X)
        
        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        return {
            'MAPE': mape,
            'RMSE': rmse,
            'R2': r2
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return feature importance from trained model [cite: 148-158].
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importances = self.model.feature_importances_
        df_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df_imp
    
    def save_model(self, path: str) -> None:
        """Save trained model to disk [cite: 160-163]."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        
        # Save feature names separately (crucial for alignment)
        params_path = path.replace('.json', '_params.json')
        with open(params_path, 'w') as f:
            json.dump({'features': self.feature_names}, f)
    
    def load_model(self, path: str) -> None:
        """Load trained model from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
            
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)
        
        # Load feature names
        params_path = path.replace('.json', '_params.json')
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                data = json.load(f)
                self.feature_names = data.get('features')