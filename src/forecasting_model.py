"""XGBoost-based load forecasting model wrapper."""
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

from logger import get_logger

log = get_logger(__name__)


class LoadForecaster:
    """
    Wrapper for XGBoost Regressor for hourly load forecasting.

    Exposes a clean fit / predict / evaluate API that is agnostic to the
    upstream data source, making it straightforward to swap the underlying
    estimator in the future.
    """

    def __init__(
        self,
        model_params: Optional[dict] = None,
        random_seed: int = 42,
    ) -> None:
        """
        Args:
            model_params:  XGBoost hyperparameters. Defaults to
                           :meth:`_default_params` when *None*.
            random_seed:   Seed forwarded to XGBoost ``random_state``
                           for reproducibility.
        """
        self.model_params: dict = model_params or self._default_params()
        self.model_params["random_state"] = random_seed
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names: Optional[list[str]] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _default_params(self) -> dict:
        """Return sensible default XGBoost hyperparameters."""
        return {
            "n_estimators": 250,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on *X* and *y*."""
        self.feature_names = list(X.columns)
        self.model = xgb.XGBRegressor(**self.model_params)
        self.model.fit(X, y)
        log.info("Model training complete. Features: %d | Samples: %d", len(self.feature_names), len(y))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        if self.feature_names is None:
            raise ValueError("Feature names not set — model may not be fitted correctly.")
        X_aligned = X[self.feature_names]
        return self.model.predict(X_aligned)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """
        Evaluate model on *X* / *y* and return metric dictionary.

        Returns:
            dict with keys ``MAPE``, ``RMSE``, ``R2``.
        """
        y_pred = self.predict(X)
        mape = float(np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100)
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        r2 = float(r2_score(y, y_pred))
        return {"MAPE": mape, "RMSE": rmse, "R2": r2}

    def get_feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame of feature importances sorted descending."""
        if self.model is None:
            raise ValueError("Model not trained.")
        return (
            pd.DataFrame(
                {"feature": self.feature_names, "importance": self.model.feature_importances_}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save_model(self, path: str) -> None:
        """
        Persist model weights and feature list to *path*.

        Feature names are stored alongside the model so that
        :meth:`load_model` can restore column alignment.
        """
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        params_path = path.replace(".json", "_params.json")
        with open(params_path, "w") as fh:
            json.dump({"features": self.feature_names}, fh)
        log.info("Model saved to %s", path)

    def load_model(self, path: str) -> None:
        """
        Restore a previously saved model from *path*.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)
        params_path = path.replace(".json", "_params.json")
        if os.path.exists(params_path):
            with open(params_path) as fh:
                self.feature_names = json.load(fh).get("features")
        log.info("Model loaded from %s", path)