"""Tests for the LoadForecaster class."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forecasting_model import LoadForecaster


class TestLoadForecasterFit:
    def test_fit_sets_feature_names(self, trained_forecaster, feature_columns):
        assert trained_forecaster.feature_names == feature_columns

    def test_model_is_set_after_fit(self, trained_forecaster):
        assert trained_forecaster.model is not None

    def test_predict_raises_before_fit(self, sample_X):
        fc = LoadForecaster()
        with pytest.raises(ValueError, match="fit"):
            fc.predict(sample_X)


class TestLoadForecasterPredict:
    def test_returns_numpy_array(self, trained_forecaster, sample_X):
        preds = trained_forecaster.predict(sample_X)
        assert isinstance(preds, np.ndarray)

    def test_prediction_shape(self, trained_forecaster, sample_X):
        preds = trained_forecaster.predict(sample_X)
        assert preds.shape == (len(sample_X),)

    def test_predictions_are_finite(self, trained_forecaster, sample_X):
        preds = trained_forecaster.predict(sample_X)
        assert np.all(np.isfinite(preds))

    def test_column_order_invariant(self, trained_forecaster, sample_X):
        """Predictions should be identical regardless of column order in input."""
        shuffled = sample_X[sample_X.columns[::-1]]
        preds_original = trained_forecaster.predict(sample_X)
        preds_shuffled = trained_forecaster.predict(shuffled)
        np.testing.assert_array_equal(preds_original, preds_shuffled)


class TestLoadForecasterEvaluate:
    def test_returns_mape_rmse_r2(self, trained_forecaster, sample_X, sample_y):
        metrics = trained_forecaster.evaluate(sample_X, sample_y)
        assert set(metrics.keys()) == {"MAPE", "RMSE", "R2"}

    def test_mape_is_non_negative(self, trained_forecaster, sample_X, sample_y):
        metrics = trained_forecaster.evaluate(sample_X, sample_y)
        assert metrics["MAPE"] >= 0

    def test_rmse_is_non_negative(self, trained_forecaster, sample_X, sample_y):
        metrics = trained_forecaster.evaluate(sample_X, sample_y)
        assert metrics["RMSE"] >= 0

    def test_r2_in_reasonable_range(self, trained_forecaster, sample_X, sample_y):
        metrics = trained_forecaster.evaluate(sample_X, sample_y)
        assert -1 <= metrics["R2"] <= 1


class TestLoadForecasterFeatureImportance:
    def test_returns_dataframe(self, trained_forecaster):
        df = trained_forecaster.get_feature_importance()
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "importance" in df.columns

    def test_sorted_descending(self, trained_forecaster):
        df = trained_forecaster.get_feature_importance()
        assert df["importance"].is_monotonic_decreasing

    def test_raises_before_fit(self):
        fc = LoadForecaster()
        with pytest.raises(ValueError):
            fc.get_feature_importance()


class TestLoadForecasterSerialization:
    def test_save_and_load(self, trained_forecaster, sample_X, tmp_path):
        model_path = str(tmp_path / "test_model.json")
        trained_forecaster.save_model(model_path)

        new_fc = LoadForecaster()
        new_fc.load_model(model_path)

        preds_original = trained_forecaster.predict(sample_X)
        preds_loaded = new_fc.predict(sample_X)
        np.testing.assert_array_almost_equal(preds_original, preds_loaded)

    def test_load_raises_on_missing_file(self):
        fc = LoadForecaster()
        with pytest.raises(FileNotFoundError):
            fc.load_model("nonexistent/model.json")

    def test_save_raises_without_fit(self, tmp_path):
        fc = LoadForecaster()
        with pytest.raises(ValueError):
            fc.save_model(str(tmp_path / "model.json"))


class TestDifferentSeeds:
    def test_different_seeds_produce_different_models(self, sample_X, sample_y):
        fc1 = LoadForecaster(random_seed=1)
        fc2 = LoadForecaster(random_seed=2)
        fc1.fit(sample_X, sample_y)
        fc2.fit(sample_X, sample_y)
        p1 = fc1.predict(sample_X)
        p2 = fc2.predict(sample_X)
        # With different seeds and subsampling, predictions should differ
        assert not np.allclose(p1, p2)

    def test_same_seed_produces_same_model(self, sample_X, sample_y):
        fc1 = LoadForecaster(random_seed=42)
        fc2 = LoadForecaster(random_seed=42)
        fc1.fit(sample_X, sample_y)
        fc2.fit(sample_X, sample_y)
        np.testing.assert_array_equal(fc1.predict(sample_X), fc2.predict(sample_X))
