"""Shared pytest fixtures for the GridSentinel test suite."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src/ importable from the tests directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Reference distributions
# ---------------------------------------------------------------------------

@pytest.fixture
def reference_series() -> pd.Series:
    """A stable, normally distributed reference load series (N=500)."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(loc=1.5, scale=0.3, size=500))


@pytest.fixture
def similar_series(reference_series: pd.Series) -> pd.Series:
    """A series drawn from the same distribution — no drift expected."""
    rng = np.random.default_rng(99)
    return pd.Series(rng.normal(loc=1.5, scale=0.3, size=500))


@pytest.fixture
def drifted_series() -> pd.Series:
    """A series from a very different distribution — drift expected."""
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(loc=3.5, scale=0.8, size=500))


# ---------------------------------------------------------------------------
# Feature DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def feature_columns() -> list[str]:
    return [
        "load_lag_1h", "load_lag_24h", "load_lag_168h",
        "load_roll_mean_24h", "load_roll_std_24h",
        "hour_sin_daily", "hour_cos_daily",
        "hour_sin_weekly", "hour_cos_weekly",
        "day_sin_yearly", "day_cos_yearly",
        "hour_of_day", "day_of_week", "day_of_month",
        "month", "is_weekend", "season", "is_holiday",
        "load_lag_2h", "load_lag_3h", "load_lag_6h",
        "load_lag_12h", "load_lag_48h", "load_lag_72h",
    ]


@pytest.fixture
def sample_X(feature_columns: list[str]) -> pd.DataFrame:
    """A small synthetic feature DataFrame with 100 rows."""
    rng = np.random.default_rng(0)
    data = {col: rng.random(100) for col in feature_columns}
    # Make calendar columns integer-like
    for col in ("hour_of_day", "day_of_week", "day_of_month", "month",
                "is_weekend", "season", "is_holiday"):
        data[col] = (data[col] * 6).astype(int)
    return pd.DataFrame(data)


@pytest.fixture
def sample_y(sample_X: pd.DataFrame) -> pd.Series:
    """Synthetic target: roughly linear combination of a few features."""
    rng = np.random.default_rng(1)
    return pd.Series(
        sample_X["load_lag_1h"] * 2 + rng.normal(0, 0.1, len(sample_X)),
        name="load",
    )


# ---------------------------------------------------------------------------
# Trained forecaster
# ---------------------------------------------------------------------------

@pytest.fixture
def trained_forecaster(sample_X: pd.DataFrame, sample_y: pd.Series):
    """A fitted LoadForecaster instance ready for predict/evaluate calls."""
    from forecasting_model import LoadForecaster
    fc = LoadForecaster(random_seed=42)
    fc.fit(sample_X, sample_y)
    return fc
