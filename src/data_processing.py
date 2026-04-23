"""Raw data loading, feature engineering, and dataset splitting."""
import os
from typing import Optional

import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DATA_PATH = "data/raw/household_power_consumption.txt"
PROCESSED_PATH = "data/processed"


# ---------------------------------------------------------------------------
# 1. Data Loading & Cleaning
# ---------------------------------------------------------------------------

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load the UCI Household Power Consumption dataset.

    Parses date/time manually to avoid ``read_csv`` parser inconsistencies
    on some platforms.

    Args:
        filepath: Path to the raw ``;``-delimited text file.

    Returns:
        DataFrame with a ``DatetimeIndex`` and a single ``load`` column (kW).

    Raises:
        FileNotFoundError: If *filepath* does not exist.
    """
    log.info("Loading raw data from %s", filepath)
    df = pd.read_csv(
        filepath,
        sep=";",
        na_values=["?", "nan", ""],
        low_memory=False,
        usecols=["Date", "Time", "Global_active_power"],
    )

    log.debug("Parsing date/time columns manually…")
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df.set_index("timestamp", inplace=True)
    df.rename(columns={"Global_active_power": "load"}, inplace=True)
    df["load"] = pd.to_numeric(df["load"], errors="coerce")
    df = df[["load"]].copy()

    log.info("Raw data loaded. Shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# 2. Resampling & Outlier Handling
# ---------------------------------------------------------------------------

def resample_and_handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-minute data to hourly means and cap extreme outliers.

    Args:
        df: DataFrame with a ``DatetimeIndex`` and ``load`` column.

    Returns:
        Hourly resampled DataFrame with interpolated gaps and capped outliers.
    """
    log.info("Resampling to hourly frequency…")
    df_hourly = df.resample("h").mean()

    missing_before = int(df_hourly["load"].isna().sum())
    df_hourly["load"] = df_hourly["load"].interpolate(method="linear")
    missing_after = int(df_hourly["load"].isna().sum())
    log.info("Missing values: %d → %d (interpolated)", missing_before, missing_after)

    q99 = df_hourly["load"].quantile(0.99)
    upper_bound = q99 * 1.5
    outlier_count = int((df_hourly["load"] > upper_bound).sum())
    df_hourly.loc[df_hourly["load"] > upper_bound, "load"] = upper_bound
    log.info("Capped %d outliers above %.4f kW", outlier_count, upper_bound)

    return df_hourly


# ---------------------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------------------

_FRENCH_HOLIDAYS = [
    "2007-01-01", "2007-05-01", "2007-07-14", "2007-12-25",
    "2008-01-01", "2008-05-01", "2008-07-14", "2008-12-25",
    "2009-01-01", "2009-05-01", "2009-07-14", "2009-12-25",
    "2010-01-01", "2010-05-01", "2010-07-14", "2010-12-25",
]


def _get_season(month: int) -> int:
    """Return integer season code (0=Winter, 1=Spring, 2=Summer, 3=Autumn)."""
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate lag, rolling-statistics, Fourier, and calendar features.

    Args:
        df: Hourly load DataFrame.

    Returns:
        Feature-rich DataFrame (NaN rows from initial lags are dropped).
    """
    log.info("Engineering features…")
    df = df.copy()

    # Lag features
    for hours in (1, 2, 3, 6, 12, 24, 48, 72, 168):
        df[f"load_lag_{hours}h"] = df["load"].shift(hours)

    # Rolling statistics
    df["load_roll_mean_24h"] = df["load"].rolling(window=24).mean()
    df["load_roll_std_24h"] = df["load"].rolling(window=24).std()

    # Fourier features — daily, weekly, yearly cycles
    df["hour_sin_daily"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos_daily"] = np.cos(2 * np.pi * df.index.hour / 24)
    hour_of_week = df.index.dayofweek * 24 + df.index.hour
    df["hour_sin_weekly"] = np.sin(2 * np.pi * hour_of_week / 168)
    df["hour_cos_weekly"] = np.cos(2 * np.pi * hour_of_week / 168)
    df["day_sin_yearly"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df["day_cos_yearly"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

    # Calendar features
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["season"] = df["month"].apply(_get_season)

    # Public holidays
    df["is_holiday"] = 0
    holiday_dates = pd.to_datetime(_FRENCH_HOLIDAYS)
    df.loc[df.index.normalize().isin(holiday_dates), "is_holiday"] = 1

    df.dropna(inplace=True)
    log.info("Features created. Shape after dropna: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# 4. Weather Integration
# ---------------------------------------------------------------------------

def load_weather_data(weather_path: str = "data/weather/historical_weather.csv") -> pd.DataFrame:
    """Load historical weather data from a CSV file."""
    log.info("Loading weather data from %s", weather_path)
    weather = pd.read_csv(weather_path, index_col=0, parse_dates=True)
    log.info("Weather data shape: %s", weather.shape)
    return weather


def add_weather_features(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Merge weather data and derive weather-based predictive features.

    Args:
        df:      Feature-engineered load DataFrame.
        weather: Weather DataFrame indexed by timestamp.

    Returns:
        Merged DataFrame with additional temperature-derived features.
    """
    log.info("Adding weather features…")
    initial_rows = len(df)
    df = df.merge(weather, left_index=True, right_index=True, how="left")

    df["temp_lag_1h"] = df["temperature"].shift(1)
    df["temp_lag_24h"] = df["temperature"].shift(24)

    base_temp = 18.0
    df["heating_degree_hours"] = np.maximum(base_temp - df["temperature"], 0)
    df["cooling_degree_hours"] = np.maximum(df["temperature"] - base_temp, 0)
    df["temp_x_hour"] = df["temperature"] * df["hour_of_day"]
    df["temp_x_season"] = df["temperature"] * df["season"]
    df["temp_squared"] = df["temperature"] ** 2
    df["apparent_temp"] = df["temperature"] - (
        0.4 * (df["temperature"] - 10) * (1 - df["humidity"] / 100)
    )

    df.dropna(inplace=True)
    log.info(
        "Weather features added. Rows: %d → %d (dropped %d NaN rows)",
        initial_rows, len(df), initial_rows - len(df),
    )
    return df


# ---------------------------------------------------------------------------
# 5. Train / Val / Test Splitting & Scenario Labelling
# ---------------------------------------------------------------------------

def split_and_label_scenarios(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split the dataset and annotate test rows with drift scenario labels.

    Split:
        * Train: before 2009-01-01 (≈2 years)
        * Val:   2009-01 to 2009-06 (6 months)
        * Test:  2009-07 onwards    (≈1.5 years)

    Returns:
        Tuple of (train, val, test) DataFrames.
    """
    log.info("Splitting data into train / val / test…")
    train = df.loc[df.index < "2009-01-01"].copy()
    val = df.loc[(df.index >= "2009-01-01") & (df.index < "2009-07-01")].copy()
    test = df.loc[df.index >= "2009-07-01"].copy()

    # Scenario labels
    test["scenario"] = "baseline"
    test.loc[(test.index.month.isin([1, 2])) & (test.index.year == 2010), "scenario"] = "seasonal_drift_winter"
    test.loc[(test.index.month.isin([7, 8])) & (test.index.year == 2010), "scenario"] = "seasonal_drift_summer"
    test.loc[(test.index >= "2009-12-20") & (test.index <= "2009-12-31"), "scenario"] = "holiday_drift"
    test.loc[test.index >= "2010-09-01", "scenario"] = "long_term_drift"

    log.info("Split sizes — Train: %d | Val: %d | Test: %d", len(train), len(val), len(test))
    log.info("Test scenario breakdown:\n%s", test["scenario"].value_counts().to_string())
    return train, val, test


# ---------------------------------------------------------------------------
# 6. Main Orchestration
# ---------------------------------------------------------------------------

def build_all_datasets() -> None:
    """
    End-to-end data pipeline: load → clean → features → weather → split → save.

    Raises:
        FileNotFoundError: If the raw UCI file is not at ``RAW_DATA_PATH``.
    """
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Place the UCI dataset at: {RAW_DATA_PATH}")

    os.makedirs(PROCESSED_PATH, exist_ok=True)

    df_raw = load_and_clean_data(RAW_DATA_PATH)
    df_hourly = resample_and_handle_outliers(df_raw)
    df_features = create_features(df_hourly)
    weather = load_weather_data()
    df_features = add_weather_features(df_features, weather)
    train, val, test = split_and_label_scenarios(df_features)

    assert train.isna().sum().sum() == 0, "Train set contains NaNs!"
    assert val.isna().sum().sum() == 0, "Val set contains NaNs!"
    assert test.isna().sum().sum() == 0, "Test set contains NaNs!"

    train.to_csv(f"{PROCESSED_PATH}/train.csv")
    val.to_csv(f"{PROCESSED_PATH}/val.csv")
    test.to_csv(f"{PROCESSED_PATH}/test.csv")
    log.info("All processed datasets saved to %s/", PROCESSED_PATH)


if __name__ == "__main__":
    build_all_datasets()