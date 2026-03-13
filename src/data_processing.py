import pandas as pd
import numpy as np
import os

# --- Configuration Constants ---
RAW_DATA_PATH = 'data/raw/household_power_consumption.txt'
PROCESSED_PATH = 'data/processed'
RANDOM_SEED = 42

# --- 1. Data Loading & Cleaning (FIXED) ---
# --- 1. Data Loading & Cleaning (ROBUST VERSION) ---
def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads the raw text dataset.
    Manually parses Date and Time to avoid read_csv parser errors.
    """
    print(f"Loading raw data from {filepath}...")
    
    # 1. Read columns as strings first (avoiding parse_dates completely)
    df = pd.read_csv(
        filepath,
        sep=';',
        na_values=['?', 'nan', ''],
        low_memory=False,
        usecols=['Date', 'Time', 'Global_active_power']
    )
    
    print("Parsing dates manually...")
    # 2. Combine Date and Time columns manually
    # Format in file is dd/mm/yyyy and hh:mm:ss
    raw_time_str = df['Date'] + ' ' + df['Time']
    
    # 3. Convert to datetime objects
    df['timestamp'] = pd.to_datetime(raw_time_str, format='%d/%m/%Y %H:%M:%S')
    
    # 4. Set Index
    df.set_index('timestamp', inplace=True)
    
    # 5. Clean Target
    df.rename(columns={'Global_active_power': 'load'}, inplace=True)
    df['load'] = pd.to_numeric(df['load'], errors='coerce')
    
    # Return only the target column
    df = df[['load']].copy()
    
    print("Data loaded. Raw shape:", df.shape)
    return df

# --- 2. Resampling & Outlier Handling (FIXED for Pandas 2.0+) ---
def resample_and_handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples 1-min data to hourly averages and handles missing values/outliers.
    """
    print("Resampling to hourly frequency...")
    
    # FIX: Changed 'H' to 'h' because newer pandas versions enforce lowercase for hours
    df_hourly = df.resample('h').mean()
    
    # Missing Value Handling: Linear Interpolation for short gaps
    missing_before = df_hourly['load'].isna().sum()
    df_hourly['load'] = df_hourly['load'].interpolate(method='linear')
    missing_after = df_hourly['load'].isna().sum()
    print(f"Missing values handled: {missing_before} -> {missing_after}")

    # Outlier Handling: Cap at 1.5 * 99th percentile
    q99 = df_hourly['load'].quantile(0.99)
    upper_bound = q99 * 1.5
    outlier_count = (df_hourly['load'] > upper_bound).sum()
    
    df_hourly.loc[df_hourly['load'] > upper_bound, 'load'] = upper_bound
    print(f"Capped {outlier_count} outliers above {upper_bound:.4f} kW.")
    
    return df_hourly

# --- 3. Feature Engineering ---
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates lag, rolling, and calendar features.
    """
    print("Engineering features...")
    df = df.copy()
    
    # 3.1 Lag Features (Original + Phase 1 Improvements)
    df['load_lag_1h'] = df['load'].shift(1)
    df['load_lag_2h'] = df['load'].shift(2)
    df['load_lag_3h'] = df['load'].shift(3)
    df['load_lag_6h'] = df['load'].shift(6)
    df['load_lag_12h'] = df['load'].shift(12)
    df['load_lag_24h'] = df['load'].shift(24)
    df['load_lag_48h'] = df['load'].shift(48)
    df['load_lag_72h'] = df['load'].shift(72)
    df['load_lag_168h'] = df['load'].shift(168) # 1 week
    
    # 3.2 Rolling Statistics
    df['load_roll_mean_24h'] = df['load'].rolling(window=24).mean()
    df['load_roll_std_24h'] = df['load'].rolling(window=24).std()
    
    # 3.3 Fourier Features for Seasonality (Phase 1 Improvement)
    # Daily cycle (24 hours)
    df['hour_sin_daily'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos_daily'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # Weekly cycle (168 hours)
    hour_of_week = df.index.dayofweek * 24 + df.index.hour
    df['hour_sin_weekly'] = np.sin(2 * np.pi * hour_of_week / 168)
    df['hour_cos_weekly'] = np.cos(2 * np.pi * hour_of_week / 168)
    
    # Yearly cycle (8760 hours)
    day_of_year = df.index.dayofyear
    df['day_sin_yearly'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['day_cos_yearly'] = np.cos(2 * np.pi * day_of_year / 365.25)
    
    # 3.4 Calendar Features
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # Simple Season Logic (approximate for France)
    def get_season(month):
        if month in [12, 1, 2]: return 0 # Winter
        elif month in [3, 4, 5]: return 1 # Spring
        elif month in [6, 7, 8]: return 2 # Summer
        else: return 3 # Autumn
    df['season'] = df['month'].apply(get_season)
    
    # French Public Holidays (Simplified list for the dataset years 2007-2010)
    # Ideally, this would be a comprehensive external list.
    holidays = [
        '2007-01-01', '2007-05-01', '2007-07-14', '2007-12-25',
        '2008-01-01', '2008-05-01', '2008-07-14', '2008-12-25',
        '2009-01-01', '2009-05-01', '2009-07-14', '2009-12-25',
        '2010-01-01', '2010-05-01', '2010-07-14', '2010-12-25'
    ]
    df['is_holiday'] = 0
    # Mark holidays (mapped to dates)
    df.loc[df.index.normalize().isin(pd.to_datetime(holidays)), 'is_holiday'] = 1
    
    # Drop rows with NaNs created by lags/rolling (first week of data)
    df.dropna(inplace=True)
    
    return df

# --- 4. Weather Data Integration (Phase 2) ---
def load_weather_data(weather_path: str = 'data/weather/historical_weather.csv') -> pd.DataFrame:
    """
    Loads historical weather data from Open-Meteo API.
    """
    print(f"Loading weather data from {weather_path}...")
    weather = pd.read_csv(weather_path, index_col=0, parse_dates=True)
    print(f"Weather data loaded. Shape: {weather.shape}")
    return weather

def add_weather_features(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Merges weather data and creates weather-based features.
    """
    print("Adding weather features...")
    
    # Merge on timestamp index
    df = df.merge(weather, left_index=True, right_index=True, how='left')
    
    # 1. Temperature lags
    df['temp_lag_1h'] = df['temperature'].shift(1)
    df['temp_lag_24h'] = df['temperature'].shift(24)
    
    # 2. Heating/Cooling Degree Hours (base 18°C)
    base_temp = 18.0
    df['heating_degree_hours'] = np.maximum(base_temp - df['temperature'], 0)
    df['cooling_degree_hours'] = np.maximum(df['temperature'] - base_temp, 0)
    
    # 3. Temperature-Load Interactions
    df['temp_x_hour'] = df['temperature'] * df['hour_of_day']
    df['temp_x_season'] = df['temperature'] * df['season']
    
    # 4. Non-linear temperature effect
    df['temp_squared'] = df['temperature'] ** 2
    
    # 5. Apparent temperature (simplified feels-like temperature)
    # Considers humidity and wind chill
    df['apparent_temp'] = df['temperature'] - (0.4 * (df['temperature'] - 10) * (1 - df['humidity']/100))
    
    # Drop rows with NaNs from weather merge or new lags
    initial_rows = len(df)
    df.dropna(inplace=True)
    dropped_rows = initial_rows - len(df)
    print(f"Dropped {dropped_rows} rows with NaN values after weather merge.")
    print(f"Weather features added. New shape: {df.shape}")
    
    return df

# --- 5. Splitting & Scenario Labeling ---
def split_and_label_scenarios(df: pd.DataFrame):
    """
    Splits data into Train/Val/Test and applies drift labels to the Test set.
    """
    print("Splitting data and labeling drift scenarios...")
    
    # Define Split Dates based on dataset range (Dec 2006 - Nov 2010)
    # Train: 2007-01 to 2008-12 (2 Years)
    # Val: 2009-01 to 2009-06 (6 Months)
    # Test: 2009-07 to End (approx 1.5 Years)
    
    train_mask = (df.index < '2009-01-01')
    val_mask = (df.index >= '2009-01-01') & (df.index < '2009-07-01')
    test_mask = (df.index >= '2009-07-01')
    
    train = df.loc[train_mask].copy()
    val = df.loc[val_mask].copy()
    test = df.loc[test_mask].copy()
    
    # --- Scenario Labeling (Test Set Only) ---
    test['scenario'] = 'baseline' # Default
    
    # 1. Seasonal Drift (Winter 2010 vs Summer 2010)
    test.loc[(test.index.month.isin([1, 2])) & (test.index.year == 2010), 'scenario'] = 'seasonal_drift_winter'
    test.loc[(test.index.month.isin([7, 8])) & (test.index.year == 2010), 'scenario'] = 'seasonal_drift_summer'
    
    # 2. Holiday Drift (Christmas 2009)
    test.loc[(test.index >= '2009-12-20') & (test.index <= '2009-12-31'), 'scenario'] = 'holiday_drift'
    
    # 3. Long Term Drift (Data from late 2010 vs 2007 training data)
    test.loc[test.index >= '2010-09-01', 'scenario'] = 'long_term_drift'
    
    return train, val, test

# --- 6. Main Orchestration ---
def build_all_datasets():
    """
    Orchestrates the full pipeline with weather data integration (Phase 2).
    """
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Please place the dataset at: {RAW_DATA_PATH}")
    
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    
    # Pipeline execution
    df_raw = load_and_clean_data(RAW_DATA_PATH)
    df_hourly = resample_and_handle_outliers(df_raw)
    df_features = create_features(df_hourly)
    
    # Phase 2: Load and merge weather data
    weather = load_weather_data()
    df_features = add_weather_features(df_features, weather)
    
    train, val, test = split_and_label_scenarios(df_features)
    
    # Final Sanity Checks
    assert train.isna().sum().sum() == 0, "Train set contains NaNs!"
    assert val.isna().sum().sum() == 0, "Val set contains NaNs!"
    assert test.isna().sum().sum() == 0, "Test set contains NaNs!"
    
    # Save
    print(f"Saving to {PROCESSED_PATH}...")
    train.to_csv(f"{PROCESSED_PATH}/train.csv")
    val.to_csv(f"{PROCESSED_PATH}/val.csv")
    test.to_csv(f"{PROCESSED_PATH}/test.csv")
    
    print("\n--- Pipeline Complete ---")
    print(f"Train samples: {len(train)}")
    print(f"Val samples:   {len(val)}")
    print(f"Test samples:  {len(test)}")
    print("Test Scenarios breakdown:")
    print(test['scenario'].value_counts())

if __name__ == "__main__":
    build_all_datasets()