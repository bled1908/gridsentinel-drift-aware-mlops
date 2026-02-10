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
    
    # 3.1 Lag Features
    df['load_lag_1h'] = df['load'].shift(1)
    df['load_lag_24h'] = df['load'].shift(24)
    df['load_lag_168h'] = df['load'].shift(168) # 1 week
    
    # 3.2 Rolling Statistics
    df['load_roll_mean_24h'] = df['load'].rolling(window=24).mean()
    df['load_roll_std_24h'] = df['load'].rolling(window=24).std()
    
    # 3.3 Calendar Features
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

# --- 4. Splitting & Scenario Labeling ---
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

# --- 5. Main Orchestration ---
def build_all_datasets():
    """
    Orchestrates the full pipeline and saves outputs.
    """
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Please place the dataset at: {RAW_DATA_PATH}")
    
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    
    # Pipeline execution
    df_raw = load_and_clean_data(RAW_DATA_PATH)
    df_hourly = resample_and_handle_outliers(df_raw)
    df_features = create_features(df_hourly)
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