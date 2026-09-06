import pandas as pd
import numpy as np

from src.config import calculate_overall_aqi, ALL_INPUT_FEATURES

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw merged data (pollutant + weather columns + datetime) and adds engineered features.
    """
    if df.empty:
        return df

    df = df.copy()
    
    # Ensure datetime is sorted and set as index temporarily if needed, but we keep it as column
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 1. AQI Calculation (Target)
    def row_aqi(row):
        concentrations = {
            'pm25': row.get('pm25'),
            'pm10': row.get('pm10'),
            'o3': row.get('o3'),
            'no2': row.get('no2'),
            'so2': row.get('so2'),
            'co': row.get('co')
        }
        valid_vals = [v for v in concentrations.values() if v is not None and pd.notna(v)]
        if not valid_vals:
            return np.nan
        aqi_val, _ = calculate_overall_aqi(concentrations)
        return float(aqi_val)

    df['aqi'] = df.apply(row_aqi, axis=1)
    
    # 2. Time Features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    
    # Season: 0=spring(Mar-May), 1=summer(Jun-Aug), 2=autumn(Sep-Nov), 3=winter(Dec-Feb)
    def get_season(month):
        if month in [3, 4, 5]: return 0
        if month in [6, 7, 8]: return 1
        if month in [9, 10, 11]: return 2
        return 3
    df['season'] = df['month'].apply(get_season)
    
    # 3. Derived Features
    df['aqi_change_rate'] = df['aqi'].diff()
    df['aqi_rolling_mean_6h'] = df['aqi'].rolling(window=6, min_periods=1).mean()
    df['aqi_rolling_mean_24h'] = df['aqi'].rolling(window=24, min_periods=1).mean()
    df['aqi_rolling_std_24h'] = df['aqi'].rolling(window=24, min_periods=1).std()
    
    # pm25_pm10_ratio
    df['pm25_pm10_ratio'] = df.apply(
        lambda row: row['pm25'] / row['pm10'] if pd.notnull(row.get('pm10')) and row['pm10'] > 0 else 0,
        axis=1
    )
    
    # Wind components
    if 'wind_speed' in df.columns and 'wind_direction' in df.columns:
        df['wind_x'] = df['wind_speed'] * np.cos(df['wind_direction'] * np.pi / 180)
        df['wind_y'] = df['wind_speed'] * np.sin(df['wind_direction'] * np.pi / 180)
    else:
        df['wind_x'] = np.nan
        df['wind_y'] = np.nan
        
    # temp_humidity_interaction
    if 'temperature' in df.columns and 'humidity' in df.columns:
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
    else:
        df['temp_humidity_interaction'] = np.nan
        
    # Pollution index
    pollutants = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
    weights = {'pm25': 0.3, 'pm10': 0.2, 'o3': 0.15, 'no2': 0.15, 'so2': 0.1, 'co': 0.1}
    
    # Normalize and calculate weighted sum
    # Simple min-max scaling approximation per column for the current batch
    poll_idx_series = np.zeros(len(df))
    for p in pollutants:
        if p in df.columns:
            p_min = df[p].min()
            p_max = df[p].max()
            if pd.notnull(p_min) and pd.notnull(p_max) and p_max > p_min:
                norm_val = (df[p] - p_min) / (p_max - p_min)
                poll_idx_series += norm_val.fillna(0) * weights[p]
    df['pollution_index'] = poll_idx_series
    
    # 4. Handle NaNs
    # Drop rows where target (aqi) is NaN
    df = df.dropna(subset=['aqi'])

    # Some stations expose only a subset of pollutants. Keep those rows and
    # use zero for pollutant inputs that are unavailable for the whole batch.
    for pollutant in ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']:
        if pollutant in df.columns and df[pollutant].isna().all():
            df[pollutant] = 0.0
    
    # Forward-fill remaining NaN features
    df = df.ffill().bfill()
    
    # Drop any remaining NaN rows
    df = df.dropna()
    
    return df.reset_index(drop=True)

def prepare_tabular_data(df, target_col='aqi'):
    """
    Splits features and target. Returns X (DataFrame), y (Series).
    Only includes features in config.ALL_INPUT_FEATURES that exist in df.
    """
    feature_cols = [f for f in ALL_INPUT_FEATURES if f in df.columns]
    X = df[feature_cols]
    y = df[target_col]
    return X, y

def prepare_sequence_data(df, seq_length=72, target_col='aqi'):
    """
    Creates sliding window sequences for LSTM/GRU.
    Returns X_seq, y_seq as numpy arrays.
    """
    feature_cols = [f for f in ALL_INPUT_FEATURES if f in df.columns]
    
    data = df[feature_cols].values
    target = df[target_col].values
    
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i : i + seq_length])
        y_seq.append(target[i + seq_length])
        
    return np.array(X_seq), np.array(y_seq)

def get_feature_names(df):
    """Returns list of feature column names used."""
    return [f for f in ALL_INPUT_FEATURES if f in df.columns]
