"""Exploratory Data Analysis for Lahore AQI data."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import DATA_DIR, POLLUTANT_FEATURES, WEATHER_FEATURES, TARGET

def main():
    print("Loading data...")
    data_path = DATA_DIR / "engineered_features.csv"
    if not data_path.exists():
        print(f"Data file not found at {data_path}. Please run feature pipeline first.")
        return

    df = pd.read_csv(data_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
    
    # Create EDA output directory
    eda_dir = DATA_DIR / "eda_plots"
    eda_dir.mkdir(exist_ok=True)
    
    # Set seaborn style
    sns.set_theme(style="darkgrid")
    
    print("Generating plots...")
    
    # 1. AQI Time Series
    if TARGET in df.columns:
        plt.figure(figsize=(15, 6))
        sns.lineplot(data=df, x=df.index, y=TARGET)
        plt.title('AQI Time Series')
        plt.tight_layout()
        plt.savefig(eda_dir / '1_aqi_timeseries.png')
        plt.close()
        
    # 2. Correlation Heatmap
    cols = [c for c in POLLUTANT_FEATURES + WEATHER_FEATURES + [TARGET] if c in df.columns]
    if cols:
        plt.figure(figsize=(12, 10))
        corr = df[cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(eda_dir / '2_correlation_heatmap.png')
        plt.close()
        
    # 3. AQI Distribution
    if TARGET in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[TARGET], kde=True, bins=50)
        plt.title('AQI Distribution')
        plt.tight_layout()
        plt.savefig(eda_dir / '3_aqi_distribution.png')
        plt.close()
        
    # 4. PM2.5 vs PM10
    if 'pm25' in df.columns and 'pm10' in df.columns:
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df, x='pm25', y='pm10', alpha=0.5)
        plt.title('PM2.5 vs PM10')
        plt.tight_layout()
        plt.savefig(eda_dir / '4_pm25_vs_pm10.png')
        plt.close()
        
    # 5. AQI by hour of day
    if TARGET in df.columns and 'hour' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='hour', y=TARGET)
        plt.title('AQI by Hour of Day')
        plt.tight_layout()
        plt.savefig(eda_dir / '5_aqi_by_hour.png')
        plt.close()
        
    # 6. AQI by month
    if TARGET in df.columns and 'month' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, x='month', y=TARGET)
        plt.title('AQI by Month')
        plt.tight_layout()
        plt.savefig(eda_dir / '6_aqi_by_month.png')
        plt.close()
        
    # 7. AQI by season
    if TARGET in df.columns and 'season' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='season', y=TARGET)
        plt.title('Average AQI by Season')
        plt.tight_layout()
        plt.savefig(eda_dir / '7_aqi_by_season.png')
        plt.close()
        
    # 8. Temperature vs AQI with color = wind_speed
    if all(c in df.columns for c in ['temperature', TARGET, 'wind_speed']):
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df, x='temperature', y=TARGET, hue='wind_speed', palette='viridis', alpha=0.7)
        plt.title('Temperature vs AQI (Color: Wind Speed)')
        plt.tight_layout()
        plt.savefig(eda_dir / '8_temp_vs_aqi.png')
        plt.close()
        
    # 9. Seasonal Decomposition
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        if TARGET in df.columns:
            # Need a complete date range without missing indices for decompose
            ts = df[TARGET].resample('D').mean().interpolate()
            decomp = seasonal_decompose(ts, period=365)
            fig = decomp.plot()
            fig.set_size_inches(12, 8)
            plt.tight_layout()
            plt.savefig(eda_dir / '9_seasonal_decomposition.png')
            plt.close()
    except ImportError:
        print("statsmodels not installed, skipping seasonal decomposition.")
        
    # 10. Feature pair plots
    top_features = ['pm25', 'temperature', 'humidity', 'wind_speed', TARGET]
    avail_features = [f for f in top_features if f in df.columns]
    if len(avail_features) > 1:
        # Subsample to avoid huge plotting times
        sns.pairplot(df[avail_features].sample(min(1000, len(df))), corner=True)
        plt.tight_layout()
        plt.savefig(eda_dir / '10_pairplots.png')
        plt.close()
        
    print(f"Plots saved to {eda_dir}")
    
    print("\nSummary Statistics:")
    print(df.describe())

if __name__ == "__main__":
    main()
