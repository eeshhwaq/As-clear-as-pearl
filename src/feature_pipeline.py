import argparse
import logging
import os
import pandas as pd

from src.data_fetcher import fetch_all_historical, fetch_current_data
from src.feature_engineering import engineer_features
from src.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    DATA_DIR
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_feature_pipeline(backfill=False, days=365):
    logger.info(f"Starting feature pipeline. Backfill: {backfill}, Days: {days}")
    
    if backfill:
        logger.info(f"Fetching historical data for {days} days...")
        raw_df = fetch_all_historical(days=days)
    else:
        logger.info("Fetching current data...")
        raw_df = fetch_current_data()
        
    if raw_df.empty:
        logger.warning("No data fetched. Pipeline terminating.")
        return
        
    logger.info(f"Fetched {len(raw_df)} rows. Engineering features...")
    features_df = engineer_features(raw_df)
    
    if features_df.empty:
        logger.warning("No features engineered (possibly all NaNs). Pipeline terminating.")
        return
        
    logger.info(f"Engineered {len(features_df)} feature rows.")
    
    # Ensure timestamp is the name of the datetime column if Hopsworks expects it, 
    # but we will just rename 'datetime' to 'timestamp' for the feature group.
    if 'datetime' in features_df.columns:
        features_df = features_df.rename(columns={'datetime': 'timestamp'})
        
    # Local Cache
    local_path = DATA_DIR / 'engineered_features.csv'
    try:
        if backfill or not local_path.exists():
            features_df.to_csv(local_path, index=False)
            logger.info(f"Saved local cache (overwrite/new) to {local_path}")
        else:
            features_df.to_csv(local_path, mode='a', header=False, index=False)
            logger.info(f"Appended to local cache at {local_path}")
    except Exception as e:
        logger.error(f"Error saving local cache: {e}")

    # Hopsworks Integration
    if HOPSWORKS_API_KEY and HOPSWORKS_PROJECT_NAME:
        logger.info("Connecting to Hopsworks...")
        try:
            import hopsworks
            project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
            fs = project.get_feature_store()
            
            fg = fs.get_or_create_feature_group(
                name=FEATURE_GROUP_NAME,
                version=FEATURE_GROUP_VERSION,
                primary_key=['timestamp'],
                event_time='timestamp',
                description="Engineered features for Lahore AQI prediction"
            )
            
            logger.info("Inserting data into Hopsworks feature group...")
            fg.insert(features_df, write_options={"wait_for_job": False})
            logger.info("Hopsworks insert job triggered successfully.")
            
        except ImportError:
            logger.error("hopsworks package is not installed. Skipping Hopsworks upload.")
        except Exception as e:
            logger.error(f"Error connecting to or inserting into Hopsworks: {e}. Falling back to local CSV.")
    else:
        logger.info("Hopsworks API key or project name not set. Skipping Hopsworks upload.")
        
    logger.info("Feature pipeline completed successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Feature Pipeline for Lahore AQI")
    parser.add_argument('--backfill', action='store_true', help="Run in backfill mode (fetch historical data)")
    parser.add_argument('--days', type=int, default=365, help="Number of days to backfill (default: 365)")
    args = parser.parse_args()
    
    try:
        run_feature_pipeline(backfill=args.backfill, days=args.days)
    except Exception as e:
        logger.error(f"Feature pipeline failed: {e}")
