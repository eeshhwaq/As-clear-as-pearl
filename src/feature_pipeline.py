"""
As Clear as Pearl — Feature Pipeline
Orchestrates: fetch data → engineer features → store (GCS or local CSV).
"""
import argparse
import logging
import os
import pandas as pd

from src.data_fetcher import fetch_all_historical, fetch_current_data
from src.feature_engineering import engineer_features
from src.config import (
    GCP_PROJECT_ID,
    GCS_BUCKET_NAME,
    GCS_FEATURES_PREFIX,
    FEATURE_GROUP_NAME,
    DATA_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Google Cloud Storage helpers
# ------------------------------------------------------------------

def _upload_features_to_gcs(df: pd.DataFrame, backfill: bool = False):
    """Upload engineered features to a GCS bucket as a Parquet file."""
    try:
        from google.cloud import storage

        client = storage.Client(project=GCP_PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET_NAME)

        blob_name = f"{GCS_FEATURES_PREFIX}{FEATURE_GROUP_NAME}.parquet"
        blob = bucket.blob(blob_name)

        if not backfill and blob.exists():
            # Append mode: download existing, concat, re-upload
            logger.info("Downloading existing features from GCS for append...")
            existing_bytes = blob.download_as_bytes()
            existing_df = pd.read_parquet(
                pd.io.common.BytesIO(existing_bytes)
            )
            df = pd.concat([existing_df, df], ignore_index=True)
            # Deduplicate on timestamp
            if "timestamp" in df.columns:
                df = df.drop_duplicates(subset=["timestamp"], keep="last")
                df = df.sort_values("timestamp").reset_index(drop=True)

        # Upload as parquet
        parquet_bytes = df.to_parquet(index=False)
        blob.upload_from_string(parquet_bytes, content_type="application/octet-stream")
        logger.info(
            f"Uploaded {len(df)} rows to gs://{GCS_BUCKET_NAME}/{blob_name}"
        )

    except ImportError:
        logger.error(
            "google-cloud-storage is not installed. "
            "Run: pip install google-cloud-storage"
        )
    except Exception as e:
        logger.error(f"GCS upload failed: {e}. Data is safe in local CSV.")


def _download_features_from_gcs() -> pd.DataFrame:
    """Download the latest features parquet from GCS."""
    try:
        from google.cloud import storage

        client = storage.Client(project=GCP_PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob_name = f"{GCS_FEATURES_PREFIX}{FEATURE_GROUP_NAME}.parquet"
        blob = bucket.blob(blob_name)

        if blob.exists():
            data = blob.download_as_bytes()
            df = pd.read_parquet(pd.io.common.BytesIO(data))
            logger.info(f"Downloaded {len(df)} rows from GCS.")
            return df
        else:
            logger.warning("No feature file found in GCS.")
    except Exception as e:
        logger.error(f"GCS download failed: {e}")
    return pd.DataFrame()


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run_feature_pipeline(backfill: bool = False, days: int = 365):
    """Fetch → engineer → store (GCS + local CSV)."""
    logger.info(f"Starting feature pipeline. Backfill={backfill}, Days={days}")

    # 1. Fetch raw data
    if backfill:
        logger.info(f"Fetching historical data for {days} days...")
        raw_df = fetch_all_historical(days=days)
    else:
        logger.info("Fetching current data...")
        raw_df = fetch_current_data()

    if raw_df.empty:
        logger.warning("No data fetched. Pipeline terminating.")
        return

    # 2. Engineer features
    logger.info(f"Fetched {len(raw_df)} rows. Engineering features...")
    features_df = engineer_features(raw_df)

    if features_df.empty:
        logger.warning(
            "No features engineered. This usually means the raw data has no "
            "usable pollutant values for the selected date range."
        )
        return

    logger.info(f"Engineered {len(features_df)} feature rows.")

    # Rename datetime → timestamp
    if "datetime" in features_df.columns:
        features_df = features_df.rename(columns={"datetime": "timestamp"})

    # 3. Save locally
    local_path = DATA_DIR / "engineered_features.csv"
    try:
        if backfill or not local_path.exists():
            features_df.to_csv(local_path, index=False)
            logger.info(f"Saved local cache (overwrite) to {local_path}")
        else:
            features_df.to_csv(local_path, mode="a", header=False, index=False)
            logger.info(f"Appended to local cache at {local_path}")
    except Exception as e:
        logger.error(f"Error saving local cache: {e}")

    # 4. Upload to GCS (Vertex AI feature store)
    if GCP_PROJECT_ID and GCS_BUCKET_NAME:
        _upload_features_to_gcs(features_df, backfill=backfill)
    else:
        logger.info(
            "GCP_PROJECT_ID or GCS_BUCKET_NAME not set. "
            "Skipping cloud upload — using local CSV only."
        )

    logger.info("Feature pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Feature Pipeline for Lahore AQI"
    )
    parser.add_argument(
        "--backfill", action="store_true",
        help="Run in backfill mode (fetch historical data)",
    )
    parser.add_argument(
        "--days", type=int, default=365,
        help="Number of days to backfill (default: 365)",
    )
    args = parser.parse_args()

    try:
        run_feature_pipeline(backfill=args.backfill, days=args.days)
    except Exception as e:
        logger.error(f"Feature pipeline failed: {e}")
