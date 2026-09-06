"""
As Clear as Pearl — Training Pipeline
Orchestrates: load data → split → train all 4 models → compare → save → explain.
Uses Google Cloud Storage for model storage (Vertex AI free tier).
"""
import os
import logging
import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    GCP_PROJECT_ID,
    GCS_BUCKET_NAME,
    GCS_FEATURES_PREFIX,
    GCS_MODELS_PREFIX,
    FEATURE_GROUP_NAME,
    SEQUENCE_LENGTH,
    TARGET,
)
from src.feature_engineering import prepare_tabular_data, prepare_sequence_data
from src.models import LinearRegressionModel, XGBoostModel, LSTMModel, GRUModel
from src.explainability import explain_all_models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_training_data(use_cloud: bool = True) -> pd.DataFrame:
    """Load training data from GCS or local CSV."""
    if use_cloud and GCP_PROJECT_ID and GCS_BUCKET_NAME:
        try:
            from google.cloud import storage

            logger.info("Downloading features from GCS...")
            client = storage.Client(project=GCP_PROJECT_ID)
            bucket = client.bucket(GCS_BUCKET_NAME)
            blob_name = f"{GCS_FEATURES_PREFIX}{FEATURE_GROUP_NAME}.parquet"
            blob = bucket.blob(blob_name)

            if blob.exists():
                data = blob.download_as_bytes()
                df = pd.read_parquet(pd.io.common.BytesIO(data))
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.sort_values("timestamp").reset_index(drop=True)
                logger.info(f"Loaded {len(df)} rows from GCS.")
                return df
            else:
                logger.warning("Feature file not found in GCS. Falling back to local.")
        except Exception as e:
            logger.warning(f"GCS download failed: {e}. Falling back to local CSV.")

    # Local fallback
    file_path = DATA_DIR / "engineered_features.csv"
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# Split / Train / Compare
# ------------------------------------------------------------------

def split_data(df, train_ratio=0.70, val_ratio=0.15):
    """Chronological split — no shuffling (critical for time series)."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def train_all_models(train_df, val_df, test_df):
    results = {}

    # --- Tabular models ---
    X_train_tab, y_train_tab = prepare_tabular_data(train_df, target_col=TARGET)
    X_val_tab, y_val_tab = prepare_tabular_data(val_df, target_col=TARGET)
    X_test_tab, y_test_tab = prepare_tabular_data(test_df, target_col=TARGET)

    logger.info("Training LinearRegressionModel...")
    lr = LinearRegressionModel()
    lr.train(X_train_tab, y_train_tab, X_val_tab, y_val_tab)
    results["LinearRegression"] = {
        "model": lr,
        "metrics": lr.evaluate(X_test_tab, y_test_tab),
        "predictions": lr.predict(X_test_tab),
    }

    logger.info("Training XGBoostModel...")
    xgb = XGBoostModel()
    xgb.train(X_train_tab, y_train_tab, X_val_tab, y_val_tab)
    results["XGBoost"] = {
        "model": xgb,
        "metrics": xgb.evaluate(X_test_tab, y_test_tab),
        "predictions": xgb.predict(X_test_tab),
    }

    # --- Sequence models ---
    X_train_seq, y_train_seq = prepare_sequence_data(train_df, target_col=TARGET, seq_length=SEQUENCE_LENGTH)
    X_val_seq, y_val_seq = prepare_sequence_data(val_df, target_col=TARGET, seq_length=SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = prepare_sequence_data(test_df, target_col=TARGET, seq_length=SEQUENCE_LENGTH)

    logger.info("Training LSTMModel...")
    lstm = LSTMModel()
    lstm.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    results["LSTM"] = {
        "model": lstm,
        "metrics": lstm.evaluate(X_test_seq, y_test_seq),
        "predictions": lstm.predict(X_test_seq),
    }

    logger.info("Training GRUModel...")
    gru = GRUModel()
    gru.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    results["GRU"] = {
        "model": gru,
        "metrics": gru.evaluate(X_test_seq, y_test_seq),
        "predictions": gru.predict(X_test_seq),
    }

    return results


def compare_models(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, res in results.items():
        m = res["metrics"]
        rows.append({
            "Model": model_name,
            "MAE": m.get("mae", 0),
            "RMSE": m.get("rmse", 0),
            "MAPE": m.get("mape", 0),
            "R2": m.get("r2", 0),
        })
    df = pd.DataFrame(rows).sort_values("RMSE", ascending=True)
    df.to_csv(DATA_DIR / "model_comparison.csv", index=False)
    logger.info(f"Model comparison saved:\n{df}")
    return df


# ------------------------------------------------------------------
# Save models (local + GCS)
# ------------------------------------------------------------------

def save_models(results: dict, best_model_name: str):
    """Save all models locally; upload best to GCS."""
    # Local save
    for model_name, res in results.items():
        model_path = MODELS_DIR / f"{model_name}.pkl"
        joblib.dump(res["model"], model_path)
        logger.info(f"Saved {model_name} to {model_path}")

    # GCS upload
    if GCP_PROJECT_ID and GCS_BUCKET_NAME:
        try:
            from google.cloud import storage

            client = storage.Client(project=GCP_PROJECT_ID)
            bucket = client.bucket(GCS_BUCKET_NAME)

            for model_name in results:
                local_path = MODELS_DIR / f"{model_name}.pkl"
                blob_name = f"{GCS_MODELS_PREFIX}{model_name}.pkl"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(local_path))
                logger.info(f"Uploaded {model_name} to gs://{GCS_BUCKET_NAME}/{blob_name}")

            # Also upload comparison CSV
            comp_path = DATA_DIR / "model_comparison.csv"
            if comp_path.exists():
                blob = bucket.blob(f"{GCS_MODELS_PREFIX}model_comparison.csv")
                blob.upload_from_filename(str(comp_path))

            logger.info("All models uploaded to GCS (Vertex AI storage).")
        except Exception as e:
            logger.warning(f"GCS model upload failed: {e}. Models are safe locally.")
    else:
        logger.info("GCS not configured. Models saved locally only.")


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------

def run_training_pipeline(use_cloud: bool = True):
    df = load_training_data(use_cloud=use_cloud)
    train_df, val_df, test_df = split_data(df)
    results = train_all_models(train_df, val_df, test_df)
    comp_df = compare_models(results)
    best_model_name = comp_df.iloc[0]["Model"]
    save_models(results, best_model_name)

    logger.info("Running SHAP explainability...")
    explain_all_models(results, train_df, test_df)

    return results, comp_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-local", action="store_true",
        help="Force using local data instead of GCS",
    )
    args = parser.parse_args()
    run_training_pipeline(use_cloud=not args.use_local)
