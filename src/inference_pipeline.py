"""
As Clear as Pearl — Inference Pipeline
Handles loading trained models and making real predictions.
"""
import pandas as pd
import numpy as np
import hopsworks
import logging
import joblib
from pathlib import Path
from datetime import datetime, timedelta

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    HOPSWORKS_PROJECT_NAME,
    HOPSWORKS_API_KEY,
    MODEL_NAME,
    WEATHER_VARIABLES_HOURLY,
    WEATHER_RENAME,
    OPEN_METEO_FORECAST_URL,
    LAHORE_LAT,
    LAHORE_LON,
    LAHORE_TIMEZONE,
    get_aqi_level,
    calculate_overall_aqi,
    SEQUENCE_LENGTH,
    POLLUTANT_FEATURES,
    TARGET,
)
from src.feature_engineering import engineer_features, prepare_tabular_data, prepare_sequence_data

logger = logging.getLogger(__name__)


def load_best_model():
    """Load the best model from Hopsworks or local fallback."""
    # Try Hopsworks first
    if HOPSWORKS_API_KEY and HOPSWORKS_PROJECT_NAME:
        try:
            project = hopsworks.login(
                project=HOPSWORKS_PROJECT_NAME,
                api_key_value=HOPSWORKS_API_KEY,
            )
            mr = project.get_model_registry()
            best_model = mr.get_model("lahore_aqi_best_model", version=1)
            model_dir = best_model.download()
            for f in Path(model_dir).glob("*.pkl"):
                return joblib.load(f)
        except Exception as e:
            logger.warning(f"Failed to load from Hopsworks: {e}. Trying local.")

    # Local fallback — prefer XGBoost > LSTM > GRU > LR
    for name in ["XGBoost", "LSTM", "GRU", "LinearRegression"]:
        model_path = MODELS_DIR / f"{name}.pkl"
        if model_path.exists():
            logger.info(f"Loaded local model: {name}")
            return joblib.load(model_path)
    return None


def load_all_models() -> dict:
    """Load all 4 saved models from saved_models/."""
    models = {}
    for name in ["LinearRegression", "XGBoost", "LSTM", "GRU"]:
        model_path = MODELS_DIR / f"{name}.pkl"
        if model_path.exists():
            try:
                models[name] = joblib.load(model_path)
                logger.info(f"Loaded model: {name}")
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")
    return models


def _load_recent_features(hours: int = 168) -> pd.DataFrame:
    """Load recent engineered features from local cache."""
    file_path = DATA_DIR / "engineered_features.csv"
    if not file_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = pd.Timestamp.now() - pd.Timedelta(hours=hours)
        df = df[df["timestamp"] >= cutoff].copy()
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error loading recent features: {e}")
        return pd.DataFrame()


def _fetch_weather_forecast() -> pd.DataFrame:
    """Fetch 3-day weather forecast from Open-Meteo."""
    import requests

    try:
        params = {
            "latitude": LAHORE_LAT,
            "longitude": LAHORE_LON,
            "hourly": ",".join(WEATHER_VARIABLES_HOURLY),
            "timezone": LAHORE_TIMEZONE,
            "forecast_days": 3,
        }
        resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if "hourly" in data:
            df = pd.DataFrame(data["hourly"])
            df["time"] = pd.to_datetime(df["time"])
            df = df.rename(columns={"time": "datetime"})
            df = df.rename(columns=WEATHER_RENAME)
            return df
    except Exception as e:
        logger.error(f"Error fetching weather forecast: {e}")
    return pd.DataFrame()


def predict_next_3_days(model=None) -> pd.DataFrame:
    """
    Generate 72-hour (3-day) AQI forecast using the given model.

    For tabular models: builds features from weather forecast + recent history.
    For sequence models: uses the last SEQUENCE_LENGTH hours as input and
    autoregressively predicts forward.
    """
    if model is None:
        model = load_best_model()

    model_name = model.get_name() if model and hasattr(model, "get_name") else "unknown"
    is_sequence = model_name.lower() in ("lstm", "gru") if model else False

    # --- Fetch data ---
    recent_df = _load_recent_features(hours=SEQUENCE_LENGTH + 24)
    forecast_weather = _fetch_weather_forecast()

    now = pd.Timestamp.now().floor("h")
    future_dates = pd.date_range(start=now + pd.Timedelta(hours=1), periods=72, freq="h")
    predictions = []

    try:
        if model is None:
            raise RuntimeError("No trained model available")

        if is_sequence and not recent_df.empty:
            # --- Sequence model (LSTM / GRU) ---
            # Use the last SEQUENCE_LENGTH rows as the starting window
            feature_cols = [c for c in recent_df.columns if c not in ("timestamp", TARGET)]
            window = recent_df[feature_cols].tail(SEQUENCE_LENGTH).values.astype(np.float32)

            if len(window) < SEQUENCE_LENGTH:
                pad = np.zeros((SEQUENCE_LENGTH - len(window), window.shape[1]))
                window = np.vstack([pad, window])

            for i in range(72):
                x_input = window.reshape(1, SEQUENCE_LENGTH, -1)
                pred = model.predict(x_input)
                pred_val = float(pred.flatten()[0])
                predictions.append(max(0, pred_val))

                # Shift window forward
                new_row = window[-1].copy()
                window = np.vstack([window[1:], new_row.reshape(1, -1)])

        elif not is_sequence:
            # --- Tabular model (LR / XGBoost) ---
            if not forecast_weather.empty and not recent_df.empty:
                # Build features for each forecast hour using weather data
                last_row = recent_df.iloc[-1].copy()

                for i, dt in enumerate(future_dates):
                    row = last_row.copy()

                    # Update time features
                    row["hour"] = dt.hour
                    row["day_of_week"] = dt.weekday()
                    row["month"] = dt.month
                    row["day_of_year"] = dt.dayofyear
                    row["is_weekend"] = 1 if dt.weekday() >= 5 else 0
                    row["hour_sin"] = np.sin(2 * np.pi * dt.hour / 24)
                    row["hour_cos"] = np.cos(2 * np.pi * dt.hour / 24)
                    row["month_sin"] = np.sin(2 * np.pi * (dt.month - 1) / 12)
                    row["month_cos"] = np.cos(2 * np.pi * (dt.month - 1) / 12)
                    m = dt.month
                    row["season"] = (
                        0 if m in (3, 4, 5) else
                        1 if m in (6, 7, 8) else
                        2 if m in (9, 10, 11) else 3
                    )

                    # Update weather features from forecast if available
                    if i < len(forecast_weather):
                        for col in forecast_weather.columns:
                            if col != "datetime" and col in row.index:
                                row[col] = forecast_weather.iloc[i][col]

                    # Build feature vector
                    feature_cols = [c for c in row.index if c not in ("timestamp", TARGET)]
                    x_input = row[feature_cols].values.reshape(1, -1).astype(np.float64)
                    x_df = pd.DataFrame(x_input, columns=feature_cols)

                    pred = model.predict(x_df)
                    pred_val = float(pred.flatten()[0])
                    predictions.append(max(0, pred_val))

                    # Update rolling features for next prediction
                    last_row = row.copy()
            else:
                # Minimal fallback
                predictions = [50.0] * 72

        else:
            predictions = [50.0] * 72

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        predictions = [50.0] * 72

    # Clamp predictions to valid AQI range
    predictions = [min(max(0, p), 500) for p in predictions]

    result_df = pd.DataFrame({
        "datetime": future_dates[:len(predictions)],
        "predicted_aqi": predictions,
    })
    result_df["aqi_level"] = result_df["predicted_aqi"].apply(
        lambda x: get_aqi_level(int(x))["label"]
    )
    result_df["aqi_color"] = result_df["predicted_aqi"].apply(
        lambda x: get_aqi_level(int(x))["color"]
    )
    return result_df


def predict_all_models() -> dict:
    """Run predict_next_3_days with every available model."""
    models = load_all_models()
    results = {}
    for name, model in models.items():
        try:
            results[name] = predict_next_3_days(model)
        except Exception as e:
            logger.error(f"Prediction failed for {name}: {e}")
    return results


def get_current_aqi() -> dict:
    """Get the current AQI from the most recent cached data or OpenAQ."""
    recent = _load_recent_features(hours=6)

    if not recent.empty and TARGET in recent.columns:
        latest = recent.iloc[-1]
        aqi_val = int(latest.get(TARGET, 0))
        level_info = get_aqi_level(aqi_val)

        pollutants = {}
        for p in POLLUTANT_FEATURES:
            if p in latest.index and pd.notna(latest[p]):
                pollutants[p] = round(float(latest[p]), 1)

        weather = {}
        for w in ["temperature", "humidity", "wind_speed", "pressure"]:
            if w in latest.index and pd.notna(latest[w]):
                weather[w] = round(float(latest[w]), 1)

        return {
            "aqi": aqi_val,
            "level": level_info["label"],
            "color": level_info["color"],
            "emoji": level_info["emoji"],
            "pollutants": pollutants,
            "weather": weather,
            "timestamp": str(latest.get("timestamp", pd.Timestamp.now())),
        }

    # Fallback: try OpenAQ directly
    try:
        from src.data_fetcher import OpenAQClient

        client = OpenAQClient()
        df = client.fetch_latest()
        if not df.empty:
            concs = {p: df.iloc[0].get(p) for p in POLLUTANT_FEATURES if p in df.columns}
            concs = {k: v for k, v in concs.items() if pd.notna(v)}
            aqi_val, dominant = calculate_overall_aqi(concs)
            level_info = get_aqi_level(aqi_val)
            return {
                "aqi": aqi_val,
                "level": level_info["label"],
                "color": level_info["color"],
                "emoji": level_info["emoji"],
                "dominant_pollutant": dominant,
                "pollutants": {k: round(v, 1) for k, v in concs.items()},
                "weather": {},
                "timestamp": str(df.iloc[0].get("datetime", pd.Timestamp.now())),
            }
    except Exception as e:
        logger.error(f"Error fetching current AQI: {e}")

    return {
        "aqi": 0,
        "level": "Unknown",
        "color": "#888888",
        "emoji": "❓",
        "pollutants": {},
        "weather": {},
        "timestamp": pd.Timestamp.now().isoformat(),
    }


def get_historical_data(days: int = 30) -> pd.DataFrame:
    """Load historical AQI data from local cache."""
    file_path = DATA_DIR / "engineered_features.csv"
    try:
        df = pd.read_csv(file_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        return df[df["timestamp"] >= cutoff].copy().sort_values("timestamp")
    except Exception:
        return pd.DataFrame(columns=["timestamp", "aqi"])
