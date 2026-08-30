"""
As Clear as Pearl — Configuration
Central configuration for the Lahore AQI Prediction System.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "saved_models"
SHAP_DIR = PROJECT_ROOT / "shap_outputs"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
SHAP_DIR.mkdir(exist_ok=True)

# ============================================================
# Lahore Coordinates
# ============================================================
LAHORE_LAT = 31.5204
LAHORE_LON = 74.3587
LAHORE_TIMEZONE = "Asia/Karachi"

# ============================================================
# API Configuration
# ============================================================
# OpenAQ (primary air quality source)
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", "")
OPENAQ_BASE_URL = "https://api.openaq.org/v3"

# OpenWeather (supplementary — optional)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

# Open-Meteo (free weather data — no key needed)
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# ============================================================
# Hopsworks Configuration (user fills in .env)
# ============================================================
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "")
FEATURE_GROUP_NAME = "lahore_aqi_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "lahore_aqi_fv"
FEATURE_VIEW_VERSION = 1
MODEL_NAME = "lahore_aqi_predictor"

# ============================================================
# Data Parameters
# ============================================================
HISTORICAL_DAYS = 365          # 1 year bootstrap
FORECAST_HOURS = 72            # 3-day forecast
SEQUENCE_LENGTH = 72           # 3-day lookback for LSTM/GRU
OPENAQ_SEARCH_RADIUS_M = 25000  # 25 km

# Open-Meteo hourly weather variables to fetch
WEATHER_VARIABLES_HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "precipitation",
]

# Column rename mapping: Open-Meteo → internal names
WEATHER_RENAME = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "dew_point_2m": "dew_point",
    "pressure_msl": "pressure",
    "cloud_cover": "cloud_cover",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "wind_gusts_10m": "wind_gusts",
    "precipitation": "precipitation",
}

# ============================================================
# Feature Column Lists
# ============================================================
POLLUTANT_FEATURES = ["pm25", "pm10", "o3", "no2", "so2", "co"]

WEATHER_FEATURES = [
    "temperature", "humidity", "dew_point", "pressure",
    "cloud_cover", "wind_speed", "wind_direction",
    "wind_gusts", "precipitation",
]

TIME_FEATURES = [
    "hour", "day_of_week", "month", "day_of_year",
    "is_weekend", "hour_sin", "hour_cos",
    "month_sin", "month_cos", "season",
]

DERIVED_FEATURES = [
    "aqi", "aqi_change_rate",
    "aqi_rolling_mean_6h", "aqi_rolling_mean_24h",
    "aqi_rolling_std_24h", "pm25_pm10_ratio",
    "wind_x", "wind_y", "temp_humidity_interaction",
    "pollution_index",
]

TARGET = "aqi"

# All input features used for training (everything except target)
ALL_INPUT_FEATURES = POLLUTANT_FEATURES + WEATHER_FEATURES + TIME_FEATURES + [
    f for f in DERIVED_FEATURES if f != TARGET
]

# ============================================================
# US EPA AQI Breakpoints  (2024 updated standards)
# Each entry: (C_low, C_high, I_low, I_high)
# ============================================================
AQI_BREAKPOINTS = {
    "pm25": [  # μg/m³, 24-hr average
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 400),
        (325.5, 500.4, 401, 500),
    ],
    "pm10": [  # μg/m³, 24-hr average
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ],
    "o3": [  # ppb, 8-hr average
        (0, 54, 0, 50),
        (55, 70, 51, 100),
        (71, 85, 101, 150),
        (86, 105, 151, 200),
        (106, 200, 201, 300),
    ],
    "no2": [  # ppb, 1-hr average
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 2049, 301, 500),
    ],
    "so2": [  # ppb, 1-hr average
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 1004, 301, 500),
    ],
    "co": [  # ppm, 8-hr average
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 50.4, 301, 500),
    ],
}

# Unit conversions: OpenAQ μg/m³ → EPA units (at 25°C, 1 atm)
# Molar volume at STP = 24.45 L/mol
UNIT_CONVERSIONS = {
    "pm25": 1.0,      # already μg/m³
    "pm10": 1.0,      # already μg/m³
    "o3":   0.5093,    # μg/m³ → ppb  (MW=48.00, factor = 24.45/48.00 * 1000)
    "no2":  0.5315,    # μg/m³ → ppb  (MW=46.0055)
    "so2":  0.3816,    # μg/m³ → ppb  (MW=64.066)
    "co":   0.000873,  # μg/m³ → ppm  (MW=28.01)
}

# ============================================================
# AQI Alert Levels & Health Recommendations
# ============================================================
AQI_LEVELS = [
    {"range": (0, 50), "label": "Good", "color": "#00e400",
     "bg": "rgba(0,228,0,0.15)", "emoji": "🟢"},
    {"range": (51, 100), "label": "Moderate", "color": "#ffff00",
     "bg": "rgba(255,255,0,0.15)", "emoji": "🟡"},
    {"range": (101, 150), "label": "Unhealthy for Sensitive Groups",
     "color": "#ff7e00", "bg": "rgba(255,126,0,0.15)", "emoji": "🟠"},
    {"range": (151, 200), "label": "Unhealthy", "color": "#ff0000",
     "bg": "rgba(255,0,0,0.15)", "emoji": "🔴"},
    {"range": (201, 300), "label": "Very Unhealthy", "color": "#8f3f97",
     "bg": "rgba(143,63,151,0.15)", "emoji": "🟣"},
    {"range": (301, 500), "label": "Hazardous", "color": "#7e0023",
     "bg": "rgba(126,0,35,0.15)", "emoji": "🟤"},
]

AQI_HEALTH_RECOMMENDATIONS = {
    "Good": (
        "Air quality is satisfactory. Enjoy outdoor activities!"
    ),
    "Moderate": (
        "Air quality is acceptable. Unusually sensitive people should "
        "consider limiting prolonged outdoor exertion."
    ),
    "Unhealthy for Sensitive Groups": (
        "Members of sensitive groups (children, elderly, those with "
        "respiratory conditions) may experience health effects. "
        "Reduce prolonged outdoor exertion."
    ),
    "Unhealthy": (
        "Everyone may begin to experience health effects. Avoid "
        "prolonged outdoor exertion. Keep windows closed."
    ),
    "Very Unhealthy": (
        "Health alert: everyone may experience more serious health "
        "effects. Avoid all outdoor exertion. Use air purifiers indoors."
    ),
    "Hazardous": (
        "Health emergency: the entire population is likely affected. "
        "Stay indoors. Seal windows and doors. Use N95 masks if going "
        "outside is unavoidable."
    ),
}

# ============================================================
# Model Hyperparameters
# ============================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

LR_CONFIG = {"alpha": 1.0}

XGB_CONFIG = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 50,
    "random_state": 42,
    "verbosity": 0,
}

LSTM_CONFIG = {
    "units_1": 64,
    "units_2": 32,
    "dropout": 0.2,
    "epochs": 100,
    "batch_size": 32,
    "patience": 15,
    "learning_rate": 0.001,
}

GRU_CONFIG = {
    "units_1": 64,
    "units_2": 32,
    "dropout": 0.2,
    "epochs": 100,
    "batch_size": 32,
    "patience": 15,
    "learning_rate": 0.001,
}

# ============================================================
# Server Configuration
# ============================================================
API_HOST = "127.0.0.1"
API_PORT = 8000
STREAMLIT_PORT = 8501


def calculate_aqi_subindex(concentration: float, pollutant: str) -> float:
    """
    Calculate US EPA AQI sub-index for a single pollutant.
    
    Formula: I_p = ((I_high - I_low) / (BP_high - BP_low)) * (C_p - BP_low) + I_low
    """
    if pollutant not in AQI_BREAKPOINTS:
        return 0.0

    # Apply unit conversion
    c = concentration * UNIT_CONVERSIONS.get(pollutant, 1.0)

    for bp_low, bp_high, i_low, i_high in AQI_BREAKPOINTS[pollutant]:
        if bp_low <= c <= bp_high:
            aqi = ((i_high - i_low) / (bp_high - bp_low)) * (c - bp_low) + i_low
            return round(aqi)

    # If above highest breakpoint, cap at 500
    if c > AQI_BREAKPOINTS[pollutant][-1][1]:
        return 500
    return 0


def calculate_overall_aqi(concentrations: dict) -> tuple[int, str]:
    """
    Calculate overall AQI as the max sub-index across all pollutants.
    Returns (aqi_value, dominant_pollutant).
    """
    max_aqi = 0
    dominant = "pm25"

    for pollutant, conc in concentrations.items():
        if pollutant in AQI_BREAKPOINTS and conc is not None and conc >= 0:
            sub = calculate_aqi_subindex(conc, pollutant)
            if sub > max_aqi:
                max_aqi = sub
                dominant = pollutant

    return int(max_aqi), dominant


def get_aqi_level(aqi_value: int) -> dict:
    """Get the AQI level info dict for a given AQI value."""
    for level in AQI_LEVELS:
        low, high = level["range"]
        if low <= aqi_value <= high:
            return level
    return AQI_LEVELS[-1]  # Hazardous if above 500
