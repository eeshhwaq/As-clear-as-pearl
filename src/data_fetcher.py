import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path

from src.config import (
    OPENAQ_BASE_URL,
    OPENAQ_API_KEY,
    LAHORE_LAT,
    LAHORE_LON,
    OPENAQ_SEARCH_RADIUS_M,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_FORECAST_URL,
    LAHORE_TIMEZONE,
    WEATHER_VARIABLES_HOURLY,
    WEATHER_RENAME,
    DATA_DIR,
    POLLUTANT_FEATURES
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAQClient:
    def __init__(self):
        self.headers = {"X-API-Key": OPENAQ_API_KEY} if OPENAQ_API_KEY else {}

    def find_locations(self, lat=LAHORE_LAT, lon=LAHORE_LON, radius_m=OPENAQ_SEARCH_RADIUS_M):
        url = f"{OPENAQ_BASE_URL}/locations"
        params = {
            "coordinates": f"{lat},{lon}",
            "radius": radius_m,
            "limit": 100
        }
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json().get("results", [])
            return [loc["id"] for loc in data]
        except Exception as e:
            logger.error(f"Error finding locations: {e}")
            return []

    def get_measurements(self, location_id, parameter, date_from, date_to, page=1, limit=1000):
        url = f"{OPENAQ_BASE_URL}/locations/{location_id}/measurements"
        params = {
            "parameter": parameter,
            "date_from": date_from,
            "date_to": date_to,
            "page": page,
            "limit": limit
        }
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception as e:
            logger.error(f"Error getting measurements for {location_id}, param {parameter}: {e}")
            return []

    def fetch_historical_data(self, days=365):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        locations = self.find_locations()
        if not locations:
            logger.warning("No locations found near Lahore.")
            return pd.DataFrame(columns=['datetime'] + POLLUTANT_FEATURES)
            
        all_data = []
        parameters = POLLUTANT_FEATURES
        
        # Batch in 30-day chunks
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=30), end_date)
            date_from_str = current_start.isoformat()
            date_to_str = current_end.isoformat()
            
            logger.info(f"Fetching OpenAQ data from {date_from_str} to {date_to_str}")
            
            for loc_id in locations:
                for param in parameters:
                    page = 1
                    while True:
                        results = self.get_measurements(loc_id, param, date_from_str, date_to_str, page=page)
                        if not results:
                            break
                        
                        for r in results:
                            dt = r.get("period", {}).get("datetimeFrom")
                            if dt:
                                dt = dt.get("utc", dt)
                            if not dt:
                                continue
                            all_data.append({
                                "datetime": pd.to_datetime(dt).tz_convert(None).replace(minute=0, second=0, microsecond=0),
                                param: r.get("value")
                            })
                        
                        if len(results) < 1000:
                            break
                        page += 1
                        time.sleep(1) # Rate limiting
                    time.sleep(1)
                    
            current_start = current_end
            
        if not all_data:
            return pd.DataFrame(columns=['datetime'] + POLLUTANT_FEATURES)
            
        df = pd.DataFrame(all_data)
        df = df.groupby('datetime').mean().reset_index()
        
        # Ensure all columns exist
        for param in POLLUTANT_FEATURES:
            if param not in df.columns:
                df[param] = None
                
        df.to_csv(DATA_DIR / 'openaq_cache.csv', index=False)
        return df

    def fetch_latest(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=24)
        
        locations = self.find_locations()
        all_data = []
        for loc_id in locations:
            for param in POLLUTANT_FEATURES:
                results = self.get_measurements(loc_id, param, start_date.isoformat(), end_date.isoformat(), limit=100)
                for r in results:
                    dt = r.get("period", {}).get("datetimeFrom")
                    if dt:
                        dt = dt.get("utc", dt)
                    if not dt:
                        continue
                    all_data.append({
                        "datetime": pd.to_datetime(dt).tz_convert(None).replace(minute=0, second=0, microsecond=0),
                        param: r.get("value")
                    })
                time.sleep(1)
                
        if not all_data:
            df = pd.DataFrame(columns=['datetime'] + POLLUTANT_FEATURES)
            df.loc[0, 'datetime'] = pd.Timestamp.now().replace(minute=0, second=0, microsecond=0)
            for p in POLLUTANT_FEATURES:
                df[p] = None
            return df
            
        df = pd.DataFrame(all_data)
        df = df.groupby('datetime').mean().reset_index()
        # Ensure all columns exist
        for param in POLLUTANT_FEATURES:
            if param not in df.columns:
                df[param] = None
                
        # Return only the latest hour
        if not df.empty:
            df = df.sort_values('datetime').tail(1).reset_index(drop=True)
            
        return df


class WeatherClient:
    def fetch_historical_weather(self, start_date, end_date):
        url = OPEN_METEO_ARCHIVE_URL
        params = {
            "latitude": LAHORE_LAT,
            "longitude": LAHORE_LON,
            "hourly": ",".join(WEATHER_VARIABLES_HOURLY),
            "timezone": LAHORE_TIMEZONE,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }
        
        try:
            # Batch in 3-month chunks if needed
            # For simplicity in this implementation, assume API handles it or chunk it
            # Open-Meteo archive allows fetching long ranges at once, but let's do safe chunking
            total_days = (end_date - start_date).days
            df_list = []
            
            curr_start = start_date
            while curr_start < end_date:
                curr_end = min(curr_start + timedelta(days=90), end_date)
                params['start_date'] = curr_start.strftime("%Y-%m-%d")
                params['end_date'] = curr_end.strftime("%Y-%m-%d")
                
                logger.info(f"Fetching Open-Meteo from {params['start_date']} to {params['end_date']}")
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'hourly' in data:
                    df = pd.DataFrame(data['hourly'])
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.rename(columns={'time': 'datetime'})
                    df = df.rename(columns=WEATHER_RENAME)
                    df_list.append(df)
                    
                curr_start = curr_end + timedelta(days=1)
                
            if df_list:
                final_df = pd.concat(df_list, ignore_index=True)
                return final_df
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching historical weather: {e}")
            return pd.DataFrame()

    def fetch_forecast_weather(self):
        url = OPEN_METEO_FORECAST_URL
        params = {
            "latitude": LAHORE_LAT,
            "longitude": LAHORE_LON,
            "hourly": ",".join(WEATHER_VARIABLES_HOURLY),
            "timezone": LAHORE_TIMEZONE,
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if 'hourly' in data:
                df = pd.DataFrame(data['hourly'])
                df['time'] = pd.to_datetime(df['time'])
                df = df.rename(columns={'time': 'datetime'})
                df = df.rename(columns=WEATHER_RENAME)
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching weather forecast: {e}")
            return pd.DataFrame()


def fetch_all_historical(days=365):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    openaq_client = OpenAQClient()
    weather_client = WeatherClient()
    
    logger.info("Fetching historical OpenAQ data...")
    df_aq = openaq_client.fetch_historical_data(days=days)
    
    logger.info("Fetching historical Weather data...")
    df_weather = weather_client.fetch_historical_weather(start_date, end_date)
    
    if df_aq.empty and df_weather.empty:
        return pd.DataFrame()
        
    if df_aq.empty:
        return df_weather
        
    if df_weather.empty:
        return df_aq
        
    # Merge and fill
    df_merged = pd.merge(df_aq, df_weather, on='datetime', how='outer')
    df_merged = df_merged.sort_values('datetime').reset_index(drop=True)
    df_merged = df_merged.ffill().interpolate()
    
    return df_merged


def fetch_current_data():
    openaq_client = OpenAQClient()
    weather_client = WeatherClient()
    
    df_aq = openaq_client.fetch_latest()
    df_weather = weather_client.fetch_forecast_weather()
    
    if df_aq.empty:
        return pd.DataFrame()
        
    current_time = df_aq['datetime'].iloc[0]
    
    if not df_weather.empty:
        # Find closest weather data
        df_weather = df_weather[df_weather['datetime'] <= current_time + timedelta(hours=1)]
        if not df_weather.empty:
            df_weather = df_weather.sort_values('datetime').tail(1)
            # Remove datetime from weather before merging to avoid conflicts
            df_weather = df_weather.drop(columns=['datetime']).reset_index(drop=True)
            df_combined = pd.concat([df_aq, df_weather], axis=1)
            return df_combined
            
    return df_aq

def fetch_forecast_data():
    weather_client = WeatherClient()
    return weather_client.fetch_forecast_weather()

