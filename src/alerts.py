import pandas as pd
from src.config import AQI_LEVELS, AQI_HEALTH_RECOMMENDATIONS, get_aqi_level

class AQIAlertSystem:
    def __init__(self):
        self.levels = AQI_LEVELS
        self.recommendations = AQI_HEALTH_RECOMMENDATIONS

    def get_alert(self, aqi_value: int) -> dict:
        level_info = get_aqi_level(aqi_value)
        label = level_info["label"]
        return {
            "label": label,
            "color": level_info["color"],
            "bg_color": level_info["bg"],
            "emoji": level_info["emoji"],
            "recommendation": self.recommendations.get(label, ""),
            "is_hazardous": aqi_value > 150
        }

    def get_forecast_alerts(self, forecast_df: pd.DataFrame) -> list[dict]:
        alerts = []
        if forecast_df is None or forecast_df.empty:
            return alerts
        
        # Ensure we have datetime and aqi columns
        if 'datetime' not in forecast_df.columns or 'aqi' not in forecast_df.columns:
            return alerts
            
        for _, row in forecast_df.iterrows():
            aqi = row['aqi']
            if aqi > 100:
                dt = row['datetime']
                alert_info = self.get_alert(aqi)
                alerts.append({
                    "datetime": dt.isoformat() if hasattr(dt, 'isoformat') else str(dt),
                    "predicted_aqi": float(aqi),
                    "level": alert_info["label"],
                    "recommendation": alert_info["recommendation"]
                })
        return alerts

    def get_alert_summary(self, current_aqi: int, forecast_df: pd.DataFrame = None) -> dict:
        current_alert = self.get_alert(current_aqi)
        
        trend = "stable"
        peak_forecast = {"aqi": current_aqi, "datetime": None}
        hazardous_hours = 0
        
        if forecast_df is not None and not forecast_df.empty and 'aqi' in forecast_df.columns:
            mean_aqi = forecast_df['aqi'].mean()
            if mean_aqi > current_aqi * 1.1:
                trend = "worsening"
            elif mean_aqi < current_aqi * 0.9:
                trend = "improving"
                
            max_idx = forecast_df['aqi'].idxmax()
            max_row = forecast_df.loc[max_idx]
            dt = max_row['datetime']
            peak_forecast = {
                "aqi": float(max_row['aqi']),
                "datetime": dt.isoformat() if hasattr(dt, 'isoformat') else str(dt)
            }
            
            hazardous_hours = int((forecast_df['aqi'] > 200).sum())
            
        action_items = [current_alert["recommendation"]]
        if hazardous_hours > 0:
            action_items.append("Prepare for hazardous air quality in the coming days.")
            
        return {
            "current_alert": current_alert,
            "trend": trend,
            "peak_forecast": peak_forecast,
            "hazardous_hours": hazardous_hours,
            "action_items": action_items
        }

def format_alert_html(alert: dict) -> str:
    bg_color = alert.get('bg_color', '#ffffff')
    border_color = alert.get('color', '#000000')
    emoji = alert.get('emoji', '⚠️')
    label = alert.get('label', 'Unknown')
    recommendation = alert.get('recommendation', '')
    
    return f"""
    <div style="background-color: {bg_color}; border-left: 5px solid {border_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <h4 style="margin-top: 0; margin-bottom: 5px; color: {border_color};">{emoji} {label}</h4>
        <p style="margin: 0;">{recommendation}</p>
    </div>
    """

def format_alert_banner(alert: dict) -> str:
    emoji = alert.get('emoji', '⚠️')
    label = alert.get('label', 'Unknown')
    return f"{emoji} <strong>AQI Level: {label}</strong> — {alert.get('recommendation', '')}"
