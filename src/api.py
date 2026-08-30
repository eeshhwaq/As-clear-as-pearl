from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import pandas as pd
import json
import os
import base64

from src.config import API_HOST, API_PORT, MODELS_DIR, SHAP_DIR
from src.alerts import AQIAlertSystem

app = FastAPI(
    title='As Clear as Pearl API',
    description='AQI Prediction API for Lahore',
    version='1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AppState:
    def __init__(self):
        self.models_loaded = False
        self.pipeline = None
        self.last_prediction = None
        self.last_prediction_time = None
        self.alert_system = AQIAlertSystem()

state = AppState()

@app.on_event("startup")
async def startup_event():
    # Will lazy-load models in the endpoints when needed
    pass

def get_pipeline():
    if not state.models_loaded:
        try:
            from src.inference_pipeline import InferencePipeline
            state.pipeline = InferencePipeline()
            state.models_loaded = True
        except Exception as e:
            print(f"Failed to load InferencePipeline: {e}")
            raise HTTPException(status_code=500, detail="Model inference pipeline could not be loaded")
    return state.pipeline

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/current-aqi")
def get_current_aqi():
    try:
        pipeline = get_pipeline()
        
        # Check cache
        now = datetime.now()
        if state.last_prediction and state.last_prediction_time and (now - state.last_prediction_time) < timedelta(hours=1):
            return state.last_prediction
            
        current_data = pipeline.get_current_data()
        if not current_data:
            raise HTTPException(status_code=404, detail="Current AQI data not available")
            
        aqi = current_data.get('aqi', 0)
        alert_info = state.alert_system.get_alert(aqi)
        
        response = {
            "aqi": aqi,
            "level": alert_info["label"],
            "color": alert_info["color"],
            "emoji": alert_info["emoji"],
            "dominant_pollutant": current_data.get("dominant_pollutant", "pm25"),
            "pollutants": current_data.get("pollutants", {}),
            "weather": current_data.get("weather", {}),
            "timestamp": datetime.now().isoformat(),
            "alert": {
                "recommendation": alert_info["recommendation"],
                "is_hazardous": alert_info["is_hazardous"]
            }
        }
        
        state.last_prediction = response
        state.last_prediction_time = now
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/forecast")
def get_forecast(model: str = Query("best", description="Model to use: best, linear_regression, xgboost, lstm, gru, all")):
    try:
        pipeline = get_pipeline()
        forecasts = {}
        
        if model == "all":
            models_to_run = ['linear_regression', 'xgboost', 'lstm', 'gru']
        else:
            models_to_run = [model]
            
        for m in models_to_run:
            try:
                forecast_df = pipeline.run_forecast(model_name=m)
                
                # Format forecast output
                forecast_list = []
                for _, row in forecast_df.iterrows():
                    dt = row['datetime']
                    aqi = row['aqi']
                    alert = state.alert_system.get_alert(aqi)
                    forecast_list.append({
                        "datetime": dt.isoformat() if hasattr(dt, 'isoformat') else str(dt),
                        "aqi": float(aqi),
                        "level": alert["label"],
                        "color": alert["color"]
                    })
                    
                alerts = state.alert_system.get_forecast_alerts(forecast_df)
                forecasts[m] = {
                    "model_name": m,
                    "forecast": forecast_list,
                    "alerts": alerts
                }
            except Exception as e:
                print(f"Error forecasting with model {m}: {e}")
                
        if not forecasts:
            raise HTTPException(status_code=500, detail="Could not generate forecast from any model")
            
        if model != "all" and model in forecasts:
            return forecasts[model]
            
        return {"models": forecasts}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/historical")
def get_historical(days: int = Query(30, description="Number of days to fetch")):
    try:
        pipeline = get_pipeline()
        hist_df = pipeline.get_historical_data(days=days)
        
        if hist_df is None or hist_df.empty:
            return {"data": []}
            
        data = []
        for _, row in hist_df.iterrows():
            row_dict = row.to_dict()
            if 'datetime' in row_dict and hasattr(row_dict['datetime'], 'isoformat'):
                row_dict['datetime'] = row_dict['datetime'].isoformat()
            data.append(row_dict)
            
        return {"data": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model-comparison")
def get_model_comparison():
    try:
        comp_file = MODELS_DIR / "model_comparison.csv"
        if not comp_file.exists():
            return {"models": []}
            
        df = pd.read_csv(comp_file)
        return {"models": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/shap/summary")
def get_shap_summary(model: str = Query("xgboost", description="Model name")):
    try:
        model_shap_dir = SHAP_DIR / model
        if not model_shap_dir.exists():
            raise HTTPException(status_code=404, detail=f"No SHAP data found for model {model}")
            
        response = {"model_name": model, "feature_importance": [], "plots": {}}
        
        # Load feature importance JSON if exists
        importance_file = model_shap_dir / "feature_importance.json"
        if importance_file.exists():
            with open(importance_file, 'r') as f:
                response["feature_importance"] = json.load(f)
                
        # Load plots
        summary_bar = model_shap_dir / "summary_bar.png"
        if summary_bar.exists():
            with open(summary_bar, "rb") as f:
                response["plots"]["summary_bar"] = base64.b64encode(f.read()).decode("utf-8")
                
        beeswarm = model_shap_dir / "beeswarm.png"
        if beeswarm.exists():
            with open(beeswarm, "rb") as f:
                response["plots"]["beeswarm"] = base64.b64encode(f.read()).decode("utf-8")
                
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alerts")
def get_alerts():
    try:
        pipeline = get_pipeline()
        
        # Get current AQI
        current_data = pipeline.get_current_data()
        current_aqi = current_data.get('aqi', 0) if current_data else 0
        
        # Get forecast to analyze trend
        try:
            forecast_df = pipeline.run_forecast(model_name="best")
        except:
            forecast_df = None
            
        summary = state.alert_system.get_alert_summary(current_aqi, forecast_df)
        forecast_alerts = state.alert_system.get_forecast_alerts(forecast_df)
        
        return {
            "current_alert": summary.get("current_alert", {}),
            "forecast_alerts": forecast_alerts,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('src.api:app', host=API_HOST, port=API_PORT, reload=True)
