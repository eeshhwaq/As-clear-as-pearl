import os
import logging
import argparse
import pandas as pd
import hopsworks
from pathlib import Path
import joblib

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    HOPSWORKS_PROJECT_NAME,
    HOPSWORKS_API_KEY,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    SEQUENCE_LENGTH,
    TARGET
)
from src.feature_engineering import prepare_tabular_data, prepare_sequence_data
from src.models import LinearRegressionModel, XGBoostModel, LSTMModel, GRUModel
from src.explainability import explain_all_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_training_data(use_hopsworks=True):
    if use_hopsworks:
        try:
            logger.info("Connecting to Hopsworks...")
            project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
            fs = project.get_feature_store()
            fv = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
            logger.info("Fetching batch data from Hopsworks...")
            df = fv.get_batch_data()
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp').reset_index(drop=True)
            return df
        except Exception as e:
            logger.warning(f"Failed to load from Hopsworks: {e}. Falling back to local CSV.")
            
    file_path = DATA_DIR / "engineered_features.csv"
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def split_data(df, train_ratio=0.70, val_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

def train_all_models(train_df, val_df, test_df):
    results = {}
    
    # Tabular models
    X_train_tab, y_train_tab = prepare_tabular_data(train_df, target_col=TARGET)
    X_val_tab, y_val_tab = prepare_tabular_data(val_df, target_col=TARGET)
    X_test_tab, y_test_tab = prepare_tabular_data(test_df, target_col=TARGET)
    
    logger.info("Training LinearRegressionModel...")
    lr = LinearRegressionModel()
    lr.train(X_train_tab, y_train_tab, X_val_tab, y_val_tab)
    lr_metrics = lr.evaluate(X_test_tab, y_test_tab)
    lr_pred = lr.predict(X_test_tab)
    results['LinearRegression'] = {'model': lr, 'metrics': lr_metrics, 'predictions': lr_pred}
    
    logger.info("Training XGBoostModel...")
    xgb = XGBoostModel()
    xgb.train(X_train_tab, y_train_tab, X_val_tab, y_val_tab)
    xgb_metrics = xgb.evaluate(X_test_tab, y_test_tab)
    xgb_pred = xgb.predict(X_test_tab)
    results['XGBoost'] = {'model': xgb, 'metrics': xgb_metrics, 'predictions': xgb_pred}
    
    # Sequence models
    X_train_seq, y_train_seq = prepare_sequence_data(train_df, target_col=TARGET, seq_length=SEQUENCE_LENGTH)
    X_val_seq, y_val_seq = prepare_sequence_data(val_df, target_col=TARGET, seq_length=SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = prepare_sequence_data(test_df, target_col=TARGET, seq_length=SEQUENCE_LENGTH)
    
    logger.info("Training LSTMModel...")
    lstm = LSTMModel()
    lstm.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    lstm_metrics = lstm.evaluate(X_test_seq, y_test_seq)
    lstm_pred = lstm.predict(X_test_seq)
    results['LSTM'] = {'model': lstm, 'metrics': lstm_metrics, 'predictions': lstm_pred}
    
    logger.info("Training GRUModel...")
    gru = GRUModel()
    gru.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    gru_metrics = gru.evaluate(X_test_seq, y_test_seq)
    gru_pred = gru.predict(X_test_seq)
    results['GRU'] = {'model': gru, 'metrics': gru_metrics, 'predictions': gru_pred}
    
    return results

def compare_models(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, res in results.items():
        metrics = res['metrics']
        rows.append({
            'Model': model_name,
            'MAE': metrics.get('mae', 0),
            'RMSE': metrics.get('rmse', 0),
            'MAPE': metrics.get('mape', 0),
            'R2': metrics.get('r2', 0)
        })
    df = pd.DataFrame(rows)
    df = df.sort_values('RMSE', ascending=True)
    df.to_csv(DATA_DIR / 'model_comparison.csv', index=False)
    logger.info(f"Model comparison saved:\n{df}")
    return df

def save_models(results: dict, best_model_name: str):
    for model_name, res in results.items():
        model_obj = res['model']
        model_path = MODELS_DIR / f"{model_name}.pkl"
        joblib.dump(model_obj, model_path)
        logger.info(f"Saved {model_name} to {model_path}")
        
    try:
        project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
        mr = project.get_model_registry()
        best_res = results[best_model_name]
        
        # Hopsworks currently supports specific frameworks for models, using mr.python for general/sklearn
        if best_model_name in ['LinearRegression', 'XGBoost']:
            hw_model = mr.sklearn.create_model(
                name="lahore_aqi_best_model",
                metrics=best_res['metrics'],
                description=f"Best model: {best_model_name}"
            )
        else:
            hw_model = mr.tensorflow.create_model(
                name="lahore_aqi_best_model",
                metrics=best_res['metrics'],
                description=f"Best model: {best_model_name}"
            )
            
        hw_model.save(str(MODELS_DIR / f"{best_model_name}.pkl"))
        logger.info(f"Uploaded best model ({best_model_name}) to Hopsworks Model Registry.")
    except Exception as e:
        logger.warning(f"Failed to upload model to Hopsworks: {e}")

def run_training_pipeline(use_hopsworks=True):
    df = load_training_data(use_hopsworks=use_hopsworks)
    train_df, val_df, test_df = split_data(df)
    results = train_all_models(train_df, val_df, test_df)
    comp_df = compare_models(results)
    best_model_name = comp_df.iloc[0]['Model']
    save_models(results, best_model_name)
    
    logger.info("Running SHAP explainability...")
    explain_all_models(results, train_df, test_df)
    
    return results, comp_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-local", action="store_true", help="Force using local data instead of Hopsworks")
    args = parser.parse_args()
    
    run_training_pipeline(use_hopsworks=not args.use_local)
