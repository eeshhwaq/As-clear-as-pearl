import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from src.config import SHAP_DIR

logger = logging.getLogger(__name__)

def explain_model(model, X_train, X_test, model_type='tabular', feature_names=None):
    shap_values = None
    expected_value = None
    
    try:
        actual_model = getattr(model, 'model', model)
        
        if model_type == 'LinearRegression':
            explainer = shap.LinearExplainer(actual_model, X_train)
            shap_values = explainer.shap_values(X_test)
            expected_value = explainer.expected_value
        elif model_type == 'XGBoost':
            explainer = shap.TreeExplainer(actual_model)
            shap_values = explainer.shap_values(X_test)
            expected_value = explainer.expected_value
        elif model_type in ['LSTM', 'GRU']:
            background = X_train[:50]
            explainer = shap.GradientExplainer(actual_model, background)
            shap_values_raw = explainer.shap_values(X_test)
            if isinstance(shap_values_raw, list):
                shap_values_raw = shap_values_raw[0]
            shap_values = np.mean(np.abs(shap_values_raw), axis=1)
            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[0]
    except Exception as e:
        logger.warning(f"Failed to explain model {model_type}: {e}")
        
    return {
        'shap_values': shap_values,
        'expected_value': expected_value,
        'feature_names': feature_names
    }

def generate_shap_plots(shap_result, model_name, save_dir=SHAP_DIR):
    shap_values = shap_result.get('shap_values')
    feature_names = shap_result.get('feature_names')
    
    if shap_values is None:
        return
        
    plt.figure()
    try:
        shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False)
        plt.savefig(save_dir / f"{model_name}_summary_bar.png", bbox_inches='tight')
    except Exception as e:
        logger.warning(f"Failed to generate summary bar plot for {model_name}: {e}")
    plt.close()
    
    if model_name in ['LinearRegression', 'XGBoost']:
        plt.figure()
        try:
            shap.summary_plot(shap_values, feature_names=feature_names, show=False)
            plt.savefig(save_dir / f"{model_name}_beeswarm.png", bbox_inches='tight')
        except Exception as e:
            logger.warning(f"Failed to generate beeswarm plot for {model_name}: {e}")
        plt.close()
        
        try:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[-3:][::-1]
            for idx in top_indices:
                feat_name = feature_names[idx] if feature_names is not None else str(idx)
                plt.figure()
                shap.dependence_plot(idx, shap_values, None, feature_names=feature_names, show=False)
                plt.savefig(save_dir / f"{model_name}_dependence_{feat_name}.png", bbox_inches='tight')
                plt.close()
        except Exception as e:
            logger.warning(f"Failed to generate dependence plots for {model_name}: {e}")

def explain_all_models(results, train_df, test_df):
    from src.feature_engineering import prepare_tabular_data, prepare_sequence_data
    from src.config import TARGET, SEQUENCE_LENGTH
    
    shap_results = {}
    
    for model_name, res in results.items():
        logger.info(f"Explaining {model_name}...")
        model_obj = res['model']
        
        if model_name in ['LinearRegression', 'XGBoost']:
            X_train, _ = prepare_tabular_data(train_df, target_col=TARGET)
            X_test, _ = prepare_tabular_data(test_df, target_col=TARGET)
            feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None
            
            X_train_sample = X_train.sample(min(100, len(X_train))) if isinstance(X_train, pd.DataFrame) else X_train[:100]
            X_test_sample = X_test.sample(min(100, len(X_test))) if isinstance(X_test, pd.DataFrame) else X_test[:100]
            
            if isinstance(X_train_sample, pd.DataFrame):
                X_train_arr = X_train_sample.to_numpy()
                X_test_arr = X_test_sample.to_numpy()
            else:
                X_train_arr = X_train_sample
                X_test_arr = X_test_sample
                
            shap_res = explain_model(model_obj, X_train_arr, X_test_arr, model_type=model_name, feature_names=feature_names)
            
        elif model_name in ['LSTM', 'GRU']:
            X_train, _ = prepare_sequence_data(train_df, target_col=TARGET, seq_length=SEQUENCE_LENGTH)
            X_test, _ = prepare_sequence_data(test_df, target_col=TARGET, seq_length=SEQUENCE_LENGTH)
            
            # Subsample for speed
            X_train_sample = X_train[:100]
            X_test_sample = X_test[:100]
            feature_names = [f"Feature_{i}" for i in range(X_train_sample.shape[2])]
            
            shap_res = explain_model(model_obj, X_train_sample, X_test_sample, model_type=model_name, feature_names=feature_names)
            
        shap_results[model_name] = shap_res
        generate_shap_plots(shap_res, model_name)
        
    return shap_results

def get_feature_importance_summary(shap_results: dict) -> pd.DataFrame:
    importance_data = {}
    
    for model_name, res in shap_results.items():
        shap_vals = res.get('shap_values')
        feat_names = res.get('feature_names')
        if shap_vals is not None and feat_names is not None:
            if len(np.array(shap_vals).shape) > 1:
                mean_abs_shap = np.abs(shap_vals).mean(axis=0)
                importance_data[f"{model_name}_importance"] = pd.Series(mean_abs_shap, index=feat_names)
                
    if not importance_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(importance_data)
    df['avg_importance'] = df.mean(axis=1)
    df = df.sort_values('avg_importance', ascending=False)
    df = df.reset_index().rename(columns={'index': 'feature'})
    return df
