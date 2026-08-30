import logging
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import XGB_CONFIG

logger = logging.getLogger(__name__)

class XGBoostModel:
    """XGBoost model for AQI prediction."""
    
    def __init__(self, config=None):
        if config is None:
            config = XGB_CONFIG
        self.config = config
        self.model = XGBRegressor(**self.config)
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the XGBoost model with early stopping."""
        logger.info(f"Training {self.get_name()}...")
        
        if X_val is None or y_val is None:
            logger.info("No validation set provided. Using 20% of training data for validation.")
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.config.get('random_state', 42)
            )
            
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        logger.info(f"{self.get_name()} trained successfully.")
        
    def predict(self, X):
        """Make predictions using the trained model."""
        if len(X) == 0:
            return np.array([])
        return self.model.predict(X)
        
    def evaluate(self, X_test, y_test):
        """Evaluate the model using MAE, RMSE, MAPE, and R2."""
        if len(X_test) == 0 or len(y_test) == 0:
            return {'mae': None, 'rmse': None, 'mape': None, 'r2': None}
            
        y_pred = self.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        y_test_safe = np.where(y_test == 0, 1, y_test)
        mape = np.mean(np.abs((y_test - y_pred) / y_test_safe)) * 100
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2)
        }
        logger.info(f"Evaluation metrics for {self.get_name()}: {metrics}")
        return metrics
        
    def get_feature_importance(self):
        """Get feature importances from the trained model."""
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        feature_names = self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else [f"feature_{i}" for i in range(len(importances))]
        
        return dict(zip(feature_names, importances))
        
    def save(self, path):
        """Save the trained model to disk."""
        joblib.dump(self.model, path)
        logger.info(f"{self.get_name()} saved to {path}")
        
    @classmethod
    def load(cls, path):
        """Load a trained model from disk."""
        instance = cls()
        instance.model = joblib.load(path)
        logger.info(f"{instance.get_name()} loaded from {path}")
        return instance
        
    def get_name(self):
        """Return the model name."""
        return "XGBoostModel"
