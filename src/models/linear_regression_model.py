import logging
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import LR_CONFIG

logger = logging.getLogger(__name__)

class LinearRegressionModel:
    """Linear Regression model using Ridge regression with a StandardScaler pipeline."""
    
    def __init__(self, config=None):
        if config is None:
            config = LR_CONFIG
        self.config = config
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=self.config.get('alpha', 1.0)))
        ])
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the Ridge regression model."""
        logger.info(f"Training {self.get_name()}...")
        self.model.fit(X_train, y_train)
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
        
        # Handle zero values in y_test for MAPE
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
        
    def save(self, path):
        """Save the trained model pipeline to disk."""
        joblib.dump(self.model, path)
        logger.info(f"{self.get_name()} saved to {path}")
        
    @classmethod
    def load(cls, path):
        """Load a trained model pipeline from disk."""
        instance = cls()
        instance.model = joblib.load(path)
        logger.info(f"{instance.get_name()} loaded from {path}")
        return instance
        
    def get_name(self):
        """Return the model name."""
        return "LinearRegressionModel"
