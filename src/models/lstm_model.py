import logging
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import LSTM_CONFIG, SEQUENCE_LENGTH

logger = logging.getLogger(__name__)

class LSTMModel:
    """LSTM model for sequential AQI prediction."""
    
    def __init__(self, config=None):
        if config is None:
            config = LSTM_CONFIG
        self.config = config
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.seq_length = SEQUENCE_LENGTH
        
    def _build_model(self, input_shape):
        """Build and compile the LSTM model."""
        model = Sequential([
            LSTM(self.config.get('units_1', 64), return_sequences=True, input_shape=input_shape),
            Dropout(self.config.get('dropout', 0.2)),
            LSTM(self.config.get('units_2', 32)),
            Dropout(self.config.get('dropout', 0.2)),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=self.config.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss='mse')
        return model
        
    def prepare_sequences(self, X, y=None, seq_length=None):
        """Create sliding windows for time series data."""
        if seq_length is None:
            seq_length = self.seq_length
            
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:(i + seq_length)])
            if y is not None:
                y_seq.append(y[i + seq_length])
                
        if y is not None:
            return np.array(X_seq), np.array(y_seq)
        return np.array(X_seq)
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the LSTM model."""
        logger.info(f"Training {self.get_name()}...")
        
        # Scale data
        X_train_scaled = self.feature_scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        ).reshape(X_train.shape)
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        X_train_seq, y_train_seq = self.prepare_sequences(X_train_scaled, y_train_scaled)
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.feature_scaler.transform(
                X_val.reshape(-1, X_val.shape[-1])
            ).reshape(X_val.shape)
            y_val_scaled = self.target_scaler.transform(y_val.reshape(-1, 1)).flatten()
            X_val_seq, y_val_seq = self.prepare_sequences(X_val_scaled, y_val_scaled)
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None
            
        # Build model if not built
        if self.model is None:
            self.model = self._build_model((self.seq_length, X_train_seq.shape[2]))
            
        callbacks = [
            EarlyStopping(
                patience=self.config.get('patience', 15),
                restore_best_weights=True,
                monitor='val_loss' if validation_data else 'loss'
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        self.model.fit(
            X_train_seq, y_train_seq,
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 32),
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        logger.info(f"{self.get_name()} trained successfully.")
        
    def predict(self, X):
        """Make predictions using the trained model."""
        if len(X) <= self.seq_length:
            return np.array([])
            
        X_scaled = self.feature_scaler.transform(
            X.reshape(-1, X.shape[-1])
        ).reshape(X.shape)
        X_seq = self.prepare_sequences(X_scaled)
        
        preds_scaled = self.model.predict(X_seq, verbose=0)
        return self.inverse_transform_predictions(preds_scaled).flatten()
        
    def inverse_transform_predictions(self, predictions):
        """Inverse transform the scaled predictions."""
        return self.target_scaler.inverse_transform(predictions.reshape(-1, 1))
        
    def evaluate(self, X_test, y_test):
        """Evaluate the model using MAE, RMSE, MAPE, and R2."""
        if len(X_test) <= self.seq_length or len(y_test) <= self.seq_length:
            return {'mae': None, 'rmse': None, 'mape': None, 'r2': None}
            
        y_pred = self.predict(X_test)
        
        # Align y_test with predictions (drop the first seq_length elements)
        y_test_aligned = y_test[self.seq_length:]
        
        mae = mean_absolute_error(y_test_aligned, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_aligned, y_pred))
        r2 = r2_score(y_test_aligned, y_pred)
        
        y_test_safe = np.where(y_test_aligned == 0, 1, y_test_aligned)
        mape = np.mean(np.abs((y_test_aligned - y_pred) / y_test_safe)) * 100
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2)
        }
        logger.info(f"Evaluation metrics for {self.get_name()}: {metrics}")
        return metrics
        
    def save(self, path):
        """Save the trained model and scalers to disk."""
        if self.model:
            self.model.save(f"{path}.h5")
        joblib.dump({
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'config': self.config,
            'seq_length': self.seq_length
        }, f"{path}_scalers.pkl")
        logger.info(f"{self.get_name()} saved to {path}.h5 and {path}_scalers.pkl")
        
    @classmethod
    def load(cls, path):
        """Load a trained model and scalers from disk."""
        instance = cls()
        instance.model = load_model(f"{path}.h5")
        
        scalers_dict = joblib.load(f"{path}_scalers.pkl")
        instance.feature_scaler = scalers_dict['feature_scaler']
        instance.target_scaler = scalers_dict['target_scaler']
        instance.config = scalers_dict['config']
        instance.seq_length = scalers_dict['seq_length']
        
        logger.info(f"{instance.get_name()} loaded from {path}")
        return instance
        
    def get_name(self):
        """Return the model name."""
        return "LSTMModel"
