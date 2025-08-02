import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings('ignore')

# Handle optional dependencies gracefully
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some models will be disabled.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. XGBoost model will be disabled.")

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. ARIMA model will be disabled.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM model will be disabled.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except (ImportError, AttributeError) as e:
    PROPHET_AVAILABLE = False
    print(f"Warning: Prophet not available. Prophet model will be disabled. Error: {e}")


class BasePredictor:
    """Base class for all prediction models"""
    
    def __init__(self, name, model_type):
        self.name = name
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = {}
        self.metrics = {}
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        features = df.copy()
        
        # Time-based features
        features['year'] = pd.to_datetime(features['date']).dt.year
        features['month'] = pd.to_datetime(features['date']).dt.month
        features['quarter'] = pd.to_datetime(features['date']).dt.quarter
        features['day_of_year'] = pd.to_datetime(features['date']).dt.dayofyear
        features['is_weekend'] = pd.to_datetime(features['date']).dt.weekday >= 5
        
        # Lag features
        for lag in [1, 2, 3, 7, 14]:
            features[f'crs_lag_{lag}'] = features['lowest_crs_score'].shift(lag)
            features[f'invitations_lag_{lag}'] = features['invitations_issued'].shift(lag)
        
        # Rolling statistics
        for window in [3, 7, 14]:
            features[f'crs_rolling_mean_{window}'] = features['lowest_crs_score'].rolling(window).mean()
            features[f'crs_rolling_std_{window}'] = features['lowest_crs_score'].rolling(window).std()
            features[f'invitations_rolling_mean_{window}'] = features['invitations_issued'].rolling(window).mean()
        
        # Days since last draw
        features['days_since_last'] = features['days_since_last_draw'].fillna(14)  # Default 2 weeks
        
        # Category encoding
        category_dummies = pd.get_dummies(features['category'], prefix='category')
        features = pd.concat([features, category_dummies], axis=1)
        
        return features
    
    def split_data(self, df, test_size=0.2):
        """Split data into train/test sets"""
        split_index = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]
        return train_df, test_df
    
    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        if not SKLEARN_AVAILABLE:
            # Simple metrics calculation
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean((y_true - y_pred) ** 2)
            
            # Simple R² calculation
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2
            }
        else:
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2
            }


class ARIMAPredictor(BasePredictor):
    """ARIMA time series model for CRS score prediction"""
    
    def __init__(self, order=(2, 1, 2)):
        super().__init__("ARIMA Time Series", "ARIMA")
        self.order = order
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA model")
    
    def train(self, df):
        """Train ARIMA model"""
        # Sort by date and get CRS scores
        df_sorted = df.sort_values('date')
        crs_scores = df_sorted['lowest_crs_score'].values
        
        # Fit ARIMA model
        self.model = ARIMA(crs_scores, order=self.order)
        self.fitted_model = self.model.fit()
        self.is_trained = True
        
        # Calculate metrics on training data
        predictions = self.fitted_model.fittedvalues
        self.metrics = self.evaluate(crs_scores[1:], predictions[1:])  # Skip first value
        
        return self.metrics
    
    def predict(self, steps=1):
        """Predict next CRS scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast.tolist()


class RandomForestPredictor(BasePredictor):
    """Random Forest model for CRS score and date prediction"""
    
    def __init__(self, n_estimators=100, random_state=42):
        super().__init__("Random Forest", "RF")
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Random Forest model")
    
    def train(self, df, target_col='lowest_crs_score'):
        """Train Random Forest model"""
        features = self.prepare_features(df)
        
        # Define feature columns (exclude target and non-feature columns)
        exclude_cols = ['date', 'lowest_crs_score', 'round_number', 'url', 'category']
        feature_cols = [col for col in features.columns if col not in exclude_cols]
        
        X = features[feature_cols].fillna(0)
        y = features[target_col]
        
        # Remove rows with NaN in target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, 
            random_state=self.random_state
        )
        self.model.fit(X, y)
        
        # Calculate feature importance
        self.feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
        
        # Calculate metrics
        predictions = self.model.predict(X)
        self.metrics = self.evaluate(y, predictions)
        self.is_trained = True
        
        return self.metrics
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)


class XGBoostPredictor(BasePredictor):
    """XGBoost model for CRS score prediction"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, random_state=42):
        super().__init__("XGBoost", "XGB")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required for XGBoost model")
    
    def train(self, df, target_col='lowest_crs_score'):
        """Train XGBoost model"""
        features = self.prepare_features(df)
        
        # Define feature columns
        exclude_cols = ['date', 'lowest_crs_score', 'round_number', 'url', 'category']
        feature_cols = [col for col in features.columns if col not in exclude_cols]
        
        X = features[feature_cols].fillna(0)
        y = features[target_col]
        
        # Remove rows with NaN in target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Train model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )
        self.model.fit(X, y)
        
        # Calculate feature importance
        self.feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
        
        # Calculate metrics
        predictions = self.model.predict(X)
        self.metrics = self.evaluate(y, predictions)
        self.is_trained = True
        
        return self.metrics
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)


class LSTMPredictor(BasePredictor):
    """LSTM neural network for time series prediction"""
    
    def __init__(self, sequence_length=10, epochs=50):
        super().__init__("LSTM Neural Network", "LSTM")
        self.sequence_length = sequence_length
        self.epochs = epochs
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
    
    def create_sequences(self, data):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def train(self, df):
        """Train LSTM model"""
        # Sort by date and get CRS scores
        df_sorted = df.sort_values('date')
        crs_scores = df_sorted['lowest_crs_score'].values.reshape(-1, 1)
        
        # Normalize data
        crs_scores_scaled = self.scaler.fit_transform(crs_scores)
        
        # Create sequences
        X, y = self.create_sequences(crs_scores_scaled.flatten())
        
        # Reshape for LSTM (samples, time steps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build LSTM model
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        self.model.fit(X, y, epochs=self.epochs, batch_size=1, verbose=0)
        
        # Calculate metrics
        predictions = self.model.predict(X)
        predictions_rescaled = self.scaler.inverse_transform(predictions)
        y_rescaled = self.scaler.inverse_transform(y.reshape(-1, 1))
        
        self.metrics = self.evaluate(y_rescaled.flatten(), predictions_rescaled.flatten())
        self.is_trained = True
        
        return self.metrics
    
    def predict(self, last_sequence, steps=1):
        """Predict next CRS scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Reshape for prediction
            X = current_sequence.reshape((1, self.sequence_length, 1))
            pred = self.model.predict(X, verbose=0)
            
            # Add prediction to list
            predictions.append(pred[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[1:], pred[0, 0])
        
        # Inverse transform predictions
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_rescaled = self.scaler.inverse_transform(predictions_array)
        
        return predictions_rescaled.flatten().tolist()


class LinearRegressionPredictor(BasePredictor):
    """Simple Linear Regression model"""
    
    def __init__(self):
        super().__init__("Linear Regression", "LR")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Linear Regression model")
    
    def train(self, df, target_col='lowest_crs_score'):
        """Train Linear Regression model"""
        features = self.prepare_features(df)
        
        # Define feature columns
        exclude_cols = ['date', 'lowest_crs_score', 'round_number', 'url', 'category']
        feature_cols = [col for col in features.columns if col not in exclude_cols]
        
        X = features[feature_cols].fillna(0)
        y = features[target_col]
        
        # Remove rows with NaN in target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(X_scaled, y)
        
        # Calculate feature importance (coefficient magnitudes)
        self.feature_importance = dict(zip(feature_cols, np.abs(self.model.coef_)))
        
        # Calculate metrics
        predictions = self.model.predict(X_scaled)
        self.metrics = self.evaluate(y, predictions)
        self.is_trained = True
        
        return self.metrics
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class ProphetPredictor(BasePredictor):
    """Prophet time series model for EE draw prediction"""
    
    def __init__(self, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False):
        super().__init__("Prophet Time Series", "PROPHET")
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required for Prophet model")
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = None
    
    def train(self, df, target_col='lowest_crs_score', date_col='date'):
        """Train Prophet model
        
        Args:
            df: DataFrame with date and target columns
            target_col: Name of the target column
            date_col: Name of the date column
        """
        # Prepare data in Prophet format
        prophet_data = pd.DataFrame({
            'ds': pd.to_datetime(df[date_col]),
            'y': df[target_col]
        })
        
        # Remove rows with NaN values
        prophet_data = prophet_data.dropna()
        
        # Initialize Prophet model
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=0.05  # Sensitivity to trend changes
        )
        
        # Add custom seasonality for bi-weekly Express Entry draws
        self.model.add_seasonality(
            name='biweekly',
            period=14,  # 14 days for bi-weekly pattern
            fourier_order=3
        )
        
        # Fit the model
        self.model.fit(prophet_data)
        self.is_trained = True
        
        # Calculate metrics on fitted values
        forecast = self.model.predict(prophet_data[['ds']])
        self.metrics = self.evaluate(prophet_data['y'], forecast['yhat'])
        
        return self.metrics
    
    def predict(self, periods=30, freq='D'):
        """Make predictions
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency of predictions ('D' for daily, 'W' for weekly)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        
        # Make predictions
        forecast = self.model.predict(future)
        
        # Return predictions for the forecasted periods
        return forecast['yhat'].tail(periods).values.tolist()


class NeuralNetworkPredictor(BasePredictor):
    """Multi-Layer Perceptron Neural Network for EE prediction"""
    
    def __init__(self, hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000):
        super().__init__("Neural Network (MLP)", "MLP")
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Neural Network model")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = None
        self.scaler = None
    
    def train(self, df, target_col='lowest_crs_score'):
        """Train Neural Network model"""
        features = self.prepare_features(df)
        
        # Define feature columns (exclude target and non-feature columns)
        exclude_cols = ['date', 'lowest_crs_score', 'round_number', 'url', 'category']
        feature_cols = [col for col in features.columns if col not in exclude_cols]
        
        X = features[feature_cols].fillna(0)
        y = features[target_col]
        
        # Remove rows with NaN in target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Scale features for neural network
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train neural network
        from sklearn.neural_network import MLPRegressor
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            random_state=self.random_state,
            max_iter=self.max_iter,
            activation='relu',
            solver='adam',
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        self.model.fit(X_scaled, y)
        
        # Calculate feature importance (average absolute weights from first layer)
        if hasattr(self.model, 'coefs_'):
            first_layer_weights = np.abs(self.model.coefs_[0])
            self.feature_importance = dict(zip(
                feature_cols, 
                np.mean(first_layer_weights, axis=1)
            ))
        
        # Calculate metrics
        predictions = self.model.predict(X_scaled)
        self.metrics = self.evaluate(y, predictions)
        self.is_trained = True
        
        return self.metrics
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class EnsemblePredictor(BasePredictor):
    """Ensemble of multiple models"""
    
    def __init__(self, models=None):
        super().__init__("Ensemble Model", "ENSEMBLE")
        self.models = models or []
        self.weights = None
    
    def add_model(self, model):
        """Add a model to the ensemble"""
        self.models.append(model)
    
    def train(self, df, target_col='lowest_crs_score'):
        """Train all models in ensemble"""
        model_metrics = []
        
        for model in self.models:
            try:
                metrics = model.train(df, target_col)
                model_metrics.append(metrics['r2'])  # Use R² for weighting
            except Exception as e:
                print(f"Error training {model.name}: {e}")
                model_metrics.append(0)
        
        # Calculate weights based on R² scores
        total_r2 = sum(max(0, r2) for r2 in model_metrics)
        if total_r2 > 0:
            self.weights = [max(0, r2) / total_r2 for r2 in model_metrics]
        else:
            self.weights = [1/len(self.models)] * len(self.models)
        
        self.is_trained = True
        
        # Calculate ensemble metrics (simplified)
        self.metrics = {
            'mae': np.mean([model.metrics.get('mae', 0) for model in self.models]),
            'mse': np.mean([model.metrics.get('mse', 0) for model in self.models]),
            'r2': np.mean([model.metrics.get('r2', 0) for model in self.models])
        }
        
        return self.metrics
    
    def predict(self, X=None, steps=1):
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        predictions = []
        for i, model in enumerate(self.models):
            try:
                if hasattr(model, 'predict') and self.weights[i] > 0:
                    if isinstance(model, ARIMAPredictor):
                        pred = model.predict(steps)
                    elif isinstance(model, LSTMPredictor) and X is not None:
                        # For LSTM, we need the last sequence
                        last_sequence = X[-model.sequence_length:]
                        pred = model.predict(last_sequence, steps)
                    else:
                        pred = model.predict(X)
                    
                    # Handle different prediction formats
                    if isinstance(pred, (list, np.ndarray)):
                        if len(pred) > 0:
                            predictions.append((pred[0] if isinstance(pred, (list, np.ndarray)) else pred, self.weights[i]))
                    else:
                        predictions.append((pred, self.weights[i]))
            except Exception as e:
                print(f"Error predicting with {model.name}: {e}")
                continue
        
        if not predictions:
            return [450]  # Default prediction
        
        # Weighted average
        weighted_sum = sum(pred * weight for pred, weight in predictions)
        total_weight = sum(weight for _, weight in predictions)
        
        if total_weight > 0:
            return [weighted_sum / total_weight]
        else:
            return [450]  # Default prediction 