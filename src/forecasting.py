"""
Forecasting Module

This module implements time series forecasting models:
- ARIMA (AutoRegressive Integrated Moving Average)
- Prophet (Facebook's forecasting tool)
- LSTM (Long Short-Term Memory neural networks)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Prophet
try:
    from prophet import Prophet
except ImportError:
    Prophet = None
    print("Warning: Prophet not installed. Install with: pip install prophet")

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    tf = None
    print("Warning: TensorFlow not installed. Install with: pip install tensorflow")


class Forecaster:
    """
    Implements multiple time series forecasting methods.
    """
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.scalers = {}
    
    def check_stationarity(self, timeseries, column_name='value'):
        """
        Check if time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            timeseries (pd.Series or pd.DataFrame): Time series data
            column_name (str): Column name if DataFrame
        
        Returns:
            dict: Test results
        """
        if isinstance(timeseries, pd.DataFrame):
            ts = timeseries[column_name].values
        else:
            ts = timeseries.values
        
        # Remove any NaN values
        ts = ts[~np.isnan(ts)]
        
        # Perform ADF test
        result = adfuller(ts)
        
        test_result = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
        
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        print(f"Is Stationary: {test_result['is_stationary']}")
        
        return test_result
    
    def forecast_arima(self, timeseries, periods=10, order=(1, 1, 1), column='Production_Tonnes'):
        """
        Forecast using ARIMA model.
        
        Args:
            timeseries (pd.DataFrame): Time series data with date index
            periods (int): Number of periods to forecast
            order (tuple): ARIMA order (p, d, q)
            column (str): Column to forecast
        
        Returns:
            dict: Forecast results
        """
        print(f"\n{'='*60}")
        print(f"ARIMA Forecasting (order={order})")
        print(f"{'='*60}")
        
        # Prepare data
        ts = timeseries[[column]].copy()
        
        # Fit ARIMA model
        print(f"Fitting ARIMA{order} model...")
        model = ARIMA(ts[column], order=order)
        fitted_model = model.fit()
        
        print(f"✓ Model fitted successfully")
        print(f"\nModel Summary:")
        print(fitted_model.summary())
        
        # Make predictions
        forecast = fitted_model.forecast(steps=periods)
        
        # Create forecast dataframe
        last_year = timeseries.index[-1]
        if isinstance(last_year, int):
            forecast_index = range(last_year + 1, last_year + periods + 1)
        else:
            forecast_index = pd.date_range(start=last_year, periods=periods + 1, freq='Y')[1:]
        
        forecast_df = pd.DataFrame({
            'Forecast': forecast,
            'Model': 'ARIMA'
        }, index=forecast_index)
        
        # Store model
        self.models['arima'] = fitted_model
        self.forecasts['arima'] = forecast_df
        
        print(f"\n✓ Forecast for next {periods} periods:")
        print(forecast_df)
        
        return {
            'model': fitted_model,
            'forecast': forecast_df,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic
        }
    
    def forecast_prophet(self, timeseries, periods=10, yearly_seasonality=True):
        """
        Forecast using Facebook Prophet.
        
        Args:
            timeseries (pd.DataFrame): Time series with 'Year' and value columns
            periods (int): Number of periods to forecast
            yearly_seasonality (bool): Include yearly seasonality
        
        Returns:
            dict: Forecast results
        """
        if Prophet is None:
            print("Error: Prophet not installed")
            return None
        
        print(f"\n{'='*60}")
        print(f"Prophet Forecasting")
        print(f"{'='*60}")
        
        # Prepare data in Prophet format
        df = timeseries.reset_index()
        
        # Determine the value column (not 'Year')
        value_col = [col for col in df.columns if col not in ['Year', 'index']][0]
        
        # Create Prophet dataframe
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df['Year'], format='%Y'),
            'y': df[value_col]
        })
        
        # Initialize and fit model
        print("Fitting Prophet model...")
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        model.fit(prophet_df)
        print("✓ Model fitted successfully")
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=periods, freq='Y')
        
        # Predict
        forecast = model.predict(future)
        
        # Extract forecast period only
        forecast_only = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        forecast_only['ds'] = forecast_only['ds'].dt.year
        forecast_only = forecast_only.rename(columns={'ds': 'Year', 'yhat': 'Forecast'})
        forecast_only['Model'] = 'Prophet'
        
        # Store model
        self.models['prophet'] = model
        self.forecasts['prophet'] = forecast_only
        
        print(f"\n✓ Forecast for next {periods} periods:")
        print(forecast_only[['Year', 'Forecast', 'yhat_lower', 'yhat_upper']])
        
        return {
            'model': model,
            'forecast': forecast_only,
            'full_forecast': forecast
        }
    
    def prepare_lstm_data(self, timeseries, column='Production_Tonnes', lookback=5):
        """
        Prepare data for LSTM model.
        
        Args:
            timeseries (pd.DataFrame): Time series data
            column (str): Column to use
            lookback (int): Number of time steps to look back
        
        Returns:
            tuple: X, y, scaler
        """
        # Extract values
        data = timeseries[column].values.reshape(-1, 1)
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(data_scaled)):
            X.append(data_scaled[i-lookback:i, 0])
            y.append(data_scaled[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X for LSTM [samples, time steps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        return X, y, scaler
    
    def forecast_lstm(self, timeseries, periods=10, column='Production_Tonnes', 
                     lookback=5, epochs=50, batch_size=8):
        """
        Forecast using LSTM neural network.
        
        Args:
            timeseries (pd.DataFrame): Time series data
            periods (int): Number of periods to forecast
            column (str): Column to forecast
            lookback (int): Number of time steps to look back
            epochs (int): Training epochs
            batch_size (int): Batch size
        
        Returns:
            dict: Forecast results
        """
        if tf is None:
            print("Error: TensorFlow not installed")
            return None
        
        print(f"\n{'='*60}")
        print(f"LSTM Forecasting")
        print(f"{'='*60}")
        
        # Prepare data
        print(f"Preparing data with lookback={lookback}...")
        X, y, scaler = self.prepare_lstm_data(timeseries, column, lookback)
        self.scalers['lstm'] = scaler
        
        # Split into train/test
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Build LSTM model
        print("\nBuilding LSTM model...")
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print(model.summary())
        
        # Train model
        print(f"\nTraining for {epochs} epochs...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        print(f"✓ Training complete")
        print(f"  Final training loss: {history.history['loss'][-1]:.6f}")
        print(f"  Final validation loss: {history.history['val_loss'][-1]:.6f}")
        
        # Make predictions for future
        last_sequence = timeseries[column].values[-lookback:].reshape(-1, 1)
        last_sequence_scaled = scaler.transform(last_sequence)
        
        forecasts = []
        current_sequence = last_sequence_scaled.copy()
        
        print(f"\nGenerating {periods} period forecast...")
        for _ in range(periods):
            # Reshape for prediction
            current_input = current_sequence.reshape(1, lookback, 1)
            
            # Predict next value
            next_pred = model.predict(current_input, verbose=0)
            forecasts.append(next_pred[0, 0])
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], next_pred).reshape(-1, 1)
        
        # Inverse transform predictions
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts_original = scaler.inverse_transform(forecasts)
        
        # Create forecast dataframe
        last_year = timeseries.index[-1]
        if isinstance(last_year, int):
            forecast_index = range(last_year + 1, last_year + periods + 1)
        else:
            forecast_index = pd.date_range(start=last_year, periods=periods + 1, freq='Y')[1:]
        
        forecast_df = pd.DataFrame({
            'Forecast': forecasts_original.flatten(),
            'Model': 'LSTM'
        }, index=forecast_index)
        
        # Store model
        self.models['lstm'] = model
        self.forecasts['lstm'] = forecast_df
        
        print(f"\n✓ Forecast for next {periods} periods:")
        print(forecast_df)
        
        return {
            'model': model,
            'forecast': forecast_df,
            'history': history,
            'scaler': scaler
        }
    
    def compare_models(self):
        """
        Compare forecasts from all models.
        
        Returns:
            pd.DataFrame: Comparison table
        """
        if not self.forecasts:
            print("Error: No forecasts available. Run forecasting methods first.")
            return None
        
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        
        # Combine all forecasts
        comparison = pd.DataFrame()
        
        for model_name, forecast_df in self.forecasts.items():
            if 'Forecast' in forecast_df.columns:
                comparison[model_name] = forecast_df['Forecast'].values
        
        comparison.index = self.forecasts[list(self.forecasts.keys())[0]].index
        
        # Add statistics
        comparison['Mean'] = comparison.mean(axis=1)
        comparison['Std'] = comparison.std(axis=1)
        
        print(comparison)
        
        return comparison
    
    def calculate_forecast_metrics(self, actual, predicted):
        """
        Calculate forecast accuracy metrics.
        
        Args:
            actual (array-like): Actual values
            predicted (array-like): Predicted values
        
        Returns:
            dict: Accuracy metrics
        """
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(actual - predicted))
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'R2': r2
        }
        
        return metrics


if __name__ == "__main__":
    # Test forecasting
    from data_loader import load_data
    from preprocessing import DataPreprocessor
    
    # Load and preprocess data
    print("Loading data...")
    data = load_data()
    preprocessor = DataPreprocessor()
    processed = preprocessor.preprocess_all(data)
    
    # Get production data
    production = processed['production'].set_index('Year')
    
    # Initialize forecaster
    forecaster = Forecaster()
    
    # Check stationarity
    print("\nChecking stationarity...")
    forecaster.check_stationarity(production, 'Production_Tonnes')
    
    # ARIMA forecast
    arima_results = forecaster.forecast_arima(
        production, 
        periods=10, 
        order=(2, 1, 2),
        column='Production_Tonnes'
    )
    
    # Prophet forecast
    if Prophet is not None:
        prophet_results = forecaster.forecast_prophet(
            production.reset_index(),
            periods=10
        )
    
    # LSTM forecast
    if tf is not None:
        lstm_results = forecaster.forecast_lstm(
            production,
            periods=10,
            column='Production_Tonnes',
            lookback=5,
            epochs=100
        )
    
    # Compare models
    comparison = forecaster.compare_models()

