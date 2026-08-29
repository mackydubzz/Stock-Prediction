# models/arima/model.py

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple, Union
import warnings
warnings.filterwarnings('ignore')  # Suppress statsmodels warnings

class StockARIMA:
    def __init__(
        self,
        order: Tuple[int, int, int] = (2, 1, 0),  # Changed default parameters
        seasonal_order: Tuple[int, int, int, int] = None
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.scaler_dict = None
    
    def fit(self, data: Union[np.ndarray, pd.Series], scaler_dict: dict = None) -> None:
        """Fit ARIMA model to the training data"""
        self.scaler_dict = scaler_dict
        
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        
        # Take first difference to ensure stationarity
        # (reset index: statsmodels requires a supported index starting at 0)
        diff_data = data.diff().dropna().reset_index(drop=True)
        
        # Fit ARIMA model
        self.model = ARIMA(
            diff_data,
            order=self.order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit()
        
        print(f"ARIMA model fitted with parameters: {self.order}")
    
    def predict(self, n_steps: int = 10) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Make forecast on differenced data
        forecast = self.model.forecast(steps=n_steps)
        
        # Integrate (cumsum) to get back to original scale
        return forecast.cumsum()
    
    def predict_sequences(
        self,
        data: Union[np.ndarray, pd.Series],
        window_size: int = 100,
        prediction_horizon: int = 10,
        stride: int = 10  # Added stride parameter to reduce computations
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make sequence predictions similar to deep learning models"""
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        
        # Calculate number of sequences with stride
        n_sequences = (len(data) - window_size - prediction_horizon) // stride + 1
        predictions = np.zeros((n_sequences, prediction_horizon))
        
        for i in range(n_sequences):
            start_idx = i * stride
            train_window = data[start_idx:start_idx+window_size]
            
            try:
                # Fit model on window
                diff_window = train_window.diff().dropna().reset_index(drop=True)
                window_model = ARIMA(
                    diff_window,
                    order=self.order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit()
                
                # Make prediction and integrate
                pred = window_model.forecast(steps=prediction_horizon)
                predictions[i] = pred.cumsum()
                
            except Exception as e:
                print(f"Warning: Prediction failed for sequence {i}: {str(e)}")
                predictions[i] = np.nan
        
        # Remove sequences with failed predictions
        valid_mask = ~np.isnan(predictions).any(axis=1)
        predictions = predictions[valid_mask]
        
        # Prepare input sequences
        X = np.array([data[i*stride:i*stride+window_size] 
                     for i in range(n_sequences)])[valid_mask]
        
        print(f"Successfully predicted {len(predictions)} out of {n_sequences} sequences")
        return X, predictions
