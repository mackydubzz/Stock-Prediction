# utils/preprocessing/sliding_window.py

import numpy as np
import pandas as pd
from typing import Tuple, Union
import torch

class TimeSeriesPreprocessor:
    def __init__(
        self,
        window_size: int = 100,
        overlap: int = 90,
        prediction_horizon: int = 10,
        normalization: str = 'minmax'
    ):
        """
        Initialize the time series preprocessor.
        
        Args:
            window_size (int): Size of the sliding window (default: 100 minutes)
            overlap (int): Number of overlapping points between windows (default: 90 minutes)
            prediction_horizon (int): Number of future points to predict (default: 10 minutes)
            normalization (str): Type of normalization ('minmax' or 'standard')
        """
        self.window_size = window_size
        self.overlap = overlap
        self.prediction_horizon = prediction_horizon
        self.stride = window_size - overlap
        self.normalization = normalization
        self.scaler = None
        
    def create_sequences(
        self,
        data: Union[np.ndarray, pd.Series],
        return_tensors: bool = True
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Create sequences using sliding window approach.
        
        Args:
            data: Input time series data
            return_tensors: If True, return PyTorch tensors; if False, return numpy arrays
            
        Returns:
            X: Sequence input windows
            y: Target values
        """
        if isinstance(data, pd.Series):
            data = data.values
            
        # Calculate number of sequences
        n_samples = len(data)
        n_sequences = ((n_samples - self.window_size - self.prediction_horizon) // 
                      self.stride) + 1
        
        # Initialize arrays for sequences and targets
        X = np.zeros((n_sequences, self.window_size))
        y = np.zeros((n_sequences, self.prediction_horizon))
        
        # Create sequences
        for i in range(n_sequences):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            target_end_idx = end_idx + self.prediction_horizon
            
            X[i] = data[start_idx:end_idx]
            y[i] = data[end_idx:target_end_idx]
        
        if return_tensors:
            X = torch.FloatTensor(X)
            y = torch.FloatTensor(y)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
        
        return X, y
    
    def normalize_data(self, data: Union[np.ndarray, pd.Series], fit: bool = True) -> np.ndarray:
        """
        Normalize the data using specified method.
        
        Args:
            data: Input data
            fit: If True, fit the scaler; if False, use existing scaler
            
        Returns:
            Normalized data
        """
        if isinstance(data, pd.Series):
            data = data.values.reshape(-1, 1)
        elif isinstance(data, np.ndarray):
            data = data.reshape(-1, 1)
            
        if self.normalization == 'minmax':
            if fit:
                self.scaler = {'min': data.min(), 'max': data.max()}
            normalized_data = (data - self.scaler['min']) / (self.scaler['max'] - self.scaler['min'])
            
        elif self.normalization == 'standard':
            if fit:
                self.scaler = {'mean': data.mean(), 'std': data.std()}
            normalized_data = (data - self.scaler['mean']) / self.scaler['std']
            
        return normalized_data.reshape(-1)
    
    def inverse_normalize(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Inverse transform normalized data.
        
        Args:
            data: Normalized data
            
        Returns:
            Original scale data
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
            
        data = data.reshape(-1, 1)
        
        if self.normalization == 'minmax':
            original_data = data * (self.scaler['max'] - self.scaler['min']) + self.scaler['min']
        elif self.normalization == 'standard':
            original_data = data * self.scaler['std'] + self.scaler['mean']
            
        return original_data.reshape(-1)
