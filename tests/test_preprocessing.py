# test_preprocessing.py

import numpy as np
import pandas as pd
import torch
from utils.preprocessing.sliding_window import TimeSeriesPreprocessor

def test_preprocessor():
    # Generate sample data (simulated stock prices)
    np.random.seed(42)
    n_points = 1000
    time_points = np.linspace(0, 100, n_points)
    # Create a somewhat realistic looking price series with trend and noise
    prices = 100 + time_points + 10 * np.sin(time_points/5) + np.random.normal(0, 1, n_points)
    
    # Initialize preprocessor
    preprocessor = TimeSeriesPreprocessor(
        window_size=100,
        overlap=90,
        prediction_horizon=10
    )
    
    # Normalize data
    normalized_data = preprocessor.normalize_data(prices)
    
    # Create sequences
    X, y = preprocessor.create_sequences(normalized_data)
    
    print(f"Original data shape: {prices.shape}")
    print(f"Normalized data shape: {normalized_data.shape}")
    print(f"Input sequences shape: {X.shape}")
    print(f"Target sequences shape: {y.shape}")
    
    # Verify a few properties
    print("\nVerifying properties:")
    print(f"Data range after normalization: [{normalized_data.min():.3f}, {normalized_data.max():.3f}]")
    print(f"Number of sequences: {X.shape[0]}")
    print(f"Sequence length: {X.shape[1]}")
    print(f"Prediction horizon: {y.shape[1]}")
    
    # Verify GPU transfer if available
    if torch.cuda.is_available():
        print("\nDevice check:")
        print(f"X device: {X.device}")
        print(f"y device: {y.device}")

if __name__ == "__main__":
    test_preprocessor()
