# tests/test_arima_model.py

import numpy as np
import pandas as pd
from models.arima.model import StockARIMA

def test_arima_model():
    print("Testing ARIMA Model")
    print("-" * 50)
    
    # Generate sample data
    np.random.seed(42)
    n_points = 200
    data = pd.Series(np.cumsum(np.random.randn(n_points)) + 100)
    
    # Create model
    model = StockARIMA(order=(1,1,1))
    
    # Test fitting
    print("\nTesting model fitting...")
    model.fit(data)
    
    # Test single prediction
    print("\nTesting single prediction...")
    pred = model.predict(n_steps=10)
    print(f"Prediction shape: {pred.shape}")
    
    # Test sequence prediction
    print("\nTesting sequence prediction...")
    X, predictions = model.predict_sequences(
        data,
        window_size=100,
        prediction_horizon=10
    )
    
    print(f"Input sequence shape: {X.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    # Test with different parameters
    print("\nTesting different ARIMA parameters...")
    parameters = [(1,1,1), (2,1,2), (1,1,0)]
    for order in parameters:
        model = StockARIMA(order=order)
        model.fit(data)
        pred = model.predict(n_steps=10)
        print(f"Order {order} - Prediction shape: {pred.shape}")

if __name__ == "__main__":
    test_arima_model()
