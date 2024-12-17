# tests/test_evaluator.py

import torch
import matplotlib
matplotlib.use('Agg')
import numpy as np
from utils.evaluation.evaluator import ModelEvaluator
from models.rnn.model import StockRNN
from models.lstm.model import StockLSTM
from models.cnn.model import StockCNN

def generate_synthetic_data(n_samples=10, seq_length=100, pred_length=10):
    """Generate synthetic stock price data"""
    # Generate time steps
    t = np.linspace(0, 1, seq_length + pred_length)
    
    # Base price and trend
    base_price = 100
    trend = t * 20  # Upward trend
    
    # Create samples
    X = np.zeros((n_samples, seq_length, 1))
    y = np.zeros((n_samples, pred_length, 1))
    
    for i in range(n_samples):
        # Add seasonal component and noise
        seasonal = 5 * np.sin(2 * np.pi * (t + np.random.rand()))
        noise = np.random.normal(0, 1, len(t))
        
        # Combine components
        series = base_price + trend + seasonal + noise
        
        # Split into input and target
        X[i, :, 0] = series[:seq_length]
        y[i, :, 0] = series[seq_length:]
    
    return torch.FloatTensor(X), torch.FloatTensor(y)

def test_evaluator():
    print("Testing Model Evaluator")
    print("-" * 50)
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Dictionary to store models
    models = {}
    
    # Try loading each model
    for model_type, model_class in [
        ('rnn', StockRNN),
        ('lstm', StockLSTM),
        ('cnn', StockCNN)
    ]:
        try:
            print(f"\nTesting {model_type.upper()} evaluation...")
            models[model_type] = evaluator.load_model(model_type, model_class)
            print(f"{model_type.upper()} model loaded successfully")
        except Exception as e:
            print(f"Error loading {model_type.upper()}: {str(e)}")
            print(f"Creating new {model_type.upper()} instance for testing...")
            models[model_type] = model_class().cuda() if torch.cuda.is_available() else model_class()
    
    # Generate synthetic data
    print("\nGenerating synthetic test data...")
    X_test, y_test = generate_synthetic_data()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        X_test = X_test.cuda()
        y_test = y_test.cuda()
    
    test_data = {
        'INFY': {
            'X': X_test,
            'y': y_test
        }
    }
    
    # Test evaluation
    print("\nEvaluating models...")
    for model_type, model in models.items():
        try:
            results = evaluator.evaluate_model(model_type, model, test_data)
        except Exception as e:
            print(f"Error evaluating {model_type.upper()}: {str(e)}")
    
    # Generate plots and save results
    try:
        print("\nGenerating plots and saving results...")
        evaluator.plot_predictions('INFY')
        evaluator.save_results()
        summary = evaluator.create_summary()
        print("Evaluation complete!")
    except Exception as e:
        print(f"Error in final steps: {str(e)}")

if __name__ == "__main__":
    test_evaluator()
