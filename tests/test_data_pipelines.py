# tests/test_full_pipeline.py

from utils.preprocessing.data_pipeline import StockDataPipeline
import torch
from datetime import datetime

def test_full_pipeline():
    print("Testing Complete Stock Data Pipeline")
    print("-" * 50)
    
    # Initialize pipeline
    pipeline = StockDataPipeline()
    
    print("\nTime Periods:")
    print(f"Training: {pipeline.train_period['start']} to {pipeline.train_period['end']}")
    print(f"Testing:  {pipeline.test_period['start']} to {pipeline.test_period['end']}")
    
    print("\nFetching Data...")
    datasets = pipeline.prepare_train_test_data()
    
    print("\nTraining Data (Infosys):")
    print("-" * 30)
    train_data = datasets.get('train', {})
    if train_data and len(train_data.get('X', [])) > 0:
        print(f"Input shape: {train_data['X'].shape}")
        print(f"Target shape: {train_data['y'].shape}")
        print(f"Data range: [{train_data['X'].min():.2f}, {train_data['X'].max():.2f}]")
    else:
        print("No training data available")
    
    print("\nTesting Data:")
    print("-" * 30)
    test_data = datasets.get('test', {})
    for symbol in pipeline.paper_stocks:
        print(f"\n{symbol}:")
        if symbol in test_data and len(test_data[symbol]['X']) > 0:
            print(f"Input shape: {test_data[symbol]['X'].shape}")
            print(f"Target shape: {test_data[symbol]['y'].shape}")
            print(f"Data range: [{test_data[symbol]['X'].min():.2f}, {test_data[symbol]['X'].max():.2f}]")
        else:
            print("No test data available")

if __name__ == "__main__":
    test_full_pipeline()
