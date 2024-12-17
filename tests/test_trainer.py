# tests/test_trainer.py

import torch
from utils.training.trainer import ModelTrainer
from models.rnn.model import StockRNN
import numpy as np

def test_trainer():
    print("Testing Model Trainer")
    print("-" * 50)
    
    # Create dummy data
    batch_size = 32
    sequence_length = 100
    n_features = 1
    n_samples = 1000
    
    # Generate random data
    X = torch.randn(n_samples, sequence_length, n_features)
    y = torch.randn(n_samples, 10)  # 10-step prediction
    
    # Split into train/val
    train_size = int(0.8 * n_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Create model
    model = StockRNN(
        input_size=n_features,
        hidden_size=64,
        num_layers=2,
        output_size=10
    )
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        optimizer_name='adam',
        learning_rate=0.001,
        batch_size=32,
        epochs=5,  # Small number for testing
        patience=3
    )
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        verbose=True
    )
    
    print("\nTraining completed!")
    print(f"Best loss: {history['best_loss']:.6f}")
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Training time: {history['training_time']:.2f} seconds")

if __name__ == "__main__":
    test_trainer()
