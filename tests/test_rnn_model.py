# tests/test_rnn_model.py

import torch
from models.rnn.model import StockRNN

def test_rnn_model():
    print("Testing RNN Model")
    print("-" * 50)
    
    # Test parameters
    batch_size = 32
    sequence_length = 100
    input_size = 1
    
    # Create model
    model = StockRNN(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        output_size=10
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Create dummy input
    x = torch.randn(batch_size, sequence_length, input_size).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print("\nModel Architecture:")
    print(model)
    
    print("\nInput shape:", x.shape)
    print("Output shape:", output.shape)
    
    # Test different batch sizes
    test_batches = [1, 16, 32, 64]
    print("\nTesting different batch sizes:")
    for batch in test_batches:
        x = torch.randn(batch, sequence_length, input_size).to(device)
        with torch.no_grad():
            output = model(x)
        print(f"Batch size {batch:2d} - Output shape: {output.shape}")

if __name__ == "__main__":
    test_rnn_model()
