# models/cnn/model.py

import torch
import torch.nn as nn

# models/cnn/model.py

class StockCNN(nn.Module):
   def __init__(self, input_size=1, sequence_length=100, output_size=10, n_filters=[32, 64, 128], kernel_size=3, dropout=0.2):
       super(StockCNN, self).__init__()
       
       self.input_size = input_size
       self.sequence_length = sequence_length
       
       cnn_layers = []
       in_channels = input_size
       
       for out_channels in n_filters:
           cnn_layers.extend([
               nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
               nn.ReLU(),
               nn.BatchNorm1d(out_channels),
               nn.Dropout(dropout)
           ])
           in_channels = out_channels
       
       self.cnn = nn.Sequential(*cnn_layers)
       self.flatten = nn.Flatten()
       
       # Calculate the size after CNN layers
       cnn_output_size = n_filters[-1] * sequence_length
       
       self.fc_layers = nn.Sequential(
           nn.Linear(cnn_output_size, 128),
           nn.ReLU(),
           nn.Dropout(dropout),
           nn.Linear(128, output_size)
       )
   
   def forward(self, x):
       batch_size = x.shape[0]
       
       # Ensure input is shaped correctly (batch_size, features, seq_len)
       if len(x.shape) == 2:
           x = x.unsqueeze(1)  # Add feature dimension
       elif x.shape[-1] != self.sequence_length:
           x = x.transpose(1, 2)  # Transpose if needed
           
       x = self.cnn(x)
       x = self.flatten(x)
       return self.fc_layers(x)
