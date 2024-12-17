# models/lstm/model.py
import torch
import torch.nn as nn


class StockLSTM(nn.Module):
   def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=10, dropout=0.2):
       super(StockLSTM, self).__init__()
       
       self.hidden_size = hidden_size
       self.num_layers = num_layers
       self.input_size = input_size
       
       self.lstm = nn.LSTM(
           input_size=input_size,
           hidden_size=hidden_size,
           num_layers=num_layers,
           batch_first=True,
           dropout=dropout if num_layers > 1 else 0
       )
       
       self.fc = nn.Linear(hidden_size, output_size)
   
   def forward(self, x):
       batch_size = x.shape[0]
       
       # Ensure input is shaped correctly (batch_size, seq_len, features)
       if len(x.shape) == 2:
           x = x.unsqueeze(-1)
       elif x.shape[-1] != self.input_size:
           x = x.transpose(-1, -2)
       
       h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
       c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
       
       out, _ = self.lstm(x, (h0, c0))
       out = self.fc(out[:, -1, :])
       return out
