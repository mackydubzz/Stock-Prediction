# utils/preprocessing/data_pipeline.py

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
from typing import Dict, List, Tuple, Union

class StockDataPipeline:
    def __init__(self):
        # Use recent dates
        from datetime import datetime, timedelta
        
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=5)
        
        # Split into train/test (4 days train, 1 day test)
        train_split = self.end_date - timedelta(days=1)
        
        self.train_period = {
            'start': self.start_date.strftime('%Y-%m-%d'),
            'end': train_split.strftime('%Y-%m-%d')
        }
        
        self.test_period = {
            'start': train_split.strftime('%Y-%m-%d'),
            'end': self.end_date.strftime('%Y-%m-%d')
        }
        
        self.paper_stocks = {
            'INFY': 'Training & Testing',
            'TCS.NS': 'Testing Only',
            'CIPLA.NS': 'Testing Only'
        }
        
        self.tech_stocks = {
            'NVDA': 'Comparison',
            'TSLA': 'Comparison'
        }
        
        self.params = {
            'window_size': 100,
            'overlap': 90,
            'prediction_horizon': 10
        }

    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data for the specified period"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period='5d', interval='1m')
            print(f"Fetched {len(data)} points for {symbol}")
            
            if not data.empty:
                data = data[start_date:end_date]
                print(f"After date filtering: {len(data)} points")
                print(f"Price range: [{data['Close'].min():.2f}, {data['Close'].max():.2f}]")
            return data
            
        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")
            return pd.DataFrame()



    def normalize_prices(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        normalized_data = pd.DataFrame()
        
        if 'Close' in data.columns:
            min_val = float(data['Close'].min())
            max_val = float(data['Close'].max())
            print(f"Original range: [{min_val:.2f}, {max_val:.2f}]")
            
            normalized_data['Close'] = (data['Close'] - min_val) / (max_val - min_val)
            print(f"Normalized range: [{normalized_data['Close'].min():.2f}, {normalized_data['Close'].max():.2f}]")
            
            scaler_dict = {'Close': {'min': min_val, 'max': max_val}}
            
        return normalized_data, scaler_dict
        
    def normalize_data_with_scaler(self, data: pd.DataFrame, scaler_dict: Dict) -> pd.DataFrame:
        """Normalize data using pre-computed scaling factors"""
        normalized_data = pd.DataFrame()
        
        if 'Close' in data.columns and 'Close' in scaler_dict:
            min_val = scaler_dict['Close']['min']
            max_val = scaler_dict['Close']['max']
            
            normalized_data['Close'] = (data['Close'] - min_val) / (max_val - min_val)
            
        return normalized_data

    def create_sequences(self, prices: pd.Series, 
                        window_size: int, 
                        overlap: int, 
                        prediction_horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create sliding window sequences"""
        stride = window_size - overlap
        n_sequences = ((len(prices) - window_size - prediction_horizon) // stride) + 1
        
        X = np.zeros((n_sequences, window_size))
        y = np.zeros((n_sequences, prediction_horizon))
        
        for i in range(n_sequences):
            start_idx = i * stride
            end_idx = start_idx + window_size
            target_end_idx = end_idx + prediction_horizon
            
            X[i] = prices[start_idx:end_idx]
            y[i] = prices[end_idx:target_end_idx]
        
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
        
        return X, y

    def prepare_train_test_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        data = {
            'train': {},
            'test': {}
        }
        
        # Get training data
        train_data = self.fetch_historical_data('INFY', 
                                              self.train_period['start'],
                                              self.train_period['end'])
        
        if not train_data.empty:
            normalized_data, self.scaler_dict = self.normalize_prices(train_data)
            
            X, y = self.create_sequences(
                normalized_data['Close'],
                self.params['window_size'],
                self.params['overlap'],
                self.params['prediction_horizon']
            )
            
            data['train'] = {
                'X': X,
                'y': y
            }
        
        for symbol in ['INFY', 'TCS.NS', 'CIPLA.NS']:
            test_data = self.fetch_historical_data(symbol,
                                                 self.test_period['start'],
                                                 self.test_period['end'])
            
            if not test_data.empty:
                normalized_data = self.normalize_data_with_scaler(test_data, self.scaler_dict)
                
                X, y = self.create_sequences(
                    normalized_data['Close'],
                    self.params['window_size'],
                    self.params['overlap'],
                    self.params['prediction_horizon']
                )
                
                data['test'][symbol] = {
                    'X': X,
                    'y': y
                }
        
        return data
    
    def denormalize_prices(self, normalized_prices: torch.Tensor, scaler: Dict) -> torch.Tensor:
        """Convert normalized prices back to original scale"""
        return normalized_prices * (scaler['max'] - scaler['min']) + scaler['min']
