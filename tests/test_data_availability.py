# tests/test_data_availability.py

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

def explore_available_data():
    print("Starting data exploration...")
    
    # Original paper stocks + NVIDIA and Tesla
    symbols = {
        'Original Paper': ['INFY', 'TCS.NS', 'CIPLA.NS'],
        'Tech Comparison': ['NVDA', 'TSLA']
    }
    
    for category, stocks in symbols.items():
        print(f"\n{category} Stocks:")
        print("-" * 50)
        
        for symbol in stocks:
            print(f"\nAnalyzing {symbol}:")
            stock = yf.Ticker(symbol)
            
            # Test different intervals
            test_cases = [
                ("1m", "1d"),     # 1 minute data for today
                ("1h", "1mo"),    # hourly data for a month
                ("1d", "1y")      # daily data for a year
            ]
            
            for interval, period in test_cases:
                try:
                    data = stock.history(period=period, interval=interval)
                    print(f"\n{interval} data ({period}):")
                    print(f"Number of data points: {len(data)}")
                    if len(data) > 0:
                        print(f"Date range: {data.index[0]} to {data.index[-1]}")
                        print(f"Trading statistics:")
                        print(f"  High: ${data['High'].max():.2f}")
                        print(f"  Low: ${data['Low'].min():.2f}")
                        print(f"  Volume: {data['Volume'].sum():,}")
                        
                except Exception as e:
                    print(f"Error with {interval} data ({period}): {str(e)}")

if __name__ == "__main__":
    explore_available_data()
