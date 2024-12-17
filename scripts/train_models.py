# scripts/train_models.py

import torch
from pathlib import Path
import json
from datetime import datetime
import logging
import sys
import os

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'
LOG_DIR = RESULTS_DIR / 'logs'

# Create necessary directories
RESULTS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / 'training.log')
    ]
)

# Imports
from utils.preprocessing.data_pipeline import StockDataPipeline
from utils.training.trainer import ModelTrainer
from models.rnn.model import StockRNN
from models.lstm.model import StockLSTM
from models.cnn.model import StockCNN
from models.arima.model import StockARIMA

class ModelTrainingPipeline:
    def __init__(self):
        """Initialize the training pipeline"""
        # Initialize data pipeline first
        self.data_pipeline = StockDataPipeline()
        
        # Initialize results directory
        self.results_dir = RESULTS_DIR
        self.checkpoints_dir = self.results_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.configs = {
            'rnn': {
                'model_params': {
                    'input_size': 1,
                    'hidden_size': 64,
                    'num_layers': 2,
                    'output_size': 10,
                    'dropout': 0.2
                },
                'training_params': {
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 1000,
                    'patience': 50,
                    'optimizer': 'adam'
                }
            },
            'lstm': {
                'model_params': {
                    'input_size': 1,
                    'hidden_size': 64,
                    'num_layers': 2,
                    'output_size': 10,
                    'dropout': 0.2
                },
                'training_params': {
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 1000,
                    'patience': 50,
                    'optimizer': 'adam'
                }
            },
            'cnn': {
                'model_params': {
                    'input_size': 1,
                    'sequence_length': 100,
                    'output_size': 10,
                    'n_filters': [32, 64, 128],
                    'kernel_size': 3,
                    'dropout': 0.2
                },
                'training_params': {
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 1000,
                    'patience': 50,
                    'optimizer': 'adam'
                }
            }
        }

    def _create_model(self, model_name: str):
        """Create model instance based on name"""
        if model_name == 'rnn':
            return StockRNN(**self.configs['rnn']['model_params'])
        elif model_name == 'lstm':
            return StockLSTM(**self.configs['lstm']['model_params'])
        elif model_name == 'cnn':
            return StockCNN(**self.configs['cnn']['model_params'])
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def prepare_data_for_model(self, data, model_type):
        """Prepare data according to model requirements"""
        X_train = data['train']['X']
        y_train = data['train']['y']
        
        # Ensure on CPU for numpy operations
        if torch.is_tensor(X_train):
            X_train = X_train.cpu()
        if torch.is_tensor(y_train):
            y_train = y_train.cpu()

        if model_type == 'arima':
        # ARIMA expects 1D array of the closing prices
            if torch.is_tensor(X_train):
                X_train = X_train.numpy()
        # Take the closing prices from the last feature dimension
            if len(X_train.shape) == 3:
            # (batch, sequence, features) -> flatten the sequence
                return X_train[:, :, 0].flatten(), None
            else:
                return X_train.flatten(), None
        else:
        # Deep learning models
            if not torch.is_tensor(X_train):
                X_train = torch.FloatTensor(X_train)
                y_train = torch.FloatTensor(y_train)
        
        # Reshape if needed
            if len(X_train.shape) == 2:
                X_train = X_train.unsqueeze(-1)  # Add feature dimension
        
            return X_train, y_train

    def train_all_models(self):
        """Train all models and save results"""
        logging.info("Starting model training pipeline")
        
        # Get data
        logging.info("Preparing data...")
        raw_data = self.data_pipeline.prepare_train_test_data()
        
        results = {}
        
        # Train deep learning models
        for model_name in ['rnn', 'lstm', 'cnn']:
            logging.info(f"\nTraining {model_name.upper()} model")
            
            try:
                # Prepare data for specific model
                X_train, y_train = self.prepare_data_for_model(raw_data, model_name)
                
                # Create model instance
                model = self._create_model(model_name)
                
                # Get training parameters
                train_params = self.configs[model_name]['training_params']
                
                # Create trainer
                trainer = ModelTrainer(
                    model=model,
                    optimizer_name=train_params['optimizer'],
                    learning_rate=train_params['learning_rate'],
                    batch_size=train_params['batch_size'],
                    epochs=train_params['epochs'],
                    patience=train_params['patience'],
                    checkpoint_dir=str(self.checkpoints_dir / model_name)
                )
                
                # Train model
                history = trainer.train(
                    X_train,
                    y_train,
                    verbose=True
                )
                
                results[model_name] = {
                    'training_history': history,
                    'best_loss': history['best_loss'],
                    'training_time': history['training_time']
                }
                
                logging.info(f"{model_name.upper()} training completed:")
                logging.info(f"Best loss: {history['best_loss']:.6f}")
                logging.info(f"Training time: {history['training_time']:.2f} seconds")
                
            except Exception as e:
                logging.error(f"Error training {model_name}: {str(e)}")
                continue
        
         # Train ARIMA model
        logging.info("\nTraining ARIMA model")
        try:
            arima = StockARIMA()
    
        # Prepare data for ARIMA
            train_data, _ = self.prepare_data_for_model(raw_data, 'arima')
            arima_start_time = datetime.now()
    
            logging.info(f"ARIMA input shape: {train_data.shape}")
            logging.info(f"ARIMA input range: [{train_data.min():.2f}, {train_data.max():.2f}]")
    
            # Fit model on training data
            arima.fit(train_data)
    
            # Make predictions
            predictions = arima.predict(n_steps=10)
    
            arima_time = (datetime.now() - arima_start_time).total_seconds()
    
            results['arima'] = {
                'training_time': arima_time,
                'predictions': predictions.tolist()  # Make JSON serializable
            }
    
            logging.info("ARIMA training completed:")
            logging.info(f"Training time: {arima_time:.2f} seconds")
            logging.info(f"First few predictions: {predictions[:5]}")
    
        except Exception as e:
            logging.error(f"Error training ARIMA: {str(e)}")
            logging.error(f"Error details:", exc_info=True)  # Add full traceback      
        # Save results
        self._save_results(results)
        
        return results

    def _save_results(self, results: dict):
        """Save training results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.results_dir / f'training_results_{timestamp}.json'
        
        # Convert any non-serializable objects
        serializable_results = {}
        for model, result in results.items():
            serializable_results[model] = {
                k: v if isinstance(v, (int, float, str, list, dict)) else str(v)
                for k, v in result.items()
            }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        logging.info(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    pipeline = ModelTrainingPipeline()
    results = pipeline.train_all_models()
