# scripts/train_models.py

import torch
from pathlib import Path
import json
from datetime import datetime
import logging
import sys

from utils.preprocessing.data_pipeline import StockDataPipeline
from utils.training.trainer import ModelTrainer
from models.rnn.model import StockRNN
from models.lstm.model import StockLSTM
from models.cnn.model import StockCNN
from models.arima.model import StockARIMA

# Setup logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   handlers=[
       logging.StreamHandler(sys.stdout),
       logging.FileHandler('results/training.log')
   ]
)

class ModelTrainingPipeline:
   def __init__(self):
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
       
       # Initialize results directory
       self.results_dir = Path('results')
       self.results_dir.mkdir(exist_ok=True)
       
       # Initialize data pipeline
       self.data_pipeline = StockDataPipeline()
       
   def train_all_models(self):
       """Train all models and save results"""
       logging.info("Starting model training pipeline")
       
       # Get data
       logging.info("Preparing data...")
       data = self.data_pipeline.prepare_train_test_data()
       
       results = {}
       
       # Train deep learning models
       for model_name in ['rnn', 'lstm', 'cnn']:
           logging.info(f"\nTraining {model_name.upper()} model")
           
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
               checkpoint_dir=f'results/checkpoints/{model_name}'
           )
           
           # Train model
           try:
               history = trainer.train(
                   data['train']['X'],
                   data['train']['y'],
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
           
           # Train and evaluate ARIMA
           train_data = data['train']['X'].cpu().numpy()[:, :, 0]  # Convert to numpy and remove feature dimension
           arima_start_time = datetime.now()
           
           # Fit model on last sequence of training data
           arima.fit(train_data[-1])
           
           # Make predictions
           predictions = arima.predict(n_steps=10)
           
           arima_time = (datetime.now() - arima_start_time).total_seconds()
           
           results['arima'] = {
               'training_time': arima_time
           }
           
           logging.info("ARIMA training completed:")
           logging.info(f"Training time: {arima_time:.2f} seconds")
           
       except Exception as e:
           logging.error(f"Error training ARIMA: {str(e)}")
       
       # Save results
       self._save_results(results)
       
       return results
   
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
