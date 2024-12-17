# scripts/train_and_evaluate.py

import torch
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

from utils.preprocessing.data_pipeline import StockDataPipeline
from utils.training.trainer import ModelTrainer
from utils.evaluation.evaluator import ModelEvaluator
from models.rnn.model import StockRNN
from models.lstm.model import StockLSTM
from models.cnn.model import StockCNN
from models.arima.model import StockARIMA

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('results/training_evaluation.log')
    ]
)

class StockPredictionPipeline:
    def __init__(self):
        self.data_pipeline = StockDataPipeline()
        self.evaluator = ModelEvaluator()
        
        # Model configurations
        self.model_configs = {
            'rnn': {
                'model_class': StockRNN,
                'params': {
                    'input_size': 1,
                    'hidden_size': 64,
                    'num_layers': 2,
                    'output_size': 10
                },
                'training': {
                    'lr': 0.001,
                    'batch_size': 32,
                    'epochs': 1000,
                    'patience': 50
                }
            },
            'lstm': {
                'model_class': StockLSTM,
                'params': {
                    'input_size': 1,
                    'hidden_size': 64,
                    'num_layers': 2,
                    'output_size': 10
                },
                'training': {
                    'lr': 0.001,
                    'batch_size': 32,
                    'epochs': 1000,
                    'patience': 50
                }
            },
            'cnn': {
                'model_class': StockCNN,
                'params': {
                    'input_size': 1,
                    'sequence_length': 100,
                    'output_size': 10,
                    'n_filters': [32, 64, 128]
                },
                'training': {
                    'lr': 0.001,
                    'batch_size': 32,
                    'epochs': 1000,
                    'patience': 50
                }
            }
        }
        
        # Results storage
        self.results = {}
    
    def train_models(self, train_data):
        """Train all models"""
        trained_models = {}
        
        for model_name, config in self.model_configs.items():
            logging.info(f"\nTraining {model_name.upper()}...")
            
            # Initialize model
            model = config['model_class'](**config['params'])
            if torch.cuda.is_available():
                model = model.cuda()
            
            # Create trainer
            trainer = ModelTrainer(
                model=model,
                learning_rate=config['training']['lr'],
                batch_size=config['training']['batch_size'],
                epochs=config['training']['epochs'],
                patience=config['training']['patience']
            )
            
            # Train model
            history = trainer.train(
                train_data['X'],
                train_data['y'],
                verbose=True
            )
            
            trained_models[model_name] = {
                'model': model,
                'history': history
            }
            
            logging.info(f"{model_name.upper()} training completed")
            logging.info(f"Best loss: {history['best_loss']:.6f}")
            logging.info(f"Training time: {history['training_time']:.2f} seconds")
        
        # Train ARIMA baseline
        logging.info("\nTraining ARIMA baseline...")
        arima = StockARIMA()
        train_sequence = train_data['X'].cpu().numpy()

        arima.fit(train_sequence[0,:])
        trained_models['arima'] = {
            'model': arima,
            'history': None
        }
        
        return trained_models
    
    def evaluate_models(self, models, test_data):
        """Evaluate all models"""
        logging.info("\nEvaluating models...")
        
        for model_name, model_dict in models.items():
            logging.info(f"\nEvaluating {model_name.upper()}...")
            results = self.evaluator.evaluate_model(
                model_name,
                model_dict['model'],
                test_data
            )
            
            self.results[model_name] = {
                'evaluation': results,
                'training_history': model_dict['history']
            }
    
    def run_pipeline(self):
        """Run complete training and evaluation pipeline"""
        logging.info("Starting pipeline...")
        
        # Get data
        data = self.data_pipeline.prepare_train_test_data()
        
        # Train models
        trained_models = self.train_models(data['train'])
        
        # Evaluate models
        self.evaluate_models(trained_models, data['test'])
        
        # Generate results
        self.evaluator.plot_predictions('INFY')
        self.evaluator.save_results()
        summary = self.evaluator.create_summary()
        
        # Save complete results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'results/complete_results_{timestamp}.json'
        
        torch.save({
            'results': self.results,
            'summary': summary.to_dict(),
            'timestamp': timestamp
        }, results_file)
        
        logging.info(f"\nComplete results saved to {results_file}")
        return summary

def main():
    pipeline = StockPredictionPipeline()
    summary = pipeline.run_pipeline()
    print("\nFinal Results Summary:")
    print(summary)

if __name__ == "__main__":
    main()
