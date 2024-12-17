# utils/evaluation/evaluator.py

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Union
import logging

class ModelEvaluator:
    def __init__(
        self,
        results_dir: str = 'results',
        checkpoint_dir: str = 'results/checkpoints'
    ):
        self.results_dir = Path(results_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Store evaluation results
        self.evaluation_results = {
            'rnn': {},
            'lstm': {},
            'cnn': {},
            'arima': {}
        }
    
    def load_model(self, model_type: str, model_class: torch.nn.Module) -> torch.nn.Module:
        """Load best model checkpoint"""
        model_dir = self.checkpoint_dir / model_type
        
        # Find checkpoint with lowest loss
        checkpoints = list(model_dir.glob('model_epoch_*.pt'))
        best_loss = float('inf')
        best_checkpoint = None
        
        for checkpoint in checkpoints:
            checkpoint_data = torch.load(checkpoint)
            if checkpoint_data['loss'] < best_loss:
                best_loss = checkpoint_data['loss']
                best_checkpoint = checkpoint
        
        if best_checkpoint is None:
            raise ValueError(f"No checkpoints found for {model_type}")
        
        # Load best checkpoint
        model = model_class()
        checkpoint_data = torch.load(best_checkpoint)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        print(f"Loaded checkpoint: {best_checkpoint} with loss: {best_loss:.6f}")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
        
        model.eval()
        return model

    def calculate_error_percentage(
        self,
        true_values: Union[np.ndarray, torch.Tensor],
        predictions: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Calculate mean absolute percentage error (MAPE)
        ep = mean(|true - predicted| / |true|) * 100
        """
        if torch.is_tensor(true_values):
            true_values = true_values.cpu().numpy()
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
            
        # Ensure arrays are flattened
        true_values = true_values.flatten()
        predictions = predictions.flatten()
        
        # Avoid division by zero
        epsilon = 1e-8
        percentage_errors = np.abs(true_values - predictions) / (np.abs(true_values) + epsilon)
        
        # Remove any invalid values
        valid_errors = percentage_errors[~np.isnan(percentage_errors) & ~np.isinf(percentage_errors)]
        
        if len(valid_errors) == 0:
            return float('inf')
        
        return float(np.mean(valid_errors) * 100)

    def evaluate_model(
        self,
        model_type: str,
        model: Union[torch.nn.Module, object],
        test_data: Dict,
        scaler_dict: Dict = None
    ) -> Dict:
        """Evaluate model performance"""
        results = {}
        
        for symbol in test_data.keys():
            X_test = test_data[symbol]['X']
            y_test = test_data[symbol]['y']
            
            # Store original scale for reference
            orig_scale = {
                'X_min': float(X_test.min()),
                'X_max': float(X_test.max()),
                'y_min': float(y_test.min()),
                'y_max': float(y_test.max())
            }
            
            # Normalize input data to [0,1] range for model
            X_test_norm = (X_test - orig_scale['X_min']) / (orig_scale['X_max'] - orig_scale['X_min'])
            
            # Make predictions
            with torch.no_grad():
                if isinstance(model, torch.nn.Module):
                    # Get predictions in normalized space
                    predictions_norm = model(X_test_norm)
                    
                    # Scale predictions back to original scale
                    predictions = predictions_norm * (orig_scale['y_max'] - orig_scale['y_min']) + orig_scale['y_min']
                    
                    # Print debug info
                    print(f"\n{model_type.upper()} Prediction Stats for {symbol}:")
                    print(f"Input shape: {X_test.shape}")
                    print(f"Original scale - Input: [{orig_scale['X_min']:.2f}, {orig_scale['X_max']:.2f}]")
                    print(f"Original scale - Target: [{orig_scale['y_min']:.2f}, {orig_scale['y_max']:.2f}]")
                    print(f"Predictions range: [{float(predictions.min()):.2f}, {float(predictions.max()):.2f}]")
                    
                else:  # ARIMA
                    predictions = model.predict(n_steps=y_test.shape[1])
                    predictions = np.array(predictions)
                    predictions = torch.FloatTensor(predictions).to(y_test.device)
            
            # Convert to numpy for metric calculation
            if torch.is_tensor(y_test):
                y_test = y_test.cpu().numpy()
            if torch.is_tensor(predictions):
                predictions = predictions.cpu().numpy()
            
            # Calculate metrics
            error_percentage = self.calculate_error_percentage(y_test, predictions)
            mse = np.mean((y_test - predictions) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - predictions))
            
            results[symbol] = {
                'error_percentage': error_percentage,
                'predictions': predictions,
                'true_values': y_test,
                'metrics': {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae)
                }
            }
            
            print(f"\nMetrics for {symbol}:")
            print(f"Error Percentage: {error_percentage:.2f}%")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
        
        self.evaluation_results[model_type] = results
        return results

    def plot_predictions(
        self,
        symbol: str,
        save: bool = True
    ):
        """Plot predictions vs true values for all models"""
        plt.figure(figsize=(15, 10))
        
        for model_type, results in self.evaluation_results.items():
            if symbol in results:
                # Reshape data if needed
                true_values = results[symbol]['true_values'].reshape(-1)
                predictions = results[symbol]['predictions'].reshape(-1)
                
                error = results[symbol]['error_percentage']
                metrics = results[symbol]['metrics']
                
                plt.plot(true_values, label=f'True - {symbol}', linewidth=2)
                plt.plot(predictions, '--', 
                        label=f'{model_type.upper()}\nError: {error:.2f}%\nRMSE: {metrics["rmse"]:.4f}',
                        alpha=0.7)
        
        plt.title(f'Stock Price Predictions - {symbol}')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        if save:
            plt.savefig(self.results_dir / f'predictions_{symbol}.png', 
                       dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self):
        """Save evaluation results"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        serializable_results = {}
        for model_type, results in self.evaluation_results.items():
            serializable_results[model_type] = {}
            for symbol, metrics in results.items():
                serializable_results[model_type][symbol] = {
                    'error_percentage': float(metrics['error_percentage']),
                    'predictions': metrics['predictions'].tolist(),
                    'true_values': metrics['true_values'].tolist(),
                    'metrics': metrics['metrics']
                }
        
        results_file = self.results_dir / f'evaluation_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"\nSaved detailed results to {results_file}")

    def create_summary(self) -> pd.DataFrame:
        """Create summary of results"""
        summary = []
        
        for model_type, results in self.evaluation_results.items():
            for symbol, metrics in results.items():
                summary.append({
                    'Model': model_type.upper(),
                    'Symbol': symbol,
                    'Error %': metrics['error_percentage'],
                    'RMSE': metrics['metrics']['rmse'],
                    'MAE': metrics['metrics']['mae']
                })
        
        df = pd.DataFrame(summary)
        print("\nEvaluation Summary:")
        print(df)
        return df
