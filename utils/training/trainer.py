# utils/training/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import time

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer_name: str = 'adam',
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 1000,
        patience: int = 50,
        checkpoint_dir: str = 'results/checkpoints',
        device: Optional[str] = None
    ):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Loss function (MSE as per paper)
        self.criterion = nn.MSELoss()
        
        # Optimizer
        if optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'best_epoch': 0,
            'best_loss': float('inf')
        }
    
    def prepare_data(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Prepare training and validation dataloaders"""
        # Create training dataloader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Create validation dataloader if validation data provided
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for X_batch, y_batch in train_loader:
            # Move to device
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # Move to device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'history': self.history
        }
        
        checkpoint_path = self.checkpoint_dir / f'model_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> Dict:
        """Train model"""
        # Prepare data
        train_loader, val_loader = self.prepare_data(X_train, y_train, X_val, y_val)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        training_start = time.time()
        
        for epoch in range(self.epochs):
            # Train
            epoch_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(epoch_loss)
            
            # Validate
            if val_loader:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                current_loss = val_loss
            else:
                current_loss = epoch_loss
            
            # Check for improvement
            if current_loss < best_loss:
                best_loss = current_loss
                self.history['best_epoch'] = epoch
                self.history['best_loss'] = best_loss
                patience_counter = 0
                self.save_checkpoint(epoch, best_loss)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"\nEpoch [{epoch+1}/{self.epochs}]")
                print(f"Train Loss: {epoch_loss:.6f}")
                if val_loader:
                    print(f"Val Loss: {val_loss:.6f}")
                print(f"Best Loss: {best_loss:.6f}")
        
        training_time = time.time() - training_start
        self.history['training_time'] = training_time
        
        # Save final results
        results_path = self.checkpoint_dir / 'training_history.json'
        with open(results_path, 'w') as f:
            json.dump(self.history, f)
        
        return self.history
