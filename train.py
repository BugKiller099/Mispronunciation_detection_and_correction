# ============================================================================
# FILE 7: train.py
# Training script with full training loop
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json

from config import Config
from model import MispronunciationDetector
from dataset import create_dataloaders

class Trainer:
    """
    Trainer class for mispronunciation detection model.
    
    Handles:
    - Training loop with backpropagation
    - Validation during training
    - Model checkpointing
    - Early stopping
    - Logging to TensorBoard
    """
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.config.create_directories()
        
        # Set device
        self.device = torch.device(
            self.config.DEVICE if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")
        
        
        
        # Initialize model
        print("\nInitializing model...")
        self.model = MispronunciationDetector(self.config).to(self.device)
        
        # <<< CRITICAL MODIFICATION TO FIX LOSS=NAN >>>
        
        print("Freezing ENTIRE Wav2Vec2 base model...")
        
        # Access the entire Wav2Vec2 model object
        model_to_freeze = self.model.wav2vec2 
        
        # Disable gradients for ALL parameters in the Wav2Vec2 model
        for name, param in model_to_freeze.named_parameters():
            # Exclude the 'feature_extractor' which you already did, and now include the 'encoder'
            param.requires_grad = False
            
        print("Wav2Vec2 base model frozen. Only the classification/regression head will train.")
        
        print(f"✓ Model loaded with {self.count_parameters()} parameters")

        
        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss()
        self.regression_criterion = nn.MSELoss()
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            #verbose=True
        )
        
        # TensorBoard writer
        self.writer = SummaryWriter(self.config.TENSORBOARD_DIR)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average loss and accuracy for the epoch
        """
        self.model.train()
        
        total_loss = 0
        total_classification_loss = 0
        total_regression_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            audio = batch['audio'].to(self.device)
            labels = batch['label'].to(self.device)
            scores = batch['score'].to(self.device).unsqueeze(1)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, predicted_scores = self.model(audio)
            
            # Calculate losses
            classification_loss = self.classification_criterion(logits, labels)
            regression_loss = self.regression_criterion(predicted_scores, scores)
            
            # Combined loss (weighted)
            loss = classification_loss + 0.3 * regression_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.MAX_GRAD_NORM
            )
            
            # Update weights
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Accumulate losses
            total_loss += loss.item()
            total_classification_loss += classification_loss.item()
            total_regression_loss += regression_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * correct / total
            })
            
            # Log to TensorBoard
            global_step = self.current_epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100 * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss and accuracy
        """
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                audio = batch['audio'].to(self.device)
                labels = batch['label'].to(self.device)
                scores = batch['score'].to(self.device).unsqueeze(1)
                
                # Forward pass
                logits, predicted_scores = self.model(audio)
                
                # Calculate loss
                classification_loss = self.classification_criterion(logits, labels)
                regression_loss = self.regression_criterion(predicted_scores, scores)
                loss = classification_loss + 0.3 * regression_loss
                
                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = 100 * correct / total
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': vars(self.config)
        }
        
        # Save regular checkpoint
        checkpoint_path = self.config.CHECKPOINT_DIR / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.config.BEST_MODEL_DIR / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved (val_loss: {self.best_val_loss:.4f})")
    
    def train(self, train_loader, val_loader):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        for epoch in range(self.config.NUM_EPOCHS):
            self.current_epoch = epoch + 1
            
            print(f"\nEpoch {self.current_epoch}/{self.config.NUM_EPOCHS}")
            print("-" * 70)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            self.writer.add_scalar('Epoch/TrainLoss', train_loss, self.current_epoch)
            self.writer.add_scalar('Epoch/ValLoss', val_loss, self.current_epoch)
            self.writer.add_scalar('Epoch/TrainAcc', train_acc, self.current_epoch)
            self.writer.add_scalar('Epoch/ValAcc', val_acc, self.current_epoch)
            
            # Print epoch summary
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {self.current_epoch} epochs")
                break
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save training history
        history_path = self.config.RESULTS_DIR / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=4)
        
        self.writer.close()

def main():
    """Main training function"""
    # Set random seed
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders()
    
    # Initialize trainer
    trainer = Trainer()
    
    # Start training
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
