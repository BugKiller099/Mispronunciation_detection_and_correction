# simple_train.py - Fast training with simple CNN model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from pathlib import Path
from config import Config

# ============================================================================
# Simple CNN Model (Much faster than Wav2Vec2)
# ============================================================================

class SimpleCNNModel(nn.Module):
    """
    Simple CNN for audio classification.
    Much faster than Wav2Vec2, works well on CPU.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=80, stride=16)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, config.NUM_LABELS)
        )
        
        # Score regression head
        self.regressor = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, audio):
        """Forward pass"""
        # Add channel dimension: [batch, samples] -> [batch, 1, samples]
        x = audio.unsqueeze(1)
        
        # Conv layers
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        
        # Pool to [batch, 128, 1]
        x = self.pool(x)
        x = x.squeeze(-1)  # [batch, 128]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification and regression
        logits = self.classifier(x)
        scores = self.regressor(x)
        
        return logits, scores

# ============================================================================
# Simple Dataset
# ============================================================================

class SimpleDataset(Dataset):
    """Simple PyTorch Dataset"""
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'audio': sample['audio'],
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'score': torch.tensor(sample['overall_score'], dtype=torch.float)
        }

# ============================================================================
# Simple Trainer
# ============================================================================

class SimpleTrainer:
    """Simple trainer for the CNN model"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"\n{'='*70}")
        print(f"INITIALIZING SIMPLE TRAINER")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        
        # Model
        self.model = SimpleCNNModel(self.config).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.NUM_EPOCHS}")
        
        for batch in pbar:
            # Move to device
            audio = batch['audio'].to(self.device)
            labels = batch['label'].to(self.device)
            scores = batch['score'].to(self.device).unsqueeze(1)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward
            logits, pred_scores = self.model(audio)
            
            # Calculate loss
            loss_cls = self.classification_loss(logits, labels)
            loss_reg = self.regression_loss(pred_scores, scores)
            loss = loss_cls + 0.3 * loss_reg
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"\n⚠️  NaN loss detected! Skipping batch...")
                continue
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })
        
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_acc = 100 * correct / total if total > 0 else 0
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                audio = batch['audio'].to(self.device)
                labels = batch['label'].to(self.device)
                scores = batch['score'].to(self.device).unsqueeze(1)
                
                # Forward
                logits, pred_scores = self.model(audio)
                
                # Calculate loss
                loss_cls = self.classification_loss(logits, labels)
                loss_reg = self.regression_loss(pred_scores, scores)
                loss = loss_cls + 0.3 * loss_reg
                
                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_acc = 100 * correct / total if total > 0 else 0
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        # Save checkpoint
        checkpoint_path = self.config.CHECKPOINT_DIR / f"checkpoint_latest.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.config.BEST_MODEL_DIR / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved (val_loss: {self.best_val_loss:.4f})")
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print(f"\n{'='*70}")
        print("STARTING TRAINING")
        print(f"{'='*70}\n")
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Check if best
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
            else:
                self.save_checkpoint(is_best=False)
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save history
        history_path = self.config.RESULTS_DIR / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"✓ Training history saved to: {history_path}")

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function"""
    config = Config()
    config.create_directories()
    
    # Load preprocessed data
    data_file = config.PROCESSED_DATA_DIR / "preprocessed_data.pt"
    
    if not data_file.exists():
        print(f"✗ Preprocessed data not found: {data_file}")
        print("Please run preprocessing.py first!")
        return
    
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")
    print(f"Loading from: {data_file}")
    
    all_data = torch.load(data_file)
    print(f"✓ Loaded {len(all_data)} samples")
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    # Get train/test split
    train_data = [d for d in all_data if d.get('split') == 'train']
    test_data = [d for d in all_data if d.get('split') == 'test']
    
    # If no split exists, create one
    if not train_data:
        labels = [d['label'] for d in all_data]
        train_data, test_data = train_test_split(
            all_data,
            test_size=0.2,
            random_state=config.SEED,
            stratify=labels
        )
    
    # Split train into train and validation
    if len(train_data) > 10:
        train_labels = [d['label'] for d in train_data]
        train_data, val_data = train_test_split(
            train_data,
            test_size=0.15,
            random_state=config.SEED,
            stratify=train_labels
        )
    else:
        val_data = test_data
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Test:  {len(test_data)}")
    
    # Create datasets
    train_dataset = SimpleDataset(train_data)
    val_dataset = SimpleDataset(val_data)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Train
    trainer = SimpleTrainer(config)
    trainer.train(train_loader, val_loader)
    
    print("\n✅ Training completed successfully!")

if __name__ == "__main__":
    main()