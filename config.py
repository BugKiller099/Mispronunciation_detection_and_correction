# ============================================================================
# FILE 2: config.py
# Save this in: mispronunciation-detection/config.py
# ============================================================================

import os
from pathlib import Path

class Config:
    """Configuration settings for the project"""
    
    # Get the current directory where this file is located
    PROJECT_ROOT = Path(__file__).parent
    
    # Data directories
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # Model directories
    MODEL_DIR = PROJECT_ROOT / "models"
    CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
    BEST_MODEL_DIR = MODEL_DIR / "best_model"
    
    # Logging directories
    LOG_DIR = PROJECT_ROOT / "logs"
    TENSORBOARD_DIR = LOG_DIR / "tensorboard"
    
    # Results directory
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # Model settings
    PRETRAINED_MODEL = "facebook/wav2vec2-base-960h"
    HIDDEN_SIZE = 768
    NUM_LABELS = 3  # Poor, Acceptable, Good
    
    # Audio settings
    SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 10  # seconds
    
    # Training settings
    BATCH_SIZE = 4  # Small batch size for beginners (uses less memory)
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10  # Reduced for faster testing
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    EARLY_STOPPING_PATIENCE = 3
    MAX_GRAD_NORM = 1.0
    
    # Data splits
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Data augmentation
    USE_AUGMENTATION = False  # Disabled for beginners
    NOISE_FACTOR = 0.005
    TIME_STRETCH_RATE = 0.1
    
    # Evaluation settings
    EVAL_BATCH_SIZE = 8
    GOOD_THRESHOLD = 4.0
    ACCEPTABLE_THRESHOLD = 2.5
    
    # System settings
    NUM_WORKERS = 0  # Reduced for stability
    PIN_MEMORY = False
    DEVICE = "cuda"  # Will auto-switch to CPU if CUDA not available
    SEED = 42
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR,
            cls.MODEL_DIR, cls.CHECKPOINT_DIR, cls.BEST_MODEL_DIR,
            cls.LOG_DIR, cls.TENSORBOARD_DIR, cls.RESULTS_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        print("✓ All directories created!")

# Test if this file works
if __name__ == "__main__":
    print("Testing config.py...")
    Config.create_directories()
    print(f"Project root: {Config.PROJECT_ROOT}")
    print("✓ Config file is working correctly!")


# ============================================================================
# FILE 3: download_dataset.py
# Save this in: mispronunciation-detection/download_dataset.py
# ============================================================================

import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from config import Config

class DatasetDownloader:
    """Download and prepare the Speechocean762 dataset"""
    
    def __init__(self):
        self.config = Config()
        self.config.create_directories()
    
    def download_instructions(self):
        """Show download instructions"""
        print("\n" + "="*70)
        print("SPEECHOCEAN762 DATASET SETUP")
        print("="*70)
        
        print("\n📥 DOWNLOAD INSTRUCTIONS:")
        print("-" * 70)
        print("\n1. Open your web browser")
        print("2. Go to: https://www.openslr.org/101/")
        print("3. Download these two files:")
        print("   • train.tar.gz")
        print("   • test.tar.gz")
        print(f"\n4. Save them to this folder:")
        print(f"   {self.config.RAW_DATA_DIR}")
        print("\n5. Come back here and press Enter to continue")
        print("-" * 70)
        
        input("\nPress Enter after you've downloaded the files...")
        
        return self.check_files()
    
    def check_files(self):
        """Check if dataset files exist"""
        train_file = self.config.RAW_DATA_DIR / "train.tar.gz"
        test_file = self.config.RAW_DATA_DIR / "test.tar.gz"
        
        if train_file.exists() and test_file.exists():
            print("\n✓ Files found! Starting extraction...")
            return True
        else:
            print("\n✗ Files not found!")
            print(f"Looking for files in: {self.config.RAW_DATA_DIR}")
            print("\nMake sure you placed:")
            print("  • train.tar.gz")
            print("  • test.tar.gz")
            print(f"in the folder: {self.config.RAW_DATA_DIR}")
            return False
    
    def extract_dataset(self):
        """Extract the downloaded files"""
        import tarfile
        
        print("\nExtracting dataset files...")
        
        files = [
            self.config.RAW_DATA_DIR / "train.tar.gz",
            self.config.RAW_DATA_DIR / "test.tar.gz"
        ]
        
        for file_path in files:
            if file_path.exists():
                print(f"  Extracting {file_path.name}...")
                try:
                    with tarfile.open(file_path, 'r:gz') as tar:
                        tar.extractall(path=self.config.RAW_DATA_DIR)
                    print(f"  ✓ {file_path.name} extracted")
                except Exception as e:
                    print(f"  ✗ Error extracting {file_path.name}: {e}")
                    return False
        
        print("\n✓ Extraction complete!")
        return True
    
    def create_sample_dataset(self):
        """Create a small sample dataset for testing"""
        print("\n" + "="*70)
        print("CREATING SAMPLE DATASET FOR TESTING")
        print("="*70)
        print("\nSince the real dataset is large, let's create a small")
        print("sample dataset so you can test the pipeline quickly.")
        
        # Create sample data
        sample_data = []
        
        # Create 30 sample entries
        for i in range(30):
            # Simulate different quality levels
            if i < 10:
                score = 4.5 - (i * 0.1)  # Good
                label = 2
            elif i < 20:
                score = 3.0 + (i % 5) * 0.2  # Acceptable
                label = 1
            else:
                score = 2.0 - (i % 5) * 0.1  # Poor
                label = 0
            
            split = 'train' if i < 24 else 'test'
            
            sample_data.append({
                'audio_path': f'sample_audio_{i}.wav',
                'split': split,
                'accuracy': score,
                'completeness': score,
                'fluency': score,
                'prosody': score,
                'overall_score': score,
                'label': label
            })
        
        # Create DataFrame
        df = pd.DataFrame(sample_data)
        
        # Save to CSV
        output_file = self.config.PROCESSED_DATA_DIR / "dataset_info.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ Sample dataset created!")
        print(f"✓ Saved to: {output_file}")
        print(f"\n📊 Dataset statistics:")
        print(f"  Total samples: {len(df)}")
        print(f"  Training samples: {len(df[df['split']=='train'])}")
        print(f"  Test samples: {len(df[df['split']=='test'])}")
        print(f"\n  Label distribution:")
        print(df['label'].value_counts().to_string())
        
        return df
    
    def organize_real_dataset(self):
        """Organize the real Speechocean762 dataset"""
        print("\nOrganizing dataset files...")
        
        # Look for extracted directories
        train_dir = self.config.RAW_DATA_DIR / "train"
        test_dir = self.config.RAW_DATA_DIR / "test"
        
        if not train_dir.exists() or not test_dir.exists():
            print("✗ Could not find extracted directories")
            return None
        
        data_entries = []
        
        # Process train and test
        for split_name, split_dir in [("train", train_dir), ("test", test_dir)]:
            wav_files = list(split_dir.rglob("*.wav"))
            print(f"  Found {len(wav_files)} audio files in {split_name}")
            
            for wav_file in tqdm(wav_files[:100], desc=f"Processing {split_name}"):  # Limit to 100 for testing
                label_file = wav_file.with_suffix('.txt')
                
                # Default scores
                score = 3.0
                
                if label_file.exists():
                    try:
                        with open(label_file, 'r') as f:
                            content = f.read()
                            # Try to extract score (format varies)
                            if 'score' in content.lower():
                                import re
                                scores = re.findall(r'\d+\.?\d*', content)
                                if scores:
                                    score = float(scores[0])
                    except:
                        pass
                
                # Determine label
                if score >= 4.0:
                    label = 2  # Good
                elif score >= 2.5:
                    label = 1  # Acceptable
                else:
                    label = 0  # Poor
                
                data_entries.append({
                    'audio_path': str(wav_file),
                    'split': split_name,
                    'accuracy': score,
                    'completeness': score,
                    'fluency': score,
                    'prosody': score,
                    'overall_score': score,
                    'label': label
                })
        
        if not data_entries:
            print("✗ No data found")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(data_entries)
        
        # Save
        output_file = self.config.PROCESSED_DATA_DIR / "dataset_info.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ Dataset organized!")
        print(f"✓ Total samples: {len(df)}")
        print(f"✓ Saved to: {output_file}")
        
        return df
    
    def run(self):
        """Main function to run the dataset setup"""
        print("\n" + "="*70)
        print("DATASET SETUP WIZARD")
        print("="*70)
        
        print("\nChoose an option:")
        print("1. Download real dataset (Speechocean762)")
        print("2. Create sample dataset for testing (recommended for beginners)")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == "2":
            # Create sample dataset
            self.create_sample_dataset()
            print("\n✓ Sample dataset ready!")
            print("\n⚠️  NOTE: This is a DEMO dataset with dummy audio paths.")
            print("For real training, you'll need the actual Speechocean762 dataset.")
            return True
        
        elif choice == "1":
            # Download real dataset
            if not self.download_instructions():
                return False
            
            if not self.extract_dataset():
                return False
            
            df = self.organize_real_dataset()
            
            if df is not None:
                print("\n✓ Real dataset ready!")
                return True
            else:
                print("\n✗ Dataset organization failed")
                return False
        
        else:
            print("Invalid choice!")
            return False

if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.run()




# ============================================================================
# FILE 5: simple_train.py
# Save this in: mispronunciation-detection/simple_train.py
# A simplified training script for beginners
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from config import Config

# Simple Dataset class
class SimpleDataset(Dataset):
    def __init__(self, data, split='train'):
        self.data = [d for d in data if d['split'] == split or 
                     (split == 'val' and d['split'] == 'test')]
        print(f"{split.upper()}: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'audio': sample['audio'],
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'score': torch.tensor(sample['overall_score'], dtype=torch.float)
        }

# Simple Model (without Wav2Vec2 for faster testing)
class SimpleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Simple CNN for audio processing
        self.conv1 = nn.Conv1d(1, 32, kernel_size=80, stride=16)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, config.NUM_LABELS)
        )
        
        # Score regression
        self.regressor = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, audio):
        # Add channel dimension
        x = audio.unsqueeze(1)  # [B, 1, T]
        
        # Convolutions
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Pool
        x = self.pool(x).squeeze(-1)  # [B, 128]
        
        # Outputs
        logits = self.classifier(x)
        scores = self.regressor(x)
        
        return logits, scores

# Simple Trainer
class SimpleTrainer:
    def __init__(self, config=None):
        self.config = config or Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Model
        self.model = SimpleModel(self.config).to(self.device)
        print(f"Model created with {self.count_parameters()} parameters")
        
        # Loss and optimizer
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        self.best_val_loss = float('inf')
    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            audio = batch['audio'].to(self.device)
            labels = batch['label'].to(self.device)
            scores = batch['score'].to(self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            logits, pred_scores = self.model(audio)
            
            loss_cls = self.classification_loss(logits, labels)
            loss_reg = self.regression_loss(pred_scores, scores)
            loss = loss_cls + 0.3 * loss_reg
            
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
        
        return total_loss / len(train_loader), 100 * correct / total
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                audio = batch['audio'].to(self.device)
                labels = batch['label'].to(self.device)
                scores = batch['score'].to(self.device).unsqueeze(1)
                
                logits, pred_scores = self.model(audio)
                
                loss_cls = self.classification_loss(logits, labels)
                loss_reg = self.regression_loss(pred_scores, scores)
                loss = loss_cls + 0.3 * loss_reg
                
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader), 100 * correct / total
    
    def train(self, train_loader, val_loader):
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print("-" * 70)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model()
                print("✓ Best model saved!")
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save history
        history_file = self.config.RESULTS_DIR / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"✓ Training history saved to: {history_file}")
    
    def save_model(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config)
        }
        save_path = self.config.BEST_MODEL_DIR / "best_model.pt"
        torch.save(checkpoint, save_path)

def main():
    print("\n" + "="*70)
    print("SIMPLE TRAINING SCRIPT FOR BEGINNERS")
    print("="*70)
    
    config = Config()
    
    # Load preprocessed data
    data_file = config.PROCESSED_DATA_DIR / "preprocessed_data.pt"
    
    if not data_file.exists():
        print(f"\n✗ Preprocessed data not found: {data_file}")
        print("Please run preprocessing.py first!")
        return
    
    print(f"\nLoading data from: {data_file}")
    all_data = torch.load(data_file)
    print(f"✓ Loaded {len(all_data)} samples")
    
    # Create datasets
    train_data = [d for d in all_data if d['split'] == 'train']
    val_data = train_data[:int(len(train_data) * 0.2)]  # Use 20% for validation
    train_data = train_data[int(len(train_data) * 0.2):]
    
    train_dataset = SimpleDataset(train_data, split='train')
    val_dataset = SimpleDataset(val_data, split='train')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)
    
    # Train
    trainer = SimpleTrainer(config)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()


print("\n" + "="*70)
print("ALL FILES CREATED!")
print("="*70)
print("\nYou should now have these files:")
print("1. config.py")
print("2. download_dataset.py")
print("3. preprocessing.py")
print("4. simple_train.py")
print("\nNext: Follow the step-by-step instructions in the guide!")