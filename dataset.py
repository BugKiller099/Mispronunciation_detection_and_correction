# ============================================================================
# FILE 5: dataset.py
# PyTorch Dataset class
# ============================================================================

# import torch
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# from config import Config
# from preprocessing import AudioPreprocessor

# class MispronunciationDataset(Dataset):
#     """
#     PyTorch Dataset for mispronunciation detection.
    
#     Handles loading preprocessed audio and labels for training/evaluation.
#     """
    
#     def __init__(self, data, split='train', config=None):
#         """
#         Args:
#             data: List of preprocessed data dictionaries
#             split: 'train', 'val', or 'test'
#             config: Configuration object
#         """
#         self.config = config or Config()
#         self.split = split
        
#         # Filter data by split
#         self.data = [d for d in data if d['split'] == split or 
#                      (split == 'val' and d['split'] == 'test')]
        
#         print(f"{split.upper()} set: {len(self.data)} samples")
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         """
#         Get a single sample.
        
#         Returns:
#             Dictionary with 'audio' and 'label' tensors
#         """
#         sample = self.data[idx]
        
#         return {
#             'audio': sample['audio'],
#             'label': torch.tensor(sample['label'], dtype=torch.long),
#             'score': torch.tensor(sample['overall_score'], dtype=torch.float)
#         }

# def create_dataloaders(config=None):
#     """
#     Create train, validation, and test dataloaders.
    
#     Returns:
#         Tuple of (train_loader, val_loader, test_loader)
#     """
#     config = config or Config()
    
#     # Load preprocessed data
#     data_file = config.PROCESSED_DATA_DIR / "preprocessed_data.pt"
    
#     if not data_file.exists():
#         raise FileNotFoundError(
#             f"Preprocessed data not found at {data_file}. "
#             "Please run preprocessing.py first."
#         )
    
#     print("Loading preprocessed data...")
#     all_data = torch.load(data_file)
    
#     # Split data
#     # Manually split if needed, or use existing split column
#     train_data = [d for d in all_data if d['split'] == 'train']
#     test_data = [d for d in all_data if d['split'] == 'test']
    
#     # Further split train into train and val
#     from sklearn.model_selection import train_test_split
#     train_data, val_data = train_test_split(
#         train_data, 
#         test_size=0.1, 
#         random_state=config.SEED,
#         stratify=[d['label'] for d in train_data]
#     )
    
#     # Create datasets
#     train_dataset = MispronunciationDataset(train_data, split='train', config=config)
#     val_dataset = MispronunciationDataset(val_data, split='val', config=config)
#     test_dataset = MispronunciationDataset(test_data, split='test', config=config)
    
#     # Create dataloaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config.BATCH_SIZE,
#         shuffle=True,
#         num_workers=config.NUM_WORKERS,
#         pin_memory=config.PIN_MEMORY
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config.EVAL_BATCH_SIZE,
#         shuffle=False,
#         num_workers=config.NUM_WORKERS,
#         pin_memory=config.PIN_MEMORY
#     )
    
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=config.EVAL_BATCH_SIZE,
#         shuffle=False,
#         num_workers=config.NUM_WORKERS,
#         pin_memory=config.PIN_MEMORY
#     )
    
#     print("✓ Dataloaders created successfully")
    
#     return train_loader, val_loader, test_loader


# dataset.py - FIXED VERSION

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import Config

class MispronunciationDataset(Dataset):
    """PyTorch Dataset for mispronunciation detection"""
    
    def __init__(self, data, config=None):
        """
        Args:
            data: List of preprocessed data dictionaries
            config: Configuration object
        """
        self.config = config or Config()
        self.data = data
        
        print(f"Dataset: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        sample = self.data[idx]
        
        return {
            'audio': sample['audio'],
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'score': torch.tensor(sample['overall_score'], dtype=torch.float)
        }

def create_dataloaders(config=None):
    """
    Create train, validation, and test dataloaders.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    config = config or Config()
    
    # Load preprocessed data
    data_file = config.PROCESSED_DATA_DIR / "preprocessed_data.pt"
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {data_file}. "
            "Please run preprocessing.py first."
        )
    
    print("Loading preprocessed data...")
    all_data = torch.load(data_file)
    print(f"✓ Loaded {len(all_data)} samples")
    
    # Split by existing 'split' field if available
    train_data = [d for d in all_data if d.get('split') == 'train']
    test_data = [d for d in all_data if d.get('split') == 'test']
    
    # If no train/test split exists, create one
    if not train_data:
        print("No train/test split found. Creating 80/20 split...")
        labels = [d['label'] for d in all_data]
        train_data, test_data = train_test_split(
            all_data,
            test_size=0.2,
            random_state=config.SEED,
            stratify=labels
        )
    
    # Further split train into train and validation
    if len(train_data) > 10:  # Only split if we have enough data
        train_labels = [d['label'] for d in train_data]
        train_data, val_data = train_test_split(
            train_data,
            test_size=0.15,  # 15% of training for validation
            random_state=config.SEED,
            stratify=train_labels
        )
    else:
        val_data = test_data  # Use test as validation if too little data
    
    # Create datasets
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Test:  {len(test_data)}")
    
    train_dataset = MispronunciationDataset(train_data, config=config)
    val_dataset = MispronunciationDataset(val_data, config=config)
    test_dataset = MispronunciationDataset(test_data, config=config)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False  # Disable pin_memory on CPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print("✓ Dataloaders created successfully\n")
    
    return train_loader, val_loader, test_loader