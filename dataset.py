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

# ============================================================================
# dataset.py - FULLY CORRECTED create_dataloaders function
# ============================================================================

def create_dataloaders(config=None):
    """
    Create train, validation, and test dataloaders from the separate .pt files.
    """
    config = config or Config()
    
    # 1. Define the correct separate file paths
    train_val_file = config.PROCESSED_DATA_DIR / "preprocessed_train_val_data.pt"
    test_file = config.PROCESSED_DATA_DIR / "preprocessed_test_data.pt"
    
    if not train_val_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            "Preprocessed data not found. Required files are missing. "
            f"Please run preprocessing.py first to create: {train_val_file.name} and {test_file.name}"
        )
    
    print("Loading preprocessed data from separate files...")
    
    # 2. Load the TRULY separated train/val and dedicated test data
    train_val_data = torch.load(train_val_file)
    test_data = torch.load(test_file)
    
    # Now, split the train_val_data into actual TRAIN and VALIDATION
    from sklearn.model_selection import train_test_split
    
    if len(train_val_data) > 10:
        # Split the training block into training and validation sets
        train_labels = [d['label'] for d in train_val_data]
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=0.15,  # 15% of the initial block for validation
            random_state=config.SEED,
            stratify=train_labels
        )
    else:
        # Fallback (should not happen with your data)
        train_data = train_val_data
        val_data = test_data 
    
    # 3. Create datasets (Test data is now correctly populated!)
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Test:  {len(test_data)}")
    
    # Assuming MispronunciationDataset is defined elsewhere
    train_dataset = MispronunciationDataset(train_data, config=config)
    val_dataset = MispronunciationDataset(val_data, config=config)
    test_dataset = MispronunciationDataset(test_data, config=config)
    
    # 4. Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
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