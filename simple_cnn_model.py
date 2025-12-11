import torch.nn as nn
import torch
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