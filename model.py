# ============================================================================
# FILE 6: model.py
# Wav2Vec2-based mispronunciation detection model
# ============================================================================

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from config import Config

class MispronunciationDetector(nn.Module):
    """
    Mispronunciation detection model based on Wav2Vec2.
    
    Architecture:
    1. Wav2Vec2 encoder (pre-trained) - extracts speech representations
    2. Pooling layer - aggregates frame-level features
    3. Classification head - predicts pronunciation quality
    
    The model classifies pronunciation into 3 categories:
    - 0: Poor
    - 1: Acceptable  
    - 2: Good
    """
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or Config()
        
        # Load pre-trained Wav2Vec2 model
        print(f"Loading pre-trained model: {self.config.PRETRAINED_MODEL}")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            self.config.PRETRAINED_MODEL
        )
        
        # Freeze some layers initially (optional - can fine-tune later)
        # Uncomment to freeze feature extractor
        # for param in self.wav2vec2.feature_extractor.parameters():
        #     param.requires_grad = False
        
        # Pooling layer - average pool over time dimension
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.config.HIDDEN_SIZE, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.config.NUM_LABELS)
        )
        
        # Additional regression head for continuous score prediction
        self.score_regressor = nn.Sequential(
            nn.Linear(self.config.HIDDEN_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, audio, return_features=False):
        """
        Forward pass.
        
        Args:
            audio: Audio waveform tensor [batch_size, audio_length]
            return_features: If True, return intermediate features
            
        Returns:
            logits: Classification logits [batch_size, num_labels]
            scores: Continuous pronunciation scores [batch_size, 1]
        """
        # Extract features with Wav2Vec2
        outputs = self.wav2vec2(audio)
        features = outputs.last_hidden_state  # [batch, time, hidden_size]
        
        # Pool over time dimension
        # Transpose to [batch, hidden_size, time] for pooling
        features_t = features.transpose(1, 2)
        pooled = self.pooling(features_t).squeeze(-1)  # [batch, hidden_size]
        
        # Classification
        logits = self.classifier(pooled)
        
        # Regression for continuous score
        scores = self.score_regressor(pooled)
        
        if return_features:
            return logits, scores, pooled
        
        return logits, scores
    
    def predict(self, audio):
        """
        Make prediction on audio.
        
        Args:
            audio: Audio waveform tensor
            
        Returns:
            Dictionary with predictions
        """
        self.eval()
        with torch.no_grad():
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0)
            
            logits, scores = self.forward(audio)
            
            # Get class predictions
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1)
            
            return {
                'class': predicted_class.item(),
                'class_name': ['Poor', 'Acceptable', 'Good'][predicted_class.item()],
                'probabilities': probs[0].cpu().numpy(),
                'score': scores[0].item()
            }




