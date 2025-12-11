# ============================================================================
# FILE 9: inference.py
# Inference script for making predictions on new audio
# ============================================================================

import torch
import torchaudio
from pathlib import Path

from config import Config
from model import MispronunciationDetector
from preprocessing import AudioPreprocessor

class MispronunciationInference:
    """
    Inference class for making predictions on new audio files.
    """
    
    def __init__(self, model_path=None, config=None):
        self.config = config or Config()
        self.device = torch.device(
            self.config.DEVICE if torch.cuda.is_available() else "cpu"
        )
        
        # Load model
        self.model = MispronunciationDetector(self.config).to(self.device)
        
        if model_path is None:
            model_path = self.config.BEST_MODEL_DIR / "best_model.pt"
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(self.config)
        
        print("✓ Inference engine ready")
    
    def predict_file(self, audio_path):
        """
        Make prediction on audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess audio
        audio = self.preprocessor.preprocess_audio(audio_path, augment=False)
        
        if audio is None:
            return {'error': 'Failed to load audio'}
        
        # Move to device and add batch dimension
        audio = audio.to(self.device).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            logits, score = self.model(audio)
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1)
        
        class_names = ['Poor', 'Acceptable', 'Good']
        
        result = {
            'file': str(audio_path),
            'predicted_class': predicted_class.item(),
            'predicted_label': class_names[predicted_class.item()],
            'confidence': probs[0][predicted_class].item(),
            'probabilities': {
                'Poor': probs[0][0].item(),
                'Acceptable': probs[0][1].item(),
                'Good': probs[0][2].item()
            },
            'pronunciation_score': score.item()
        }
        
        return result
    
    def print_prediction(self, result):
        """Print prediction in nice format"""
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        print("\n" + "="*60)
        print("PRONUNCIATION ANALYSIS")
        print("="*60)
        print(f"\nFile: {result['file']}")
        print(f"\nPrediction: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"Pronunciation Score: {result['pronunciation_score']:.2f}/5.0")
        
        print("\nDetailed Probabilities:")
        for label, prob in result['probabilities'].items():
            bar = "█" * int(prob * 50)
            print(f"  {label:12s}: {bar} {prob*100:.1f}%")

def main():
    """Example usage"""
    # Initialize inference engine
    inference = MispronunciationInference()
    
    # Example: predict on a single file
    # audio_path = "path/to/your/audio.wav"
    # result = inference.predict_file(audio_path)
    # inference.print_prediction(result)
    
    print("\nInference engine initialized!")
    print("Usage example:")
    print("  result = inference.predict_file('audio.wav')")
    print("  inference.print_prediction(result)")

if __name__ == "__main__":
    main()

print("\n" + "="*70)
print("ALL PROJECT FILES CREATED SUCCESSFULLY!")
print("="*70)
print("\nFiles created:")
print("1. requirements.txt - Dependencies")
print("2. config.py - Configuration")
print("3. download_dataset.py - Dataset downloader")
print("4. preprocessing.py - Audio preprocessing")
print("5. dataset.py - PyTorch dataset")
print("6. model.py - Wav2Vec2 model")
print("7. train.py - Training script")
print("8. evaluate.py - Evaluation script")
print("9. inference.py - Inference script")