import librosa
import numpy as np
import soundfile as sf
from scipy.signal import wiener
import os
from pathlib import Path
import torchaudio
class MispronunciationPreprocessor:
    """
    Preprocessing pipeline for English mispronunciation detection
    """
    
    def __init__(self, target_sr=16000, n_mfcc=13, n_mels=40):
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        
    def load_audio(self, audio_path):
        """Load and resample audio file"""
        audio, sr = librosa.load(audio_path, sr=self.target_sr)
        return audio, sr
    
    def remove_silence(self, audio, top_db=20):
        """Remove leading and trailing silence"""
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return audio_trimmed
    
    def normalize_audio(self, audio):
        """Normalize audio to [-1, 1] range"""
        return librosa.util.normalize(audio)
    
    def reduce_noise(self, audio):
        """Basic noise reduction using Wiener filter"""
        # Simple noise reduction - can be improved with more sophisticated methods
        return wiener(audio)
    
    def extract_mfcc(self, audio, sr):
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=self.n_mfcc,
            n_fft=2048,
            hop_length=512
        )
        # Add delta and delta-delta features
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack features
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        return features
    
    def extract_mel_spectrogram(self, audio, sr):
        """Extract mel spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.n_mels,
            n_fft=2048,
            hop_length=512
        )
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_pitch_features(self, audio, sr):
        """Extract pitch (F0) features"""
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        return f0, voiced_flag, voiced_probs
    
    def extract_formants(self, audio, sr):
        """Extract formant frequencies (simplified)"""
        # This is a basic implementation
        # For production, use more robust formant extraction
        spectrum = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        
        # Find peaks in spectrum (simplified formant estimation)
        magnitude = np.abs(spectrum)
        peaks = []
        for i in range(1, len(magnitude)-1):
            if magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]:
                peaks.append((freqs[i], magnitude[i]))
        
        # Sort by magnitude and take top formants
        peaks.sort(key=lambda x: x[1], reverse=True)
        formants = [p[0] for p in peaks[:4]]  # F1, F2, F3, F4
        
        return formants
    
    def preprocess_file(self, audio_path, extract_all=True):
        """
        Complete preprocessing pipeline for a single audio file
        
        Args:
            audio_path: Path to audio file
            extract_all: If True, extract all features; if False, only MFCC
            
        Returns:
            Dictionary containing preprocessed audio and features
        """
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Preprocessing steps
        audio = self.remove_silence(audio)
        audio = self.normalize_audio(audio)
        audio = self.reduce_noise(audio)
        
        # Feature extraction
        features = {
            'audio': audio,
            'sample_rate': sr,
            'duration': len(audio) / sr,
            'mfcc': self.extract_mfcc(audio, sr)
        }
        
        if extract_all:
            features['mel_spectrogram'] = self.extract_mel_spectrogram(audio, sr)
            f0, voiced_flag, voiced_probs = self.extract_pitch_features(audio, sr)
            features['pitch'] = f0
            features['voiced_flag'] = voiced_flag
            features['voiced_probs'] = voiced_probs
            features['formants'] = self.extract_formants(audio, sr)
        
        return features
    
    def preprocess_dataset(self, data_dir, output_dir, file_extension='.wav'):
        """
        Preprocess entire dataset
        
        Args:
            data_dir: Directory containing audio files
            output_dir: Directory to save preprocessed features
            file_extension: Audio file extension to process
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        audio_files = list(Path(data_dir).rglob(f'*{file_extension}'))
        print(f"Found {len(audio_files)} audio files")
        
        for i, audio_path in enumerate(audio_files):
            try:
                print(f"Processing {i+1}/{len(audio_files)}: {audio_path.name}")
                
                # Preprocess
                features = self.preprocess_file(str(audio_path))
                
                # Save features
                output_path = Path(output_dir) / f"{audio_path.stem}.npz"
                np.savez_compressed(
                    output_path,
                    mfcc=features['mfcc'],
                    mel_spectrogram=features['mel_spectrogram'],
                    pitch=features['pitch'],
                    voiced_flag=features['voiced_flag'],
                    duration=features['duration']
                )
                
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
        
        print("Preprocessing complete!")


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = MispronunciationPreprocessor(
        target_sr=16000,
        n_mfcc=13,
        n_mels=40
    )
    
    # Example: Preprocess single file
    # features = preprocessor.preprocess_file('path/to/audio.wav')
    # print(f"MFCC shape: {features['mfcc'].shape}")
    
    # Example: Preprocess entire dataset
    # preprocessor.preprocess_dataset(
    #     data_dir='path/to/dataset',
    #     output_dir='path/to/output',
    #     file_extension='.wav'
    # )
    
    print("Preprocessor initialized successfully!")
    print("Uncomment example usage lines to process your data")


# ============================================================================
# FILE 4: preprocessing.py
# Save this in: mispronunciation-detection/preprocessing.py
# ============================================================================

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import Config

class AudioPreprocessor:
    """Preprocess audio for the model"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.target_sr = self.config.SAMPLE_RATE
        self.max_length = self.config.MAX_AUDIO_LENGTH
    
    def create_dummy_audio(self):
        """Create dummy audio data for testing (when real audio not available)"""
        # Create random audio-like data
        duration = self.max_length
        num_samples = int(duration * self.target_sr)
        audio = torch.randn(num_samples) * 0.1  # Random noise
        return audio
    
    # def preprocess_audio(self, audio_path, augment=False):
    #     """
    #     Preprocess a single audio file.
    #     For testing purposes, creates dummy audio if file doesn't exist.
    #     """
    #     try:
    #         # Try to load real audio
    #         import torchaudio
      
    #         audio, sr = torchaudio.load(audio_path)
            
    #         # Convert to mono
    #         if audio.shape[0] > 1:
    #             audio = torch.mean(audio, dim=0)
    #         else:
    #             audio = audio.squeeze()
            
    #         # Resample if needed
    #         if sr != self.target_sr:
    #             resampler = torchaudio.transforms.Resample(sr, self.target_sr)
    #             audio = resampler(audio)
            
    #         # Normalize
    #         max_val = torch.abs(audio).max()
    #         if max_val > 0:
    #             audio = audio / max_val
            
    #         # Pad or truncate
    #         target_length = int(self.max_length * self.target_sr)
    #         if audio.shape[0] < target_length:
    #             padding = target_length - audio.shape[0]
    #             audio = torch.nn.functional.pad(audio, (0, padding))
    #         else:
    #             audio = audio[:target_length]
            
    #         return audio
            
    #     except Exception as e:
    #         # If loading fails, create dummy audio
    #         print(f"  Creating dummy audio (file not found: {audio_path})")
    #         return self.create_dummy_audio()

    ########################################
    def preprocess_audio(self, audio_path, augment=False):
        """
        Preprocess a single audio file.
        Includes a robust check for case-sensitivity issues on mounted drives.
        """
        import torchaudio
        
        # 1. Check the path case (handles .WAV vs .wav mismatch)
        path_to_try = Path(audio_path)
        
        # Check if the exact path exists
        if not path_to_try.exists():
            # If not, try checking the alternative case (e.g., .WAV -> .wav)
            if path_to_try.suffix.upper() == '.WAV':
                path_to_try_lower = path_to_try.with_suffix('.wav')
                if path_to_try_lower.exists():
                    audio_path = str(path_to_try_lower)
            # You could add similar logic for uppercase directories if needed
            
        try:
            # Try to load the audio using the verified path
            audio, sr = torchaudio.load(audio_path)
            
            # Convert to mono
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0)
            else:
                audio = audio.squeeze()
            
            # Resample if needed
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                audio = resampler(audio)
            
            # Normalize and Pad/Truncate (as before)
            max_val = torch.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val
            
            target_length = int(self.max_length * self.target_sr)
            if audio.shape[0] < target_length:
                padding = target_length - audio.shape[0]
                audio = torch.nn.functional.pad(audio, (0, padding))
            else:
                audio = audio[:target_length]
            
            return audio
            
        except Exception as e:
            # If loading fails even after path correction, create dummy audio
            print(f"  Creating dummy audio (file not found or load error: {audio_path}) - Error: {e}")
            return self.create_dummy_audio()
    ##########################################
    
    # def preprocess_dataset(self, dataset_csv):
    #     """Preprocess entire dataset"""
    #     print("\n" + "="*70)
    #     print("PREPROCESSING DATASET")
    #     print("="*70)
        
    #     # Load dataset info
    #     print(f"\nLoading dataset from: {dataset_csv}")
    #     df = pd.read_csv(dataset_csv)
    #     print(f"✓ Found {len(df)} samples")
        
    #     print("\nPreprocessing audio files...")
    #     processed_data = []
        
    #     for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
    #         audio = self.preprocess_audio(row['audio_path'], augment=False)
            
    #         if audio is not None:
    #             processed_data.append({
    #                 'audio': audio,
    #                 'label': int(row['label']),
    #                 'split': row['split'],
    #                 'overall_score': float(row['overall_score'])
    #             })
        
    #     # Save processed data
    #     output_file = self.config.PROCESSED_DATA_DIR / "preprocessed_data.pt"
    #     torch.save(processed_data, output_file)
        
    #     print(f"\n✓ Preprocessing complete!")
    #     print(f"✓ Processed {len(processed_data)} samples")
    #     print(f"✓ Saved to: {output_file}")
        
    #     # Show statistics
    #     labels = [d['label'] for d in processed_data]
    #     print(f"\n📊 Label distribution:")
    #     print(f"  Poor (0): {labels.count(0)}")
    #     print(f"  Acceptable (1): {labels.count(1)}")
    #     print(f"  Good (2): {labels.count(2)}")
        
    #     return processed_data
    ##new code #####
    # ============================================================================

    def preprocess_dataset(self, dataset_csv):
        """Preprocess entire dataset and save train/val and test data separately."""
        print("\n" + "="*70)
        print("PREPROCESSING DATASET")
        print("="*70)
        
        # Load dataset info
        print(f"\nLoading dataset from: {dataset_csv}")
        df = pd.read_csv(dataset_csv)
        print(f"✓ Found {len(df)} samples total.")
        
        print("\nPreprocessing audio files...")
        
        # Temporary list to hold all processed data
        all_processed_data = [] 
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            audio = self.preprocess_audio(row['audio_path'], augment=False)
            
            if audio is not None:
                all_processed_data.append({
                    'audio': audio,
                    'label': int(row['label']),
                    'split': row['split'], # Keep the split information
                    'overall_score': float(row['overall_score'])
                })
        
        # --- CRITICAL NEW LOGIC: SEPARATE AND SAVE ---
        
        # 1. Separate data into train/val and test based on the 'split' column
        train_val_data = [d for d in all_processed_data if d['split'] in ['train', 'val']]
        test_data = [d for d in all_processed_data if d['split'] == 'test']
        
        # 2. Save the splits to separate files
        train_val_output_file = self.config.PROCESSED_DATA_DIR / "preprocessed_train_val_data.pt"
        test_output_file = self.config.PROCESSED_DATA_DIR / "preprocessed_test_data.pt"
        
        torch.save(train_val_data, train_val_output_file)
        torch.save(test_data, test_output_file)
        
        print(f"\n✓ Preprocessing complete!")
        print(f"✓ Saved Train/Validation samples ({len(train_val_data)}) to: {train_val_output_file}")
        print(f"✓ Saved Test samples ({len(test_data)}) to: {test_output_file}")
        
        # Show statistics for all processed data (for context)
        labels = [d['label'] for d in all_processed_data]
        print(f"\n📊 Overall Label distribution:")
        print(f"  Poor (0): {labels.count(0)}")
        print(f"  Acceptable (1): {labels.count(1)}")
        print(f"  Good (2): {labels.count(2)}")
        
        return all_processed_data # Returning all data is fine for main function, but we use the files.
    ####################################################


if __name__ == "__main__":
    config = Config()
    preprocessor = AudioPreprocessor(config)
    
    dataset_csv = config.PROCESSED_DATA_DIR / "dataset_info.csv"
    
    if dataset_csv.exists():
        preprocessor.preprocess_dataset(dataset_csv)
    else:
        print(f"✗ Dataset CSV not found: {dataset_csv}")
        print("Please run download_dataset.py first!")
