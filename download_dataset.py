# download_dataset.py - FULLY CORRECTED VERSION

import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from config import Config
import json
class DatasetDownloader:
    """Organize the already-extracted Speechocean762 dataset"""
    
    def __init__(self):
        self.config = Config()
        self.config.create_directories()
    
    def find_dataset_location(self):
        """Find where the speechocean762 folder is"""
        print("\n" + "="*70)
        print("LOCATING SPEECHOCEAN762 DATASET")
        print("="*70)
        
        # Your dataset location
        dataset_root = self.config.RAW_DATA_DIR / "speechocean762"
        
        if not dataset_root.exists():
            print(f"\n✗ Dataset not found at: {dataset_root}")
            return None, None, None, None
        
        print(f"\n✓ Found dataset at: {dataset_root}")
        
        # Check for key folders
        train_dir = dataset_root / "train"
        test_dir = dataset_root / "test"
        wave_dir = dataset_root / "WAVE"
        
        print(f"\n📁 Checking structure...")
        print(f"  {'✓' if train_dir.exists() else '✗'} train/ folder")
        print(f"  {'✓' if test_dir.exists() else '✗'} test/ folder")
        print(f"  {'✓' if wave_dir.exists() else '✗'} WAVE/ folder")
        
        return dataset_root, wave_dir, train_dir, test_dir
    ###############################################
    def load_scores_from_json(self, path):
        """
        Load Speechocean762 scores.json (dict format).
        Extract a single score per utterance using 'total'.
        """
        import json

        with open(path, "r") as f:
            data = json.load(f)

        # Case 1: correct format (dict of objects)
        if isinstance(data, dict):
            scores = {}
            for utt_id, info in data.items():
                if isinstance(info, dict) and "total" in info:
                    scores[utt_id] = float(info["total"])
                else:
                    # fallback for weird cases
                    scores[utt_id] = None
            return scores

        # Case 2: fallback for list format (not used here)
        scores = {}
        for item in data:
            utt_id = item["UtteranceID"]
            score = item["Score"]
            scores[utt_id] = score
        return scores



    ############################################
    
    def load_scores_from_text(self, text_file):
        """Load pronunciation scores from a single text file"""
        scores = {}
        
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Format: utterance_id accuracy completeness fluency prosody total
                    parts = line.split()
                    if len(parts) >= 2:
                        utt_id = parts[0]
                        # Try to get the overall score (usually last number)
                        try:
                            score = float(parts[-1])
                            scores[utt_id] = score
                        except:
                            pass
        except Exception as e:
            print(f"    Warning: Could not read {text_file.name}: {e}")
        
        return scores
    
    def load_all_scores_from_directory(self, score_dir, split_name):
        """Load score files from a directory (Kaldi format)"""
        all_scores = {}
        
        if not score_dir.exists():
            print(f"  ✗ Directory not found: {score_dir}")
            return all_scores
        
        # Check for Kaldi-style 'text' file (no extension)
        text_file = score_dir / "text"
        
        if text_file.exists():
            print(f"\n📄 Loading {split_name} scores from: {text_file}")
            all_scores = self.load_scores_from_text(text_file)
            print(f"  ✓ Loaded {len(all_scores)} {split_name} scores")
        else:
            # Fall back to looking for .txt files
            txt_files = list(score_dir.rglob("*.txt"))
            
            if txt_files:
                print(f"\n📄 Loading {split_name} scores from: {score_dir}")
                print(f"  Found {len(txt_files)} text files")
                
                for txt_file in txt_files:
                    scores = self.load_scores_from_text(txt_file)
                    all_scores.update(scores)
                    if scores:
                        print(f"    ✓ {txt_file.name}: {len(scores)} scores")
                
                print(f"  ✓ Total {split_name} scores loaded: {len(all_scores)}")
            else:
                print(f"  ⚠️  No score files found in {score_dir}")
        
        return all_scores
    
    def organize_dataset(self):
        """Organize the Speechocean762 dataset"""
        print("\n" + "="*70)
        print("ORGANIZING SPEECHOCEAN762 DATASET")
        print("="*70)
        
        # Find dataset
        dataset_root, wave_dir, train_dir, test_dir = self.find_dataset_location()
        
        if not dataset_root:
            print("\n✗ Cannot proceed without dataset")
            return None
        
        # Check if WAVE directory has the audio files (RECURSIVE search)
        if wave_dir and wave_dir.exists():
            print(f"\n🔍 Searching for audio files in WAVE/ (including subdirectories)...")
            # Search for both .wav and .WAV (case-insensitive)
            wav_files = list(wave_dir.rglob("*.wav")) + list(wave_dir.rglob("*.WAV"))
            print(f"✓ Found {len(wav_files)} audio files")
            
            # Show sample of structure
            if wav_files:
                print(f"\n  Example audio file locations:")
                for wav in wav_files[:3]:
                    relative_path = wav.relative_to(wave_dir)
                    print(f"    • {relative_path}")
                if len(wav_files) > 3:
                    print(f"    • ... and {len(wav_files) - 3} more")
        else:
            print("\n✗ WAVE directory not found")
            return None
        
        if not wav_files:
            print("\n✗ No WAV files found!")
            return None
        
        # Load scores from train and test directories (RECURSIVE)
        # train_scores = self.load_all_scores_from_directory(train_dir, "training")
        # test_scores = self.load_all_scores_from_directory(test_dir, "test")
        resource_dir = dataset_root / "resource"

        train_scores = self.load_scores_from_json(resource_dir / "scores.json")
        test_scores  = train_scores   # SO762 does not provide separate test scores

        
        if not train_scores and not test_scores:
            print("\n⚠️  Warning: No scores found! Using default scores.")
        
        # Now match audio files with scores
        print("\n📊 Matching audio files with scores...")
        data_entries = []
        
        matched_train = 0
        matched_test = 0
        unmatched = 0
        
        # Limit to first 500 files for faster processing
        # Remove [:500] to use full dataset
        for wav_file in tqdm(wav_files[:500], desc="Processing"):
            # Get utterance ID from filename (without .wav)
            utt_id = wav_file.stem
            
            # Check if this is in train or test
            score = None
            split = None
            
            if utt_id in train_scores:
                score = train_scores[utt_id]
                split = 'train'
                matched_train += 1
            elif utt_id in test_scores:
                score = test_scores[utt_id]
                split = 'test'
                matched_test += 1
            else:
                # Default if not found
                score = 3.0
                split = 'train'  # Assume train by default
                unmatched += 1
            
            # Determine label based on score
            if score >= 4.0:
                label = 2  # Good
            elif score >= 2.5:
                label = 1  # Acceptable
            else:
                label = 0  # Poor
            
            data_entries.append({
                'audio_path': str(wav_file),
                'utterance_id': utt_id,
                'split': split,
                'overall_score': score,
                'label': label,
                'accuracy': score,
                'completeness': score,
                'fluency': score,
                'prosody': score
            })
        
        if not data_entries:
            print("\n✗ No data entries created")
            return None
        
        # Print matching statistics
        print(f"\n📈 Matching Statistics:")
        print(f"  ✓ Matched with training scores: {matched_train}")
        print(f"  ✓ Matched with test scores: {matched_test}")
        print(f"  ⚠️  Used default scores: {unmatched}")
        
        # Create DataFrame
        df = pd.DataFrame(data_entries)
        
        # Save to CSV
        output_file = self.config.PROCESSED_DATA_DIR / "dataset_info.csv"
        df.to_csv(output_file, index=False)
        
        # Print statistics
        print(f"\n{'='*70}")
        print("DATASET SUMMARY")
        print(f"{'='*70}")
        print(f"\n✓ Total samples: {len(df)}")
        print(f"\n📊 Split distribution:")
        print(df['split'].value_counts().to_string())
        print(f"\n🎯 Label distribution:")
        label_names = {0: 'Poor', 1: 'Acceptable', 2: 'Good'}
        for label, count in df['label'].value_counts().sort_index().items():
            print(f"  {label_names[label]:12s} ({label}): {count}")
        
        print(f"\n📈 Score statistics:")
        print(f"  Mean: {df['overall_score'].mean():.2f}")
        print(f"  Min:  {df['overall_score'].min():.2f}")
        print(f"  Max:  {df['overall_score'].max():.2f}")
        
        print(f"\n✓ Saved to: {output_file}")
        print(f"{'='*70}\n")
        
        return df
    
    def run(self):
        """Main function to run dataset organization"""
        result = self.organize_dataset()
        
        if result is not None:
            print("\n✅ SUCCESS! Dataset is ready.")
            print("\nNext step: Run 'python preprocessing.py'")
            return True
        else:
            print("\n❌ FAILED! Please check the errors above.")
            return False

if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.run()