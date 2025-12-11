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
    
    # Replace your current organize_dataset method with this:

    def organize_dataset(self):
        """Organize the Speechocean762 dataset by using Kaldi directories for splitting."""
        print("\n" + "="*70)
        print("ORGANIZING SPEECHOCEAN762 DATASET (FIXED SPLITTING)")
        print("="*70)
        
        # 1. Find dataset location
        dataset_root, wave_dir, train_dir, test_dir = self.find_dataset_location()
        if not dataset_root: return None
        
        resource_dir = dataset_root / "resource"
        
        # 2. Load ALL scores from the single JSON file
        print(f"\n📄 Loading ALL scores from: {resource_dir / 'scores.json'}")
        all_scores = self.load_scores_from_json(resource_dir / "scores.json")
        if not all_scores:
            print("\n✗ Critical: No scores loaded!")
            return None
        print(f"✓ Loaded {len(all_scores)} total utterances with scores.")
        
        # 3. Determine the official train and test UTTERANCE IDs using wav.scp
        print("\n🔍 Determining official splits via wav.scp...")
        train_utt_ids = self.load_utt_ids_from_wav_scp(train_dir)
        test_utt_ids = self.load_utt_ids_from_wav_scp(test_dir)
        
        print(f"  Train set size (from wav.scp): {len(train_utt_ids)}")
        print(f"  Test set size (from wav.scp): {len(test_utt_ids)}")
        
        # 4. Find ALL audio files
        print(f"\n🔍 Searching for audio files in WAVE/...")
        wav_files = list(wave_dir.rglob("*.wav")) + list(wave_dir.rglob("*.WAV"))
        print(f"✓ Found {len(wav_files)} total audio files")
        
        if not wav_files: return None
        
        # 5. Match audio files with scores and assign split
        print("\n📊 Matching audio files, scores, and assigning splits...")
        data_entries = []
        matched_train = 0
        matched_test = 0
        unmatched_score = 0
        unmatched_split = 0
        
        for wav_file in tqdm(wav_files[:500], desc="Processing"): 
            utt_id = wav_file.stem
            
            # 5a. Determine Split based on wav.scp lists (CRITICAL FIX)
            split = 'unknown'
            if utt_id in train_utt_ids:
                split = 'train'
                matched_train += 1
            elif utt_id in test_utt_ids:
                split = 'test'
                matched_test += 1
            else:
                unmatched_split += 1
                # If not in train or test wav.scp, we skip it (or leave it as 'unknown')

            # 5b. Get Score
            score = all_scores.get(utt_id)
            if score is None:
                score = 3.0 # Default score if not found
                unmatched_score += 1
                
            # Skip if we didn't find a split OR a score (keeping the 500 limit logic)
            if split == 'unknown' or score is None: 
                continue
            
            # 5c. Determine label based on score
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
                'label': label
            })
        
        # ... (Rest of the function for saving DataFrame to CSV)
        df = pd.DataFrame(data_entries)
        
        # Save to CSV
        output_file = self.config.PROCESSED_DATA_DIR / "dataset_info.csv"
        df.to_csv(output_file, index=False)
        
        # Print matching statistics
        # ... (The rest of your summary print statements)
        
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
    # Add this helper function inside the DatasetDownloader class:

    def load_utt_ids_from_wav_scp(self, kaldi_dir):
        """Load utterance IDs from the Kaldi wav.scp file."""
        wav_scp_path = kaldi_dir / "wav.scp"
        utt_ids = set()
        if wav_scp_path.exists():
            try:
                with open(wav_scp_path, 'r') as f:
                    for line in f:
                        parts = line.split()
                        if parts:
                            # The first part of the line is the utterance ID
                            utt_ids.add(parts[0]) 
            except Exception as e:
                print(f"    Warning: Could not read {wav_scp_path}: {e}")
        return utt_ids

if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.run()