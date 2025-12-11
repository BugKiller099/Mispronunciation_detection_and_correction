"""
Quick Start Script for Mispronunciation Detection Project

This script runs the entire pipeline from start to finish:
1. Setup directories
2. Download dataset
3. Preprocess audio
4. Train model
5. Evaluate model

Usage:
    python run_pipeline.py --mode all
    python run_pipeline.py --mode train  # Train only
    python run_pipeline.py --mode eval   # Evaluate only
"""

import argparse
import sys
import os
from pathlib import Path

def print_banner(text):
    """Print a nice banner"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def check_requirements():
    """Check if all required packages are installed"""
    print_banner("CHECKING REQUIREMENTS")
    
    required_packages = [
        'torch', 'torchaudio', 'transformers', 'librosa', 
        'soundfile', 'pandas', 'numpy', 'sklearn', 'matplotlib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Please install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All requirements satisfied!")
    return True

def setup_directories():
    """Setup all required directories"""
    print_banner("SETTING UP DIRECTORIES")
    
    from config import Config
    Config.create_directories()
    print("✓ All directories created")
    return True

def download_dataset_step():
    """Download and organize dataset"""
    print_banner("DOWNLOADING DATASET")
    
    print("This step requires manual download of Speechocean762 dataset.")
    print("\nInstructions:")
    print("1. Visit: https://www.openslr.org/101/")
    print("2. Download 'train.tar.gz' and 'test.tar.gz'")
    print("3. Place files in: data/raw/")
    print()
    
    response = input("Have you downloaded and placed the files? (yes/no): ")
    
    if response.lower() != 'yes':
        print("\n⚠ Please download the dataset first and run again.")
        return False
    
    # Run download script
    try:
        from download_dataset import DatasetDownloader
        downloader = DatasetDownloader()
        success = downloader.download_dataset()
        
        if success:
            print("\n✓ Dataset organized successfully!")
            return True
        else:
            print("\n❌ Dataset organization failed")
            return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def preprocess_data():
    """Preprocess all audio data"""
    print_banner("PREPROCESSING DATA")
    
    from config import Config
    from preprocessing import AudioPreprocessor
    
    config = Config()
    dataset_csv = config.PROCESSED_DATA_DIR / "dataset_info.csv"
    
    if not dataset_csv.exists():
        print(f"❌ Dataset CSV not found at {dataset_csv}")
        print("Please run download step first.")
        return False
    
    try:
        preprocessor = AudioPreprocessor(config)
        preprocessor.preprocess_dataset(dataset_csv)
        print("\n✓ Preprocessing complete!")
        return True
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        return False

def train_model():
    """Train the model"""
    print_banner("TRAINING MODEL")
    
    from config import Config
    config = Config()
    
    preprocessed_data = config.PROCESSED_DATA_DIR / "preprocessed_data.pt"
    if not preprocessed_data.exists():
        print(f"❌ Preprocessed data not found at {preprocessed_data}")
        print("Please run preprocessing step first.")
        return False
    
    try:
        # Import and run training
        from train import main as train_main
        train_main()
        print("\n✓ Training complete!")
        return True
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False

def evaluate_model():
    """Evaluate the trained model"""
    print_banner("EVALUATING MODEL")
    
    from config import Config
    config = Config()
    
    best_model = config.BEST_MODEL_DIR / "best_model.pt"
    if not best_model.exists():
        print(f"❌ Trained model not found at {best_model}")
        print("Please run training step first.")
        return False
    
    try:
        from evaluate import main as eval_main
        eval_main()
        print("\n✓ Evaluation complete!")
        print(f"Results saved in: {config.RESULTS_DIR}")
        return True
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        return False

def run_inference_demo():
    """Run inference demo"""
    print_banner("INFERENCE DEMO")
    
    from config import Config
    from inference import MispronunciationInference
    
    config = Config()
    best_model = config.BEST_MODEL_DIR / "best_model.pt"
    
    if not best_model.exists():
        print(f"❌ Trained model not found at {best_model}")
        return False
    
    try:
        detector = MispronunciationInference()
        print("✓ Inference engine initialized")
        print("\nTo use:")
        print("  from inference import MispronunciationInference")
        print("  detector = MispronunciationInference()")
        print("  result = detector.predict_file('your_audio.wav')")
        print("  detector.print_prediction(result)")
        return True
    except Exception as e:
        print(f"❌ Inference initialization failed: {e}")
        return False

def run_all_steps():
    """Run the complete pipeline"""
    print_banner("STARTING COMPLETE PIPELINE")
    
    steps = [
        ("Requirements Check", check_requirements),
        ("Directory Setup", setup_directories),
        ("Dataset Download", download_dataset_step),
        ("Data Preprocessing", preprocess_data),
        ("Model Training", train_model),
        ("Model Evaluation", evaluate_model),
        ("Inference Setup", run_inference_demo)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*80}")
        print(f"STEP: {step_name}")
        print(f"{'='*80}\n")
        
        success = step_func()
        
        if not success:
            print(f"\n❌ Pipeline stopped at: {step_name}")
            print("Please resolve the issue and run again.")
            return False
        
        print(f"\n✓ {step_name} completed successfully!")
    
    print_banner("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print("\nYour mispronunciation detection system is ready to use!")
    print("\nNext steps:")
    print("1. Check results in: results/")
    print("2. Use inference.py to predict on new audio")
    print("3. Monitor training with: tensorboard --logdir logs/tensorboard")
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Mispronunciation Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --mode all          # Run complete pipeline
  python run_pipeline.py --mode setup        # Setup only
  python run_pipeline.py --mode download     # Download dataset
  python run_pipeline.py --mode preprocess   # Preprocess data
  python run_pipeline.py --mode train        # Train model
  python run_pipeline.py --mode eval         # Evaluate model
  python run_pipeline.py --mode inference    # Setup inference
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['all', 'setup', 'download', 'preprocess', 'train', 'eval', 'inference'],
        help='Which step to run'
    )
    
    args = parser.parse_args()
    
    # Print welcome message
    print("\n" + "="*80)
    print("  MISPRONUNCIATION DETECTION SYSTEM")
    print("  Automated Pipeline Script")
    print("="*80)
    
    # Run requested mode
    if args.mode == 'all':
        success = run_all_steps()
    elif args.mode == 'setup':
        success = setup_directories()
    elif args.mode == 'download':
        success = download_dataset_step()
    elif args.mode == 'preprocess':
        success = preprocess_data()
    elif args.mode == 'train':
        success = train_model()
    elif args.mode == 'eval':
        success = evaluate_model()
    elif args.mode == 'inference':
        success = run_inference_demo()
    
    # Exit with appropriate code
    if success:
        print("\n✅ SUCCESS!")
        sys.exit(0)
    else:
        print("\n❌ FAILED!")
        print("Check error messages above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()


# ============================================================================
# BONUS FILE: example_usage.py
# Example script showing how to use the trained model
# ============================================================================

"""
Example usage of the mispronunciation detection system
"""

def example_single_prediction():
    """Example: Predict on a single audio file"""
    from inference import MispronunciationInference
    
    print("="*60)
    print("Example 1: Single Audio Prediction")
    print("="*60)
    
    # Initialize detector
    detector = MispronunciationInference()
    
    # Predict on audio file
    audio_path = "path/to/your/audio.wav"
    result = detector.predict_file(audio_path)
    
    # Print results
    detector.print_prediction(result)
    
    # Access individual fields
    print(f"\nPredicted class: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"Score: {result['pronunciation_score']:.2f}/5.0")

def example_batch_prediction():
    """Example: Predict on multiple audio files"""
    from inference import MispronunciationInference
    from pathlib import Path
    import pandas as pd
    
    print("\n" + "="*60)
    print("Example 2: Batch Audio Prediction")
    print("="*60)
    
    # Initialize detector
    detector = MispronunciationInference()
    
    # Get all audio files from directory
    audio_dir = Path("path/to/audio/directory")
    audio_files = list(audio_dir.glob("*.wav"))
    
    # Process all files
    results = []
    for audio_file in audio_files:
        result = detector.predict_file(audio_file)
        results.append({
            'filename': audio_file.name,
            'prediction': result['predicted_label'],
            'confidence': result['confidence'],
            'score': result['pronunciation_score']
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    print(df)
    
    # Save results
    df.to_csv('batch_predictions.csv', index=False)
    print("\n✓ Results saved to batch_predictions.csv")

def example_real_time_scoring():
    """Example: Real-time pronunciation scoring"""
    from inference import MispronunciationInference
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    
    print("\n" + "="*60)
    print("Example 3: Real-time Pronunciation Scoring")
    print("="*60)
    
    detector = MispronunciationInference()
    
    # Record audio (5 seconds)
    print("Recording in 3 seconds...")
    import time
    time.sleep(3)
    
    print("🎤 Recording...")
    duration = 5  # seconds
    sample_rate = 16000
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    print("✓ Recording complete")
    
    # Save temporary file
    temp_file = "temp_recording.wav"
    sf.write(temp_file, audio, sample_rate)
    
    # Predict
    result = detector.predict_file(temp_file)
    detector.print_prediction(result)
    
    # Cleanup
    import os
    os.remove(temp_file)

def example_api_integration():
    """Example: Flask API for pronunciation detection"""
    print("\n" + "="*60)
    print("Example 4: Flask API Integration")
    print("="*60)
    
    print("""
from flask import Flask, request, jsonify
from inference import MispronunciationInference
import tempfile

app = Flask(__name__)
detector = MispronunciationInference()

@app.route('/analyze', methods=['POST'])
def analyze_pronunciation():
    # Get audio file from request
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    audio = request.files['audio']
    
    # Save temporarily
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        audio.save(tmp.name)
        
        # Predict
        result = detector.predict_file(tmp.name)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
# Test with:
# curl -X POST -F "audio=@test.wav" http://localhost:5000/analyze
    """)

def example_custom_threshold():
    """Example: Custom scoring thresholds"""
    from inference import MispronunciationInference
    
    print("\n" + "="*60)
    print("Example 5: Custom Scoring Thresholds")
    print("="*60)
    
    detector = MispronunciationInference()
    
    # Predict
    audio_path = "path/to/audio.wav"
    result = detector.predict_file(audio_path)
    
    # Custom interpretation
    score = result['pronunciation_score']
    
    # Define custom thresholds
    if score >= 4.5:
        level = "Excellent"
        feedback = "Near-native pronunciation!"
    elif score >= 4.0:
        level = "Good"
        feedback = "Good pronunciation with minor issues"
    elif score >= 3.0:
        level = "Acceptable"
        feedback = "Understandable but needs improvement"
    elif score >= 2.0:
        level = "Poor"
        feedback = "Significant pronunciation problems"
    else:
        level = "Very Poor"
        feedback = "Requires extensive practice"
    
    print(f"\nScore: {score:.2f}/5.0")
    print(f"Level: {level}")
    print(f"Feedback: {feedback}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MISPRONUNCIATION DETECTION - USAGE EXAMPLES")
    print("="*60)
    
    print("\nAvailable examples:")
    print("1. Single audio prediction")
    print("2. Batch processing")
    print("3. Real-time recording")
    print("4. Flask API integration")
    print("5. Custom thresholds")
    
    print("\nTo run an example, uncomment the function call below:")
    print("# example_single_prediction()")
    print("# example_batch_prediction()")
    print("# example_real_time_scoring()")
    print("# example_api_integration()")
    print("# example_custom_threshold()")