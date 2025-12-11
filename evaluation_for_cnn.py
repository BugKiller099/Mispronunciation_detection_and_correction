# ============================================================================
# FILE 8: evaluate.py
# Evaluation and testing script
# ============================================================================

import torch
import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from simple_cnn_model import SimpleCNNModel as ModelClass
from config import Config

from dataset import create_dataloaders

class Evaluator:
    """
    Evaluator class for testing the trained model.
    
    Provides:
    - Accuracy and F1 scores
    - Confusion matrix
    - Per-class metrics
    - Error analysis
    """
    
    def __init__(self, model_path=None, config=None):
        self.config = config or Config()
        self.device = torch.device(
            self.config.DEVICE if torch.cuda.is_available() else "cpu"
        )
        
        # Load model
        self.model = ModelClass(self.config).to(self.device)
        
        if model_path is None:
            model_path = self.config.BEST_MODEL_DIR / "best_model.pt"
        
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("✓ Model loaded successfully")
    
    def evaluate(self, test_loader):
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions = []
        all_labels = []
        all_scores = []
        all_predicted_scores = []
        
        print("\nEvaluating on test set...")
        
        with torch.no_grad():
            for batch in tqdm(test_loader):
                audio = batch['audio'].to(self.device)
                labels = batch['label'].to(self.device)
                scores = batch['score']
                
                # Forward pass
                logits, predicted_scores = self.model(audio)
                
                # Get predictions
                _, predicted = torch.max(logits, 1)
                
                # Collect results
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(scores.numpy())
                all_predicted_scores.extend(predicted_scores.cpu().numpy().flatten())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        all_predicted_scores = np.array(all_predicted_scores)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Classification report
        class_names = ['Poor', 'Acceptable', 'Good']
        # Define all possible labels (0, 1, 2) explicitly
        all_possible_labels = [0, 1, 2] 
        
        report = classification_report(
            all_labels, 
            all_predictions,
            target_names=class_names,
            labels=all_possible_labels,  # <--- THIS IS THE CRITICAL ADDITION
            output_dict=True,
            zero_division=0 # Optional: prevents warning if a class has 0 support/precision
        )
        
        # Score regression metrics
        score_mse = np.mean((all_scores - all_predicted_scores) ** 2)
        score_mae = np.mean(np.abs(all_scores - all_predicted_scores))
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'score_mse': score_mse,
            'score_mae': score_mae,
            'predictions': all_predictions,
            'labels': all_labels,
            'scores': all_scores,
            'predicted_scores': all_predicted_scores
        }
        
        return results
    
    def print_results(self, results):
        """Print evaluation results"""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        print(f"\nOverall Accuracy: {results['accuracy']*100:.2f}%")
        print(f"Weighted F1 Score: {results['f1_score']:.4f}")
        
        print("\nScore Regression Metrics:")
        print(f"MSE: {results['score_mse']:.4f}")
        print(f"MAE: {results['score_mae']:.4f}")
        
        print("\nPer-Class Metrics:")
        print("-" * 70)
        class_names = ['Poor', 'Acceptable', 'Good']
        report = results['classification_report']
        
        for class_name in class_names:
            metrics = report[class_name]
            print(f"\n{class_name}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1-score']:.4f}")
            print(f"  Support: {metrics['support']}")
    
    def plot_confusion_matrix(self, results, save_path=None):
        """Plot confusion matrix"""
        cm = results['confusion_matrix']
        class_names = ['Poor', 'Acceptable', 'Good']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path is None:
            save_path = self.config.RESULTS_DIR / "confusion_matrix.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_score_comparison(self, results, save_path=None):
        """Plot true vs predicted scores"""
        plt.figure(figsize=(10, 6))
        plt.scatter(
            results['scores'], 
            results['predicted_scores'],
            alpha=0.5
        )
        plt.plot([0, 5], [0, 5], 'r--', label='Perfect prediction')
        plt.xlabel('True Score')
        plt.ylabel('Predicted Score')
        plt.title('Pronunciation Score: True vs Predicted')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = self.config.RESULTS_DIR / "score_comparison.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Score comparison plot saved to {save_path}")
        plt.close()
    
    def save_results(self, results):
        """Save results to JSON"""
        # Convert numpy arrays to lists for JSON serialization
        results_json = {
            'accuracy': float(results['accuracy']),
            'f1_score': float(results['f1_score']),
            'score_mse': float(results['score_mse']),
            'score_mae': float(results['score_mae']),
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'classification_report': results['classification_report']
        }
        
        output_path = self.config.RESULTS_DIR / "evaluation_results.json"
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=4)
        
        print(f"✓ Results saved to {output_path}")

def main():
    """Main evaluation function"""
    config = Config()
    
    # Create test dataloader
    print("Loading test data...")
    _, _, test_loader = create_dataloaders(config)
    
    # Initialize evaluator
    evaluator = Evaluator(config=config)
    
    # Evaluate
    results = evaluator.evaluate(test_loader)
    
    # Print results
    evaluator.print_results(results)
    
    # Create visualizations
    evaluator.plot_confusion_matrix(results)
    evaluator.plot_score_comparison(results)
    
    # Save results
    evaluator.save_results(results)

if __name__ == "__main__":
    main()