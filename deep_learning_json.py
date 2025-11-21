"""
Extract test results from trained model checkpoints
Run this to create deep_learning_results.json
"""

import torch
import json
from pathlib import Path

def extract_results():
    """Extract results from model checkpoints"""
    
    OUTPUT_DIR = Path("output/metrics")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("EXTRACTING RESULTS FROM TRAINED MODELS")
    print("="*80 + "\n")
    
    # Load model checkpoints
    try:
        print("[1/3] Loading GRU4Rec checkpoint...")
        gru_checkpoint = torch.load("models_greatlakes/GRU4Rec_best.pt", map_location='cpu')
        print("✓ GRU4Rec checkpoint loaded")
        
        print("\n[2/3] Loading Transformer checkpoint...")
        trans_checkpoint = torch.load("models_greatlakes/Transformer_best.pt", map_location='cpu')
        print("✓ Transformer checkpoint loaded")
        
        # Extract final validation accuracies (as proxy for test if test results not saved)
        gru_val_accuracies = gru_checkpoint['val_accuracies'][-1]  # Last epoch
        trans_val_accuracies = trans_checkpoint['val_accuracies'][-1]
        
        print("\n[3/3] Extracted Results:")
        print("\nGRU4Rec (Final Validation Accuracy):")
        for k, acc in gru_val_accuracies.items():
            print(f"  Top-{k}: {acc:.2f}%")
        
        print("\nTransformer (Final Validation Accuracy):")
        for k, acc in trans_val_accuracies.items():
            print(f"  Top-{k}: {acc:.2f}%")
        
        # Create results JSON
        results = {
            'test_results': {
                'GRU4Rec': gru_val_accuracies,
                'Transformer': trans_val_accuracies
            },
            'config': {
                'embedding_dim': 256,
                'hidden_dim': 512,
                'num_heads': 8,
                'num_layers': 4,
                'learning_rate': 0.0005,
                'batch_size': 128
            },
            'note': 'Using final validation accuracies from training. For true test results, run evaluation on test set.'
        }
        
        # Save to file
        output_file = OUTPUT_DIR / "deep_learning_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
        print("\nYou can now run the visualization script!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Model checkpoint not found")
        print(f"   {e}")
        print("\nPlease ensure you have trained your models first:")
        print("   python main_2.py")
        return False
    
    except KeyError as e:
        print(f"\n⚠️  Warning: Checkpoint missing expected key: {e}")
        print("Creating results with mock data for visualization purposes...")
        
        # Create mock results for visualization
        results = {
            'test_results': {
                'GRU4Rec': {1: 8.5, 5: 24.2, 10: 37.8, 20: 51.3},
                'Transformer': {1: 10.2, 5: 26.8, 10: 41.5, 20: 54.7}
            },
            'config': {
                'embedding_dim': 256,
                'hidden_dim': 512,
                'num_heads': 8,
                'num_layers': 4,
                'learning_rate': 0.0005,
                'batch_size': 128
            },
            'note': 'MOCK DATA - Please train models and re-run this script for real results'
        }
        
        output_file = OUTPUT_DIR / "deep_learning_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Mock results saved to: {output_file}")
        print("⚠️  These are placeholder values - visualizations will work but show demo data")
        return False
    
    return True


def create_baseline_results():
    """Create baseline results JSON if it doesn't exist"""
    
    summary_file = Path("output/metrics/summary.json")
    
    if summary_file.exists():
        print("\n✓ summary.json already exists")
        return
    
    print("\n[Optional] Creating baseline results placeholder...")
    
    # Create placeholder baseline results
    summary = {
        'num_playlists': 100000,
        'num_unique_tracks': 50000,
        'num_sequences': 2500000,
        'train_sequences': 1750000,
        'val_sequences': 375000,
        'test_sequences': 375000,
        'baseline_results': {
            'KNN': {1: 3.5, 5: 12.8, 10: 21.3, 20: 32.5},
            'Cosine': {1: 3.2, 5: 11.9, 10: 19.8, 20: 30.1}
        },
        'note': 'Baseline results - run main.py to get actual values'
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Created {summary_file} with placeholder baseline results")


if __name__ == "__main__":
    success = extract_results()
    create_baseline_results()
    
    print("\n" + "="*80)
    if success:
        print("✅ ALL DONE! You can now run:")
        print("   python advanced_visualizations.py")
    else:
        print("⚠️  PARTIAL SUCCESS - Check warnings above")
        print("   Visualizations will work but may use mock/demo data")
    print("="*80 + "\n")