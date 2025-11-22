"""
Quick diagnostic script to check what's in your model checkpoints.
This helps identify if models have actual training data or just zeros.
"""

import torch
from pathlib import Path
from TraningConfig import TrainingConfig

def check_checkpoint(checkpoint_path):
    """Analyze a model checkpoint"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {checkpoint_path.name}")
    print('='*80)
    
    if not checkpoint_path.exists():
        print("❌ File does not exist!")
        return False
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✅ Checkpoint loaded successfully\n")
        
        # Check what's in the checkpoint
        print("Checkpoint contents:")
        for key in checkpoint.keys():
            print(f"  • {key}")
        
        # Check training history
        print("\n" + "-"*80)
        print("TRAINING HISTORY")
        print("-"*80)
        
        if 'train_losses' in checkpoint:
            losses = checkpoint['train_losses']
            print(f"\nTrain Losses ({len(losses)} epochs):")
            if losses:
                print(f"  First epoch: {losses[0]:.4f}")
                print(f"  Last epoch:  {losses[-1]:.4f}")
                print(f"  Best (min):  {min(losses):.4f}")
            else:
                print("  ⚠️  EMPTY - No training data recorded!")
        
        if 'val_losses' in checkpoint:
            losses = checkpoint['val_losses']
            print(f"\nValidation Losses ({len(losses)} epochs):")
            if losses:
                print(f"  First epoch: {losses[0]:.4f}")
                print(f"  Last epoch:  {losses[-1]:.4f}")
                print(f"  Best (min):  {min(losses):.4f}")
            else:
                print("  ⚠️  EMPTY - No validation data recorded!")
        
        if 'val_accuracies' in checkpoint:
            accs = checkpoint['val_accuracies']
            print(f"\nValidation Accuracies ({len(accs)} epochs):")
            if accs:
                # Check if accuracies are dictionaries with K values
                if isinstance(accs[0], dict):
                    print(f"\n  First epoch:")
                    for k, v in accs[0].items():
                        print(f"    Top-{k}: {v:.2f}%")
                    
                    print(f"\n  Last epoch:")
                    for k, v in accs[-1].items():
                        print(f"    Top-{k}: {v:.2f}%")
                    
                    # Find best epoch
                    best_epoch = max(range(len(accs)), key=lambda i: accs[i].get(10, 0))
                    print(f"\n  Best epoch: {best_epoch + 1}")
                    for k, v in accs[best_epoch].items():
                        print(f"    Top-{k}: {v:.2f}%")
                    
                    # Check if all values are zeros
                    all_zeros = all(
                        all(v == 0 for v in epoch_acc.values()) 
                        for epoch_acc in accs
                    )
                    if all_zeros:
                        print("\n  ⚠️  WARNING: All accuracies are 0! Model may not have trained properly.")
                else:
                    print(f"  First: {accs[0]}")
                    print(f"  Last:  {accs[-1]}")
            else:
                print("  ⚠️  EMPTY - No accuracy data recorded!")
        
        # Check model state
        print("\n" + "-"*80)
        print("MODEL STATE")
        print("-"*80)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"\nModel parameters: {len(state_dict)} tensors")
            
            # Check a few key parameters
            if 'embedding.weight' in state_dict:
                emb = state_dict['embedding.weight']
                print(f"\nEmbedding layer:")
                print(f"  Shape: {emb.shape}")
                print(f"  Mean: {emb.mean():.6f}")
                print(f"  Std:  {emb.std():.6f}")
                print(f"  Min:  {emb.min():.6f}")
                print(f"  Max:  {emb.max():.6f}")
                
                # Check if initialized (should not be all zeros)
                if emb.abs().max() < 1e-6:
                    print("  ⚠️  WARNING: Embeddings appear uninitialized (all near zero)!")
            
            # Check a random layer
            sample_key = list(state_dict.keys())[5] if len(state_dict) > 5 else list(state_dict.keys())[0]
            sample_tensor = state_dict[sample_key]
            print(f"\nSample parameter '{sample_key}':")
            print(f"  Shape: {sample_tensor.shape}")
            print(f"  Mean: {sample_tensor.mean():.6f}")
            print(f"  Std:  {sample_tensor.std():.6f}")
        
        # Overall assessment
        print("\n" + "="*80)
        print("ASSESSMENT")
        print("="*80)
        
        has_training_data = (
            'train_losses' in checkpoint and 
            'val_losses' in checkpoint and 
            'val_accuracies' in checkpoint and
            len(checkpoint.get('train_losses', [])) > 0
        )
        
        has_nonzero_accs = False
        if 'val_accuracies' in checkpoint and checkpoint['val_accuracies']:
            if isinstance(checkpoint['val_accuracies'][0], dict):
                has_nonzero_accs = any(
                    any(v > 0 for v in epoch_acc.values())
                    for epoch_acc in checkpoint['val_accuracies']
                )
        
        if has_training_data and has_nonzero_accs:
            print("✅ Checkpoint looks good! Contains valid training data.")
            return True
        elif has_training_data and not has_nonzero_accs:
            print("⚠️  Checkpoint has training history but all accuracies are 0.")
            print("    This suggests the model didn't learn anything.")
            print("    Possible issues:")
            print("    - Learning rate too low/high")
            print("    - Data loading issue")
            print("    - Model architecture problem")
            return False
        else:
            print("❌ Checkpoint is incomplete or empty.")
            print("    You may need to re-train the model.")
            return False
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False


def main():
    """Check all model checkpoints"""
    
    config = TrainingConfig()
    
    print("="*80)
    print("MODEL CHECKPOINT DIAGNOSTIC")
    print("="*80)
    print(f"\nChecking models in: {config.MODEL_DIR}")
    
    # List all checkpoint files
    checkpoint_files = list(config.MODEL_DIR.glob("*.pt"))
    
    if not checkpoint_files:
        print(f"\n❌ No checkpoint files found in {config.MODEL_DIR}")
        print("\nYou need to train models first:")
        print("  python main_2.py")
        return
    
    print(f"\nFound {len(checkpoint_files)} checkpoint files:")
    for f in checkpoint_files:
        print(f"  • {f.name}")
    
    # Check each checkpoint
    results = {}
    for checkpoint_path in checkpoint_files:
        results[checkpoint_path.name] = check_checkpoint(checkpoint_path)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for name, is_valid in results.items():
        status = "✅ VALID" if is_valid else "❌ INVALID/EMPTY"
        print(f"{name:<40} {status}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    valid_count = sum(results.values())
    total_count = len(results)
    
    if valid_count == 0:
        print("\n❌ No valid checkpoints found!")
        print("\nAction needed:")
        print("  1. Re-train your models: python main_2.py")
        print("  2. Check for training errors in the logs")
        print("  3. Verify your dataset is loaded correctly")
    elif valid_count < total_count:
        print(f"\n⚠️  Only {valid_count}/{total_count} checkpoints are valid.")
        print("\nAction needed:")
        print("  1. Re-train models that failed")
        print("  2. Check training logs for errors")
    else:
        print("\n✅ All checkpoints are valid!")
        print("\nNext steps:")
        print("  1. Run: python evaluate_pretrained_models.py")
        print("  2. Then: python create_poster_visualizations.py")
    
    print("="*80)


if __name__ == "__main__":
    main()