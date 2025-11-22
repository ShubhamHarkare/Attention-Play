"""
Evaluate Pre-trained Models and Generate Results
This script loads your trained models and evaluates them on the test set
to generate proper metrics for visualization.
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from TraningConfig import TrainingConfig
from PlayListDataset import PlaylistDataset
from GRU4Rec import GRU4Rec
from PlaylistTransformer import PlaylistTransformer

# Device detection
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")


def build_vocabulary(train_path):
    """Build track vocabulary from training data"""
    print("\n[1/5] Building vocabulary...")
    df = pd.read_parquet(train_path)
    
    all_tracks = set()
    for _, row in df.iterrows():
        if isinstance(row['history'], str):
            tracks = row['history'].split('|') if row['history'] else []
        else:
            tracks = row['history']
        all_tracks.update(tracks)
        all_tracks.add(row['target'])
    
    vocab = sorted(list(all_tracks))
    print(f"  ✓ Vocabulary size: {len(vocab):,} tracks")
    
    return vocab


def load_model(model_path, model_class, num_items, config):
    """Load a trained model from checkpoint"""
    print(f"\nLoading model from: {model_path.name}")
    
    # Create model instance
    if model_class == GRU4Rec:
        model = GRU4Rec(
            num_items=num_items,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            dropout=config.DROPOUT
        )
    elif model_class == PlaylistTransformer:
        model = PlaylistTransformer(
            num_items=num_items,
            d_model=config.EMBEDDING_DIM,
            nhead=config.NUM_HEADS,
            num_layer=config.NUM_LAYERS,
            dropout=config.DROPOUT
        )
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"  ✓ Model loaded successfully")
        
        # Print training history if available
        if 'val_accuracies' in checkpoint and len(checkpoint['val_accuracies']) > 0:
            final_acc = checkpoint['val_accuracies'][-1]
            print(f"  ✓ Final validation Top-10 accuracy: {final_acc.get(10, 0):.2f}%")
        
        return model, checkpoint
    
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return None, None


def evaluate_model(model, data_loader, config, model_name="Model"):
    """Evaluate model on dataset and calculate Top-K accuracies"""
    print(f"\n[Evaluating] {model_name}...")
    
    model.eval()
    top_k_hits = {k: 0 for k in config.TOP_K_VALUES}
    total_samples = 0
    
    all_losses = []
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {model_name}"):
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            mask = batch['attention_mask'].to(device)
            
            # Forward pass
            logits = model(input_ids, mask)
            
            # Calculate loss
            loss = criterion(logits, targets)
            all_losses.append(loss.item())
            
            # Calculate Top-K accuracy
            _, top_k_indices = torch.topk(logits, max(config.TOP_K_VALUES), dim=-1)
            
            for k in config.TOP_K_VALUES:
                top_k_preds = top_k_indices[:, :k]
                hits = (top_k_preds == targets.unsqueeze(1)).any(dim=1).sum().item()
                top_k_hits[k] += hits
            
            total_samples += len(targets)
    
    # Calculate final metrics
    avg_loss = np.mean(all_losses)
    accuracies = {k: (hits / total_samples * 100) for k, hits in top_k_hits.items()}
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"  Test Loss: {avg_loss:.4f}")
    for k, acc in accuracies.items():
        print(f"  Top-{k} Accuracy: {acc:.2f}%")
    
    return avg_loss, accuracies


def load_baseline_results(config):
    """Load baseline results if available"""
    print("\n[3/5] Loading baseline results...")
    
    baseline_results = {}
    
    try:
        summary_path = config.OUTPUT_DIR / "metrics" / "summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                if 'baseline_results' in summary:
                    baseline_results = summary['baseline_results']
                    print("  ✓ Loaded baseline results:")
                    for model, results in baseline_results.items():
                        print(f"    {model}: Top-10 = {results.get(10, 0):.2f}%")
    except Exception as e:
        print(f"  ⚠ Could not load baseline results: {e}")
    
    return baseline_results


def save_evaluation_results(results, config):
    """Save evaluation results to JSON"""
    print("\n[5/5] Saving results...")
    
    output_path = config.OUTPUT_DIR / "metrics" / "evaluation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Saved to: {output_path}")
    
    # Also update deep_learning_results.json for backward compatibility
    dl_results_path = config.OUTPUT_DIR / "metrics" / "deep_learning_results.json"
    if dl_results_path.exists():
        with open(dl_results_path, 'r') as f:
            dl_results = json.load(f)
    else:
        dl_results = {}
    
    # Update with new results
    dl_results['test_results'] = results['model_results']
    dl_results['config'] = results['config']
    
    with open(dl_results_path, 'w') as f:
        json.dump(dl_results, f, indent=2)
    
    print(f"  ✓ Updated: {dl_results_path}")


def main():
    """Main evaluation pipeline"""
    
    print("=" * 80)
    print("EVALUATING PRE-TRAINED MODELS")
    print("=" * 80)
    
    # Initialize config
    config = TrainingConfig()
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Build vocabulary
    vocab = build_vocabulary(config.DATA_DIR / "train.parquet")
    num_items = len(vocab) + 2  # +2 for PAD and UNK
    
    # Create test dataset
    print("\n[2/5] Loading test dataset...")
    test_dataset = PlaylistDataset(
        config.DATA_DIR / "test.parquet",
        vocab,
        config.MAX_SEQ_LENGTH
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=config.NUM_WORKERS
    )
    
    print(f"  ✓ Test dataset: {len(test_dataset):,} sequences")
    
    # Load baseline results
    baseline_results = load_baseline_results(config)
    
    # Evaluate models
    print("\n[4/5] Evaluating deep learning models...")
    
    model_results = {}
    model_checkpoints = {}
    
    # 1. Evaluate GRU4Rec
    gru_path = config.MODEL_DIR / "GRU4Rec_best.pt"
    if gru_path.exists():
        gru_model, gru_checkpoint = load_model(gru_path, GRU4Rec, num_items, config)
        if gru_model is not None:
            gru_loss, gru_accs = evaluate_model(gru_model, test_loader, config, "GRU4Rec")
            model_results['GRU4Rec'] = gru_accs
            model_checkpoints['GRU4Rec'] = gru_checkpoint
    else:
        print(f"\n⚠ GRU4Rec model not found at: {gru_path}")
    
    # 2. Evaluate Transformer
    trans_path = config.MODEL_DIR / "Transformer_best.pt"
    if trans_path.exists():
        trans_model, trans_checkpoint = load_model(trans_path, PlaylistTransformer, num_items, config)
        if trans_model is not None:
            trans_loss, trans_accs = evaluate_model(trans_model, test_loader, config, "Transformer")
            model_results['Transformer'] = trans_accs
            model_checkpoints['Transformer'] = trans_checkpoint
    else:
        print(f"\n⚠ Transformer model not found at: {trans_path}")
    
    # Check if we got any results
    if not model_results:
        print("\n" + "=" * 80)
        print("ERROR: No models found to evaluate!")
        print("=" * 80)
        print("\nPlease ensure you have trained models at:")
        print(f"  - {config.MODEL_DIR / 'GRU4Rec_best.pt'}")
        print(f"  - {config.MODEL_DIR / 'Transformer_best.pt'}")
        print("\nRun 'python main_2.py' to train the models first.")
        return
    
    # Combine all results
    all_results = {**baseline_results, **model_results}
    
    # Print comparison
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Model':<20} {'Top-1':<10} {'Top-5':<10} {'Top-10':<10} {'Top-20':<10}")
    print("-" * 60)
    
    for model_name in ['KNN', 'Cosine', 'GRU4Rec', 'Transformer']:
        if model_name in all_results:
            results = all_results[model_name]
            print(f"{model_name:<20} "
                  f"{results.get(1, 0):>8.2f}%  "
                  f"{results.get(5, 0):>8.2f}%  "
                  f"{results.get(10, 0):>8.2f}%  "
                  f"{results.get(20, 0):>8.2f}%")
    
    # Calculate improvements
    if 'KNN' in all_results and 'Transformer' in all_results:
        baseline_top10 = all_results['KNN'].get(10, 0)
        transformer_top10 = all_results['Transformer'].get(10, 0)
        
        if baseline_top10 > 0:
            improvement = ((transformer_top10 - baseline_top10) / baseline_top10) * 100
            print("\n" + "=" * 80)
            print("KEY FINDINGS")
            print("=" * 80)
            print(f"Baseline (KNN) Top-10 Accuracy: {baseline_top10:.2f}%")
            print(f"Transformer Top-10 Accuracy: {transformer_top10:.2f}%")
            print(f"Improvement: +{improvement:.1f}%")
    
    # Save results
    results_to_save = {
        'model_results': model_results,
        'baseline_results': baseline_results,
        'all_results': all_results,
        'config': {
            'embedding_dim': config.EMBEDDING_DIM,
            'hidden_dim': config.HIDDEN_DIM,
            'num_heads': config.NUM_HEADS,
            'num_layers': config.NUM_LAYERS,
            'batch_size': config.BATCH_SIZE,
            'vocab_size': num_items,
            'test_samples': len(test_dataset)
        },
        'summary': {
            'best_model': max(model_results.items(), key=lambda x: x[1].get(10, 0))[0] if model_results else None,
            'best_top10_accuracy': max([v.get(10, 0) for v in model_results.values()]) if model_results else 0
        }
    }
    
    save_evaluation_results(results_to_save, config)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {config.OUTPUT_DIR / 'metrics' / 'evaluation_results.json'}")
    print("\nNext steps:")
    print("  1. Run: python create_poster_visualizations.py")
    print("  2. Check: output/visualizations/ for all figures")
    print("=" * 80)


if __name__ == "__main__":
    main()