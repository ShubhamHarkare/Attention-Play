from TraningConfig import TrainingConfig
import torch
import numpy as np
import pandas as pd
from PlayListDataset import PlaylistDataset
from torch.utils.data import DataLoader
from GRU4Rec import GRU4Rec
from Trainer import Trainer
from PlaylistTransformer import PlaylistTransformer
from DeepLearningVisualizer import DeepLearningVisualizer
import json

def build_vocabulary(train_path):
    """Build track vocabulary from training data"""
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
    print(f"Vocabulary size: {len(vocab):,} tracks")
    
    return vocab


def main():
    """Main training pipeline"""
    config = TrainingConfig()
    
    # Set random seeds
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    print("=" * 80)
    print("ATTENTIONPLAY: DEEP LEARNING TRAINING")
    print("=" * 80)
    
    # Build vocabulary
    print("\n[1/5] Building vocabulary...")
    vocab = build_vocabulary(config.DATA_DIR / "train.parquet")
    num_items = len(vocab) + 2  # +2 for PAD and UNK
    
    # Create datasets
    print("\n[2/5] Creating datasets...")
    train_dataset = PlaylistDataset(
        config.DATA_DIR / "train.parquet", 
        vocab, 
        config.MAX_SEQ_LENGTH
    )
    val_dataset = PlaylistDataset(
        config.DATA_DIR / "val.parquet", 
        vocab, 
        config.MAX_SEQ_LENGTH
    )
    test_dataset = PlaylistDataset(
        config.DATA_DIR / "test.parquet", 
        vocab, 
        config.MAX_SEQ_LENGTH
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=config.NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=config.NUM_WORKERS
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=config.NUM_WORKERS
    )
    
    
    print(f"Train: {len(train_dataset):,} sequences")
    print(f"Val: {len(val_dataset):,} sequences")
    print(f"Test: {len(test_dataset):,} sequences")
    # Add this RIGHT AFTER creating your dataset in main()
    print("\n" + "=" * 80)
    print("DATASET DIAGNOSTICS")
    print("=" * 80)

    # Check vocabulary
    print(f"\nVocabulary size: {len(vocab):,}")
    print(f"Dataset expects num_items: {num_items:,}")
    print(f"Match: {'✅' if num_items == len(vocab) + 2 else '❌'}")

    # Check a sample from the dataset
    sample = train_dataset[0]
    print(f"\nSample training example:")
    print(f"  History indices: {sample['history'][:5]}... (first 5)")
    print(f"  Target index: {sample['target']}")
    print(f"  Sequence length: {sample['seq_length']}")

    # Critical checks
    print(f"\n🔍 Critical Checks:")
    print(f"  Max history index: {max([max(train_dataset[i]['history']) if len(train_dataset[i]['history']) > 0 else 0 for i in range(min(100, len(train_dataset)))])}")
    print(f"  Max target index: {max([train_dataset[i]['target'] for i in range(min(100, len(train_dataset)))])}")
    print(f"  Num items in model: {num_items}")

    # Check if indices are in range
    max_idx_found = max([max(train_dataset[i]['history'] + [train_dataset[i]['target']]) if len(train_dataset[i]['history']) > 0 else train_dataset[i]['target'] for i in range(min(1000, len(train_dataset)))])
    print(f"  Max index found: {max_idx_found}")
    print(f"  Should be < {num_items}: {'✅' if max_idx_found < num_items else '❌ PROBLEM!'}")

    # Check for unknown tokens
    unk_count = sum([1 for i in range(min(1000, len(train_dataset))) if train_dataset.UNK_IDX in train_dataset[i]['history'] or train_dataset[i]['target'] == train_dataset.UNK_IDX])
    print(f"  Samples with UNK tokens: {unk_count}/1000 ({unk_count/10:.1f}%)")
    if unk_count > 100:
        print(f"  ⚠️ HIGH UNK RATE - Vocabulary mismatch!")

    print("=" * 80 + "\n")
    # Train GRU4Rec
    print("\n[3/5] Training GRU4Rec...")
    gru_model = GRU4Rec(
        num_items=num_items,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT
    )
    
    gru_trainer = Trainer(gru_model, train_loader, val_loader, config, "GRU4Rec")
    gru_results = gru_trainer.train()
    
    # Train Transformer
    print("\n[4/5] Training Transformer...")
    transformer_model = PlaylistTransformer(
        num_items=num_items,
        d_model=config.EMBEDDING_DIM,
        nhead=config.NUM_HEADS,
        num_layer=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )
    
    transformer_trainer = Trainer(
        transformer_model, train_loader, val_loader, config, "Transformer"
    )
    transformer_results = transformer_trainer.train()
    
    # Final evaluation on test set
    print("\n[5/5] Final evaluation on test set...")
    
    # Load best models
    gru_model.load_state_dict(
        torch.load(config.MODEL_DIR / 'GRU4Rec_best.pt')['model_state_dict']
    )
    transformer_model.load_state_dict(
        torch.load(config.MODEL_DIR / 'Transformer_best.pt')['model_state_dict']
    )
    
    gru_test_loss, gru_test_accs = gru_trainer.evaluate(test_loader)
    trans_test_loss, trans_test_accs = transformer_trainer.evaluate(test_loader)
    
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    print("\nGRU4Rec:")
    for k, acc in gru_test_accs.items():
        print(f"  Top-{k}: {acc:.2f}%")
    
    print("\nTransformer:")
    for k, acc in trans_test_accs.items():
        print(f"  Top-{k}: {acc:.2f}%")
    
    # Visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    viz = DeepLearningVisualizer(config)
    
    trainers_dict = {
        'GRU4Rec': gru_trainer,
        'Transformer': transformer_trainer
    }
    viz.plot_training_curves(trainers_dict)
    
    # Load baseline results if available
    results_dict = {
        'GRU4Rec': gru_test_accs,
        'Transformer': trans_test_accs
    }
    
    # Try to load baseline results
    try:
        with open(config.OUTPUT_DIR / "metrics" / "summary.json", 'r') as f:
            summary = json.load(f)
            if 'baseline_results' in summary:
                results_dict.update(summary['baseline_results'])
    except:
        print("Could not load baseline results")
    
    viz.plot_model_comparison(results_dict)
    viz.visualize_attention(transformer_model, test_dataset, idx=10)
    
    # Save final results
    final_results = {
        'test_results': {
            'GRU4Rec': gru_test_accs,
            'Transformer': trans_test_accs
        },
        'config': {
            'embedding_dim': config.EMBEDDING_DIM,
            'hidden_dim': config.HIDDEN_DIM,
            'num_heads': config.NUM_HEADS,
            'num_layers': config.NUM_LAYERS,
            'learning_rate': config.LEARNING_RATE,
            'batch_size': config.BATCH_SIZE
        }
    }
    
    with open(config.OUTPUT_DIR / "metrics" / "deep_learning_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Models saved to: {config.MODEL_DIR}")
    print(f"✅ Visualizations saved to: {config.VIZ_DIR}")
    print(f"✅ Results saved to: {config.OUTPUT_DIR / 'metrics'}")


if __name__ == "__main__":
    main()