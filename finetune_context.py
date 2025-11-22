"""
Fine-tune Pre-trained Transformer with Context Conditioning
This script loads your existing trained Transformer and fine-tunes it with context awareness.

Key Differences from Training from Scratch:
1. Loads pre-trained weights from your existing model
2. Freezes lower layers (optional - for faster training)
3. Only trains context embedding and top layers
4. Uses lower learning rate to avoid catastrophic forgetting
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import pickle

from ContextAwareTransformer import ContextAwareTransformer
from ContextExtractor import ContextExtractor
from TraningConfig import TrainingConfig

# Device detection
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")


class ContextPlaylistDataset(Dataset):
    """Dataset with context labels"""
    
    def __init__(self, data_path, vocab, context_extractor, max_seq_length=50):
        self.df = pd.read_parquet(data_path)
        self.vocab = vocab
        self.context_extractor = context_extractor
        self.max_seq_length = max_seq_length
        
        # Special tokens
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        
        # Build vocabulary
        self.track_to_idx = {track: idx + 2 for idx, track in enumerate(self.vocab)}
        self.idx_to_track = {idx: track for track, idx in self.track_to_idx.items()}
        
        # Add context if not present
        if 'context' not in self.df.columns:
            print("Adding context labels to dataset...")
            self.df = self.context_extractor.add_context_to_sequences(self.df)
        
        print(f"Dataset size: {len(self.df):,}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Parse history
        if isinstance(row['history'], str):
            history = row['history'].split('|') if row['history'] else []
        else:
            history = row['history']
        
        target = row['target']
        context_idx = row['context_idx']
        
        # Convert to indices
        history_indices = [
            self.track_to_idx.get(track, self.UNK_IDX) 
            for track in history
        ]
        target_idx = self.track_to_idx.get(target, self.UNK_IDX)
        
        # Truncate if too long
        if len(history_indices) > self.max_seq_length:
            history_indices = history_indices[-self.max_seq_length:]
        
        return {
            'history': history_indices,
            'target': target_idx,
            'context': context_idx,
            'seq_length': len(history_indices)
        }
    
    def collate_fn(self, batch):
        """Custom collate with context"""
        max_len = max([item['seq_length'] for item in batch])
        
        padded_histories = []
        targets = []
        contexts = []
        masks = []
        
        for item in batch:
            history = item['history']
            padding_length = max_len - len(history)
            
            padded_history = history + [self.PAD_IDX] * padding_length
            padded_histories.append(padded_history)
            
            mask = [1] * len(history) + [0] * padding_length
            masks.append(mask)
            
            targets.append(item['target'])
            contexts.append(item['context'])
        
        return {
            'input_ids': torch.LongTensor(padded_histories),
            'targets': torch.LongTensor(targets),
            'contexts': torch.LongTensor(contexts),
            'attention_mask': torch.BoolTensor(masks)
        }


def load_pretrained_transformer(pretrained_path, num_contexts):
    """
    Load your existing trained Transformer and convert to ContextAwareTransformer.
    
    This function:
    1. Loads the pretrained model checkpoint
    2. Transfers weights to ContextAwareTransformer
    3. Initializes new context embedding layer
    """
    print(f"\n[Loading] Pre-trained model from: {pretrained_path}")
    
    # Load checkpoint
    checkpoint = torch.load(pretrained_path, map_location=device)
    pretrained_state = checkpoint['model_state_dict']
    
    # Get model dimensions from pretrained weights
    num_items = pretrained_state['embedding.weight'].shape[0]
    d_model = pretrained_state['embedding.weight'].shape[1]
    
    print(f"  Vocabulary size: {num_items:,}")
    print(f"  Embedding dim: {d_model}")
    
    # Create new context-aware model
    context_model = ContextAwareTransformer(
        num_items=num_items,
        num_contexts=num_contexts,
        d_model=d_model,
        nhead=8,
        num_layer=4,
        dropout=0.1
    )
    
    # Transfer weights from pretrained model to context model
    context_state = context_model.state_dict()
    
    # Map old parameter names to new ones
    weight_mapping = {
        'embedding.weight': 'song_embedding.weight',
        'pos_encoder.pe': 'pos_encoder.pe',
        # Transformer layers keep same names
        # Prediction head keeps same names
    }
    
    transferred = 0
    for old_name, new_name in weight_mapping.items():
        if old_name in pretrained_state and new_name in context_state:
            context_state[new_name] = pretrained_state[old_name]
            transferred += 1
            print(f"  ✓ Transferred: {old_name} -> {new_name}")
    
    # Transfer transformer layers (they have the same structure)
    for key in pretrained_state.keys():
        if key.startswith('transformer.'):
            if key in context_state:
                context_state[key] = pretrained_state[key]
                transferred += 1
        elif key.startswith('fc1.') or key.startswith('fc2.'):
            if key in context_state:
                context_state[key] = pretrained_state[key]
                transferred += 1
    
    # Load the transferred weights
    context_model.load_state_dict(context_state)
    
    print(f"\n  ✓ Transferred {transferred} parameter groups")
    print(f"  ✓ Context embedding initialized randomly (will be trained)")
    
    return context_model


class FineTuner:
    """Fine-tuner for context-aware model"""
    
    def __init__(self, model, train_loader, val_loader, config, freeze_base=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Optionally freeze base model layers
        if freeze_base:
            print("\n[Freezing] Base transformer layers...")
            frozen_params = 0
            trainable_params = 0
            
            for name, param in self.model.named_parameters():
                # Freeze everything except context_embedding and top prediction layers
                if 'context_embedding' in name or 'fc1' in name or 'fc2' in name:
                    param.requires_grad = True
                    trainable_params += param.numel()
                    print(f"  ✓ Training: {name}")
                else:
                    param.requires_grad = False
                    frozen_params += param.numel()
            
            print(f"\n  Frozen parameters: {frozen_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Training only {trainable_params/(frozen_params+trainable_params)*100:.1f}% of model")
        
        # Get trainable parameters only
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        
        # Use lower learning rate for fine-tuning
        finetune_lr = config.LEARNING_RATE * 0.1  # 10x lower than training from scratch
        print(f"\n[Optimizer] Using learning rate: {finetune_lr:.6f} (10x lower for fine-tuning)")
        
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=finetune_lr,
            weight_decay=0.01
        )
        
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0, 
            label_smoothing=config.LABEL_SMOOTHING
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=1e-7
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Fine-tuning")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            contexts = batch['contexts'].to(device)
            mask = batch['attention_mask'].to(device)
            
            self.optimizer.zero_grad()
            
            logits = self.model(input_ids, contexts, mask)
            loss = self.criterion(logits, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, data_loader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        top_k_hits = {k: 0 for k in self.config.TOP_K_VALUES}
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                targets = batch['targets'].to(device)
                contexts = batch['contexts'].to(device)
                mask = batch['attention_mask'].to(device)
                
                logits = self.model(input_ids, contexts, mask)
                loss = self.criterion(logits, targets)
                total_loss += loss.item()
                
                _, top_k_indices = torch.topk(logits, max(self.config.TOP_K_VALUES), dim=-1)
                
                for k in self.config.TOP_K_VALUES:
                    top_k_preds = top_k_indices[:, :k]
                    hits = (top_k_preds == targets.unsqueeze(1)).any(dim=1).sum().item()
                    top_k_hits[k] += hits
                
                total += len(targets)
        
        avg_loss = total_loss / len(data_loader)
        accuracies = {k: (hits / total * 100) for k, hits in top_k_hits.items()}
        
        return avg_loss, accuracies
    
    def train(self):
        """Full fine-tuning loop"""
        print("\n" + "=" * 80)
        print("FINE-TUNING CONTEXT-AWARE TRANSFORMER")
        print("=" * 80 + "\n")
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            val_loss, val_accs = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accs)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Top-1: {val_accs[1]:.2f}%")
            print(f"Val Top-10: {val_accs[10]:.2f}%")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if val_accs[10] > self.best_val_acc:
                self.best_val_acc = val_accs[10]
                self.save_checkpoint('best')
                print("✓ Best model saved!")
            
            self.scheduler.step()
        
        return self.val_accuracies[-1]
    
    def save_checkpoint(self, name):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'config': {
                'd_model': self.model.d_model,
                'nhead': 8,
                'num_layer': 4
            }
        }
        
        path = self.config.MODEL_DIR / f'ContextAwareTransformer_{name}.pt'
        torch.save(checkpoint, path)
        print(f"Saved: {path}")


def main():
    """Main fine-tuning pipeline"""
    config = TrainingConfig()
    
    # Reduce epochs for fine-tuning (pretrained model needs less training)
    config.NUM_EPOCHS = 10  # Much fewer than training from scratch (50)
    
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    print("=" * 80)
    print("FINE-TUNE PRETRAINED TRANSFORMER WITH CONTEXT")
    print("=" * 80)
    
    # Check if pretrained model exists
    pretrained_path = config.MODEL_DIR / "Transformer_best.pt"
    if not pretrained_path.exists():
        print(f"\n❌ Error: Pretrained model not found at {pretrained_path}")
        print("\nPlease ensure you have a trained Transformer model.")
        print("If your model has a different name, update 'pretrained_path' in this script.")
        return
    
    # Prepare context data
    print("\n[1/5] Adding context labels to datasets...")
    context_extractor = ContextExtractor()
    
    for split in ['train', 'val', 'test']:
        data_path = config.DATA_DIR / f"{split}.parquet"
        df = pd.read_parquet(data_path)
        
        if 'context' not in df.columns:
            df = context_extractor.add_context_to_sequences(df)
            df.to_parquet(data_path, index=False)
            print(f"✓ Updated {split}.parquet")
    
    # Build vocabulary
    print("\n[2/5] Building vocabulary...")
    train_df = pd.read_parquet(config.DATA_DIR / "train.parquet")
    
    all_tracks = set()
    for _, row in train_df.iterrows():
        if isinstance(row['history'], str):
            tracks = row['history'].split('|') if row['history'] else []
        else:
            tracks = row['history']
        all_tracks.update(tracks)
        all_tracks.add(row['target'])
    
    vocab = sorted(list(all_tracks))
    num_contexts = context_extractor.get_num_contexts()
    
    print(f"Vocabulary: {len(vocab):,} tracks")
    print(f"Contexts: {num_contexts} categories")
    
    # Save vocabulary
    vocab_data = {
        'vocab': vocab,
        'track_to_idx': {track: idx + 2 for idx, track in enumerate(vocab)},
        'idx_to_track': {idx + 2: track for idx, track in enumerate(vocab)}
    }
    with open(config.MODEL_DIR / 'vocabulary.pkl', 'wb') as f:
        pickle.dump(vocab_data, f)
    print("✓ Saved vocabulary")
    
    # Create datasets
    print("\n[3/5] Creating datasets...")
    train_dataset = ContextPlaylistDataset(
        config.DATA_DIR / "train.parquet",
        vocab,
        context_extractor,
        config.MAX_SEQ_LENGTH
    )
    val_dataset = ContextPlaylistDataset(
        config.DATA_DIR / "val.parquet",
        vocab,
        context_extractor,
        config.MAX_SEQ_LENGTH
    )
    
    # Use smaller subset for faster fine-tuning (optional)
    # train_subset = torch.utils.data.Subset(train_dataset, range(min(100000, len(train_dataset))))
    
    train_loader = DataLoader(
        train_dataset,  # or train_subset
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
    
    print(f"Train: {len(train_dataset):,} sequences")
    print(f"Val: {len(val_dataset):,} sequences")
    
    # Load and adapt pretrained model
    print("\n[4/5] Loading pretrained model and adding context layer...")
    model = load_pretrained_transformer(pretrained_path, num_contexts)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Ask user about freezing
    print("\n" + "=" * 80)
    print("FREEZING STRATEGY")
    print("=" * 80)
    print("\nOptions:")
    print("  1. Freeze base model, train only context layer (FAST - 30min, less accurate)")
    print("  2. Fine-tune entire model (SLOW - 2-4 hours, more accurate)")
    print("")
    
    freeze_choice = 1
    freeze_base = (freeze_choice == "1")
    
    # Fine-tune
    print("\n[5/5] Fine-tuning model...")
    finetuner = FineTuner(model, train_loader, val_loader, config, freeze_base=freeze_base)
    finetuner.train()
    
    print("\n" + "=" * 80)
    print("FINE-TUNING COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Model saved to: {config.MODEL_DIR}")
    print(f"✅ Best Top-10 accuracy: {finetuner.best_val_acc:.2f}%")
    print("\nNext steps:")
    print("1. Run the frontend: python app.py")
    print("2. Compare with baseline in Week 5 analysis")


if __name__ == "__main__":
    main()