import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim
import numpy as np
import torch.nn.functional as F
if torch.cuda.is_available():
    device = torch.device('cuda')
    # print(f"Using device: CUDA (NVIDIA GPU)")
    # print(f"GPU name: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
    # print(f"Using device: MPS (Apple Silicon GPU)")
else:
    device = torch.device('cpu')
    # print(f"Using device: CPU (no GPU acceleration)")

# print(f"Device: {device}")
class Trainer:
    '''
    Class to train the GRU and the transformer
    '''
    def __init__(self,model,train_loader,val_loader,config,model_name):
        self.model = model
        self.model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_name = model_name


        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Loss with label smoothing for better generalization
        label_smoothing = getattr(config, 'LABEL_SMOOTHING', 0.0)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0

        # Cosine annealing with warmup for better training
        self.warmup_epochs = getattr(config, 'WARMUP_EPOCHS', 0)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS - self.warmup_epochs,
            eta_min=1e-6
        )
        self.current_epoch = 0


    def train_epoch(self):
        self.model.train()

        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Training {self.model_name}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            mask = batch['attention_mask'].to(device)
            
            self.optimizer.zero_grad()
            
            logits = self.model(input_ids, mask)
            loss = self.criterion(logits, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, data_loader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        
        top_k_hits = {k: 0 for k in self.config.TOP_K_VALUES}
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {self.model_name}"):
                input_ids = batch['input_ids'].to(device)
                targets = batch['targets'].to(device)
                mask = batch['attention_mask'].to(device)
                
                logits = self.model(input_ids, mask)
                loss = self.criterion(logits, targets)
                total_loss += loss.item()
                
                # Calculate top-k accuracy
                _, top_k_indices = torch.topk(logits, max(self.config.TOP_K_VALUES), dim=-1)
                
                for k in self.config.TOP_K_VALUES:
                    top_k_preds = top_k_indices[:, :k]
                    hits = (top_k_preds == targets.unsqueeze(1)).any(dim=1).sum().item()
                    top_k_hits[k] += hits
                
                total_samples += len(targets)
        
        
        avg_loss = total_loss / len(data_loader)
        accuracies = {k: (hits / total_samples * 100) for k, hits in top_k_hits.items()}
        
        return avg_loss, accuracies
    

    def _warmup_lr(self, epoch):
        """Learning rate warmup"""
        if epoch < self.warmup_epochs:
            lr_scale = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.LEARNING_RATE * lr_scale

    def train(self):
        """Full training loop with enhanced features"""
        print(f"\n{'='*80}")
        print(f"Training {self.model_name}")
        print(f"{'='*80}\n")

        for epoch in range(self.config.NUM_EPOCHS):
            self.current_epoch = epoch

            # Apply learning rate warmup
            if epoch < self.warmup_epochs:
                self._warmup_lr(epoch)
                print(f"Warmup Epoch {epoch + 1}/{self.warmup_epochs} - LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Diagnostic after first epoch

            if epoch == 0:  # After first epoch
                print("\n" + "=" * 80)
                print("POST-EPOCH-1 DIAGNOSTICS")
                print("=" * 80)
                
                # Check if embeddings changed
                import torch
                
                # Get embedding norms
                emb_norms = self.model.embedding.weight.norm(dim=1).detach().cpu().numpy()
                print(f"\nEmbedding Statistics:")
                print(f"  Mean norm: {emb_norms[1:].mean():.4f}")  # Skip PAD
                print(f"  Std norm: {emb_norms[1:].std():.4f}")
                print(f"  Min norm: {emb_norms[1:].min():.4f}")
                print(f"  Max norm: {emb_norms[1:].max():.4f}")
                
                # Check GRU gradients
                gru_grad_norms = []
                for name, param in self.model.named_parameters():
                    if param.grad is not None and 'gru' in name:
                        gru_grad_norms.append(param.grad.norm().item())
                
                if gru_grad_norms:
                    print(f"\nGRU Gradient Norms:")
                    print(f"  Mean: {np.mean(gru_grad_norms):.6f}")
                    print(f"  Max: {np.max(gru_grad_norms):.6f}")
                    print(f"  Min: {np.min(gru_grad_norms):.6f}")
                
                # Check prediction entropy (diversity)
                self.model.eval()
                with torch.no_grad():
                    sample_batch = next(iter(self.val_loader))
                    logits = self.model(
                        sample_batch['input_ids'].to(device),
                        sample_batch['attention_mask'].to(device)
                    )
                    probs = F.softmax(logits, dim=1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean().item()
                    max_entropy = np.log(logits.size(1))  # Log of vocab size
                    
                    print(f"\nPrediction Diversity:")
                    print(f"  Entropy: {entropy:.4f} / {max_entropy:.4f} ({entropy/max_entropy*100:.1f}%)")
                    
                    if entropy / max_entropy < 0.5:
                        print(f"  ⚠️  Low entropy - model might be collapsing to few predictions")
            
            self.model.train()
            print("=" * 80 + "\n")
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_accs = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accs)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Top-1 Acc: {val_accs[1]:.2f}%")
            print(f"Val Top-5 Acc: {val_accs[5]:.2f}%")
            print(f"Val Top-10 Acc: {val_accs[10]:.2f}%")
            print(f"Val Top-20 Acc: {val_accs[20]:.2f}%")
            print(f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Track best model by validation accuracy (Top-10)
            current_val_acc = val_accs[10]
            if current_val_acc > self.best_val_acc:
                self.best_val_acc = current_val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best')
                print("✓ Best model saved! (by Top-10 accuracy)")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config.PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}')

            # Learning rate scheduling (after warmup)
            if epoch >= self.warmup_epochs:
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
        }
        
        path = self.config.MODEL_DIR / f'{self.model_name}_{name}.pt'
        torch.save(checkpoint, path)