import matplotlib.pyplot as plt
from PlaylistTransformer import PlaylistTransformer
import torch
import seaborn as sns
import numpy as np

# Device detection with proper error handling
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
class DeepLearningVisualizer:
    """Visualizations for deep learning results"""
    
    def __init__(self, config):
        self.config = config
        
    def plot_training_curves(self, trainers_dict):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        for name, trainer in trainers_dict.items():
            epochs = range(1, len(trainer.train_losses) + 1)
            axes[0].plot(epochs, trainer.train_losses, label=f'{name} Train', marker='o')
            axes[0].plot(epochs, trainer.val_losses, label=f'{name} Val', marker='s', linestyle='--')
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Accuracy curves (Top-10)
        for name, trainer in trainers_dict.items():
            epochs = range(1, len(trainer.val_accuracies) + 1)
            top10_accs = [acc[10] for acc in trainer.val_accuracies]
            axes[1].plot(epochs, top10_accs, label=name, marker='o')
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Top-10 Accuracy (%)')
        axes[1].set_title('Validation Top-10 Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.VIZ_DIR / 'training_curves.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: training_curves.png")
        plt.close()
    
    def plot_model_comparison(self, results_dict):
        """Compare all models including baselines"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(results_dict.keys())
        k_values = self.config.TOP_K_VALUES
        
        x = np.arange(len(k_values))
        width = 0.15
        
        for i, model in enumerate(models):
            accuracies = [results_dict[model][k] for k in k_values]
            ax.bar(x + i * width, accuracies, width, label=model)
        
        ax.set_xlabel('Top-K')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Model Performance Comparison (All Methods)')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels([f'Top-{k}' for k in k_values])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.VIZ_DIR / 'model_comparison_all.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: model_comparison_all.png")
        plt.close()
    
    def visualize_attention(self, model, dataset, idx=0):
        """Visualize attention weights from Transformer"""
        if not isinstance(model, PlaylistTransformer):
            print("Attention visualization only available for Transformer")
            return
        
        model.eval()
        sample = dataset[idx]
        
        input_ids = torch.LongTensor([sample['history']]).to(device)
        mask = torch.ones(1, len(sample['history']), dtype=torch.bool).to(device)
        
        attention_weights = model.get_attention_weights(input_ids, mask)
        
        if len(attention_weights) > 0:
            # Plot attention from last layer
            last_layer_attn = attention_weights[-1][0].cpu().numpy()  # (num_heads, seq_len, seq_len)
            
            # Average over heads
            avg_attn = last_layer_attn.mean(axis=0)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(avg_attn, cmap='viridis', ax=ax, cbar_kws={'label': 'Attention Weight'})
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            ax.set_title('Transformer Attention Weights (Last Layer, Averaged over Heads)')
            
            plt.tight_layout()
            plt.savefig(self.config.VIZ_DIR / 'attention_heatmap.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: attention_heatmap.png")
            plt.close()