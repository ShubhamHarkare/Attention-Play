"""
Complete Evaluation and Visualization Suite for Context-Aware Transformer
Generates comprehensive metrics and publication-quality visualizations for your report.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from collections import defaultdict, Counter

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
        
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        
        self.track_to_idx = {track: idx + 2 for idx, track in enumerate(self.vocab)}
        self.idx_to_track = {idx: track for track, idx in self.track_to_idx.items()}
        
        if 'context' not in self.df.columns:
            self.df = self.context_extractor.add_context_to_sequences(self.df)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if isinstance(row['history'], str):
            history = row['history'].split('|') if row['history'] else []
        else:
            history = row['history']
        
        target = row['target']
        context_idx = row['context_idx']
        
        history_indices = [
            self.track_to_idx.get(track, self.UNK_IDX) 
            for track in history
        ]
        target_idx = self.track_to_idx.get(target, self.UNK_IDX)
        
        if len(history_indices) > self.max_seq_length:
            history_indices = history_indices[-self.max_seq_length:]
        
        return {
            'history': history_indices,
            'target': target_idx,
            'context': context_idx,
            'seq_length': len(history_indices)
        }
    
    def collate_fn(self, batch):
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


class ContextAwareEvaluator:
    """Comprehensive evaluation for context-aware model"""
    
    def __init__(self, model, test_loader, config, context_extractor):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.context_extractor = context_extractor
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        
        # Storage for detailed analysis
        self.predictions_by_context = defaultdict(list)
        self.targets_by_context = defaultdict(list)
        self.all_predictions = []
        self.all_targets = []
        self.all_contexts = []
        
    def evaluate_comprehensive(self):
        """Complete evaluation with context breakdown"""
        print("\n" + "="*80)
        print("COMPREHENSIVE CONTEXT-AWARE EVALUATION")
        print("="*80)
        
        self.model.eval()
        
        # Overall metrics
        total_loss = 0
        top_k_hits = {k: 0 for k in self.config.TOP_K_VALUES}
        
        # Context-specific metrics
        context_top_k_hits = {ctx: {k: 0 for k in self.config.TOP_K_VALUES} 
                              for ctx in self.context_extractor.contexts}
        context_counts = defaultdict(int)
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                targets = batch['targets'].to(device)
                contexts = batch['contexts'].to(device)
                mask = batch['attention_mask'].to(device)
                
                logits = self.model(input_ids, contexts, mask)
                loss = self.criterion(logits, targets)
                total_loss += loss.item()
                
                # Get top-k predictions
                _, top_k_indices = torch.topk(logits, max(self.config.TOP_K_VALUES), dim=-1)
                
                # Store for detailed analysis
                for i in range(len(targets)):
                    target = targets[i].item()
                    context_idx = contexts[i].item()
                    context_name = self.context_extractor.idx_to_context[context_idx]
                    
                    self.all_targets.append(target)
                    self.all_predictions.append(top_k_indices[i].cpu().numpy())
                    self.all_contexts.append(context_name)
                    
                    self.targets_by_context[context_name].append(target)
                    self.predictions_by_context[context_name].append(top_k_indices[i].cpu().numpy())
                    
                    context_counts[context_name] += 1
                    
                    # Calculate top-k accuracy
                    for k in self.config.TOP_K_VALUES:
                        top_k_preds = top_k_indices[i, :k]
                        hit = (top_k_preds == targets[i]).any().item()
                        
                        if hit:
                            top_k_hits[k] += 1
                            context_top_k_hits[context_name][k] += 1
        
        # Calculate metrics
        total_samples = len(self.all_targets)
        avg_loss = total_loss / len(self.test_loader)
        
        overall_accuracies = {k: (hits / total_samples * 100) 
                             for k, hits in top_k_hits.items()}
        
        context_accuracies = {}
        for ctx in self.context_extractor.contexts:
            if context_counts[ctx] > 0:
                context_accuracies[ctx] = {
                    k: (context_top_k_hits[ctx][k] / context_counts[ctx] * 100)
                    for k in self.config.TOP_K_VALUES
                }
            else:
                context_accuracies[ctx] = {k: 0 for k in self.config.TOP_K_VALUES}
        
        # Print results
        print(f"\n{'='*80}")
        print("OVERALL RESULTS")
        print('='*80)
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Total Samples: {total_samples:,}")
        print("\nTop-K Accuracies:")
        for k, acc in overall_accuracies.items():
            print(f"  Top-{k}: {acc:.2f}%")
        
        print(f"\n{'='*80}")
        print("CONTEXT-SPECIFIC RESULTS")
        print('='*80)
        
        for ctx in self.context_extractor.contexts:
            if context_counts[ctx] > 0:
                print(f"\n{ctx.upper()} ({context_counts[ctx]:,} samples):")
                for k in [1, 10, 20]:
                    print(f"  Top-{k}: {context_accuracies[ctx][k]:.2f}%")
        
        return {
            'overall': {
                'loss': avg_loss,
                'accuracies': overall_accuracies,
                'total_samples': total_samples
            },
            'by_context': {
                'accuracies': context_accuracies,
                'counts': dict(context_counts)
            }
        }
    
    def compare_with_baseline(self, baseline_results):
        """Compare context-aware vs baseline transformer"""
        print(f"\n{'='*80}")
        print("CONTEXT-AWARE vs BASELINE COMPARISON")
        print('='*80)
        
        if 'Transformer' not in baseline_results:
            print("Warning: Baseline transformer results not found")
            return None
        
        baseline_accs = baseline_results['Transformer']
        context_accs = self.evaluate_comprehensive()['overall']['accuracies']
        
        improvements = {}
        print(f"\n{'Metric':<15} {'Baseline':<12} {'Context-Aware':<15} {'Improvement':<15}")
        print("-"*60)
        
        for k in self.config.TOP_K_VALUES:
            baseline = baseline_accs.get(k, 0)
            context = context_accs.get(k, 0)
            improvement = ((context - baseline) / baseline * 100) if baseline > 0 else 0
            
            improvements[k] = {
                'baseline': baseline,
                'context_aware': context,
                'improvement': improvement
            }
            
            print(f"Top-{k:<11} {baseline:>10.2f}%  {context:>13.2f}%  "
                  f"{improvement:>+13.1f}%")
        
        return improvements


class ContextVisualizer:
    """Create publication-quality visualizations"""
    
    def __init__(self, output_dir, context_extractor):
        self.output_dir = Path(output_dir) / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.context_extractor = context_extractor
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 10
    
    def plot_context_performance(self, context_accuracies, context_counts, 
                                 save_name="context_performance.png"):
        """Bar chart comparing performance across contexts"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Sort contexts by Top-10 accuracy
        contexts = sorted(self.context_extractor.contexts, 
                         key=lambda c: context_accuracies[c].get(10, 0), 
                         reverse=True)
        
        # Performance chart
        ax1 = axes[0]
        top10_accs = [context_accuracies[ctx].get(10, 0) for ctx in contexts]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(contexts)))
        
        bars = ax1.barh(contexts, top10_accs, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, acc in zip(bars, top10_accs):
            width = bar.get_width()
            ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2., 
                    f'{acc:.1f}%', ha='left', va='center', fontweight='bold')
        
        ax1.set_xlabel('Top-10 Accuracy (%)', fontweight='bold')
        ax1.set_ylabel('Playlist Context', fontweight='bold')
        ax1.set_title('Model Performance by Context', fontweight='bold', fontsize=14)
        ax1.grid(axis='x', alpha=0.3)
        
        # Sample distribution
        ax2 = axes[1]
        counts = [context_counts[ctx] for ctx in contexts]
        
        bars = ax2.barh(contexts, counts, color=colors, alpha=0.8, edgecolor='black')
        
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax2.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2., 
                    f'{count:,}', ha='left', va='center', fontweight='bold', fontsize=9)
        
        ax2.set_xlabel('Number of Samples', fontweight='bold')
        ax2.set_ylabel('Playlist Context', fontweight='bold')
        ax2.set_title('Test Set Distribution by Context', fontweight='bold', fontsize=14)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_name}")
        plt.close()
    
    def plot_context_vs_baseline(self, improvements, save_name="context_improvement.png"):
        """Show improvement over baseline"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        k_values = sorted(improvements.keys())
        baseline_vals = [improvements[k]['baseline'] for k in k_values]
        context_vals = [improvements[k]['context_aware'] for k in k_values]
        
        x = np.arange(len(k_values))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline Transformer',
                      color='#3498db', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, context_vals, width, label='Context-Aware',
                      color='#2ecc71', alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Add improvement percentages
        for i, k in enumerate(k_values):
            improvement = improvements[k]['improvement']
            y_pos = max(baseline_vals[i], context_vals[i]) + 2
            ax.text(i, y_pos, f'+{improvement:.1f}%', ha='center', 
                   fontsize=9, color='green', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        ax.set_xlabel('Top-K Metric', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
        ax.set_title('Context-Aware Model Improves Over Baseline', 
                    fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Top-{k}' for k in k_values])
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_name}")
        plt.close()
    
    def plot_context_heatmap(self, context_accuracies, save_name="context_heatmap.png"):
        """Heatmap of accuracy across contexts and K values"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        contexts = sorted(self.context_extractor.contexts)
        k_values = [1, 5, 10, 20]
        
        # Create matrix
        matrix = np.zeros((len(contexts), len(k_values)))
        for i, ctx in enumerate(contexts):
            for j, k in enumerate(k_values):
                matrix[i, j] = context_accuracies[ctx].get(k, 0)
        
        sns.heatmap(matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                   xticklabels=[f'Top-{k}' for k in k_values],
                   yticklabels=[ctx.capitalize() for ctx in contexts],
                   cbar_kws={'label': 'Accuracy (%)'},
                   ax=ax, linewidths=1, linecolor='gray')
        
        ax.set_xlabel('Top-K Metric', fontweight='bold', fontsize=12)
        ax.set_ylabel('Playlist Context', fontweight='bold', fontsize=12)
        ax.set_title('Accuracy Heatmap: Performance by Context and K', 
                    fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_name}")
        plt.close()
    
    def plot_context_distribution_pie(self, context_counts, save_name="context_distribution.png"):
        """Pie chart showing context distribution"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort by count
        items = sorted(context_counts.items(), key=lambda x: x[1], reverse=True)
        contexts = [item[0].capitalize() for item in items]
        counts = [item[1] for item in items]
        
        colors = plt.cm.Set3(range(len(contexts)))
        
        wedges, texts, autotexts = ax.pie(counts, labels=contexts, autopct='%1.1f%%',
                                           colors=colors, startangle=90,
                                           explode=[0.05 if i == 0 else 0 for i in range(len(contexts))],
                                           shadow=True)
        
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        for text in texts:
            text.set_fontsize(11)
            text.set_fontweight('bold')
        
        ax.set_title('Test Set: Distribution of Playlist Contexts', 
                    fontweight='bold', fontsize=14, pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_name}")
        plt.close()
    
    def plot_full_comparison(self, all_results, save_name="full_model_comparison.png"):
        """Complete comparison including baselines and both transformers"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Extract data
        models = ['KNN', 'Cosine', 'GRU4Rec', 'Transformer', 'Context-Aware']
        k_values = [1, 5, 10, 20]
        
        colors_map = {
            'KNN': '#e74c3c',
            'Cosine': '#e67e22',
            'GRU4Rec': '#3498db',
            'Transformer': '#2ecc71',
            'Context-Aware': '#9b59b6'
        }
        
        for model in models:
            if model == 'Context-Aware':
                accs = [all_results['context_aware']['overall']['accuracies'].get(k, 0) 
                       for k in k_values]
            elif model in all_results.get('baseline', {}):
                accs = [all_results['baseline'][model].get(k, 0) for k in k_values]
            else:
                continue
            
            ax.plot(k_values, accs, marker='o', label=model,
                   linewidth=2.5, markersize=10, 
                   color=colors_map.get(model, 'gray'), alpha=0.8)
        
        ax.set_xlabel('Top-K Metric', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
        ax.set_title('Complete Model Comparison: Baselines to Context-Aware', 
                    fontweight='bold', fontsize=14)
        ax.set_xticks(k_values)
        ax.set_xticklabels([f'Top-{k}' for k in k_values])
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_name}")
        plt.close()


def main():
    """Main evaluation pipeline"""
    
    config = TrainingConfig()
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    print("="*80)
    print("CONTEXT-AWARE TRANSFORMER EVALUATION")
    print("="*80)
    
    # Initialize context extractor
    context_extractor = ContextExtractor()
    
    # Build vocabulary
    print("\n[1/5] Loading vocabulary...")
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
    num_items = len(vocab) + 2
    num_contexts = context_extractor.get_num_contexts()
    
    print(f"  Vocabulary: {len(vocab):,} tracks")
    print(f"  Contexts: {num_contexts} categories")
    
    # Load test dataset
    print("\n[2/5] Loading test dataset...")
    test_dataset = ContextPlaylistDataset(
        config.DATA_DIR / "test.parquet",
        vocab,
        context_extractor,
        config.MAX_SEQ_LENGTH
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=config.NUM_WORKERS
    )
    
    print(f"  Test samples: {len(test_dataset):,}")
    
    # Load model
    print("\n[3/5] Loading context-aware model...")
    model_path = config.MODEL_DIR / "ContextAwareTransformer_best.pt"
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the context-aware model first:")
        print("  python finetune_context.py")
        return
    
    model = ContextAwareTransformer(
        num_items=num_items,
        num_contexts=num_contexts,
        d_model=config.EMBEDDING_DIM,
        nhead=config.NUM_HEADS,
        num_layer=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"  ✓ Model loaded successfully")
    
    # Evaluate
    print("\n[4/5] Running comprehensive evaluation...")
    evaluator = ContextAwareEvaluator(model, test_loader, config, context_extractor)
    results = evaluator.evaluate_comprehensive()
    
    # Load baseline results
    print("\n[5/5] Comparing with baseline...")
    baseline_results = {}
    try:
        with open(config.OUTPUT_DIR / "metrics" / "deep_learning_results.json", 'r') as f:
            dl_results = json.load(f)
            baseline_results = dl_results.get('test_results', {})
    except:
        print("  Warning: Could not load baseline results")
    
    # Compare
    if baseline_results:
        improvements = evaluator.compare_with_baseline(baseline_results)
    else:
        improvements = None
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    viz = ContextVisualizer(config.OUTPUT_DIR, context_extractor)
    
    print("\n[1/5] Context performance comparison...")
    viz.plot_context_performance(
        results['by_context']['accuracies'],
        results['by_context']['counts']
    )
    
    print("\n[2/5] Context distribution...")
    viz.plot_context_distribution_pie(results['by_context']['counts'])
    
    print("\n[3/5] Performance heatmap...")
    viz.plot_context_heatmap(results['by_context']['accuracies'])
    
    if improvements:
        print("\n[4/5] Improvement over baseline...")
        viz.plot_context_vs_baseline(improvements)
    
    # Full comparison
    try:
        with open(config.OUTPUT_DIR / "metrics" / "summary.json", 'r') as f:
            summary = json.load(f)
            all_results = {
                'baseline': summary.get('baseline_results', {}),
                'context_aware': results
            }
            all_results['baseline'].update(baseline_results)
            
            print("\n[5/5] Full model comparison...")
            viz.plot_full_comparison(all_results)
    except:
        print("\n[5/5] Skipping full comparison (baseline data not found)")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    output_results = {
        'context_aware_results': results,
        'improvements': improvements if improvements else {},
        'config': {
            'num_contexts': num_contexts,
            'context_categories': context_extractor.contexts,
            'embedding_dim': config.EMBEDDING_DIM,
            'num_layers': config.NUM_LAYERS
        }
    }
    
    results_path = config.OUTPUT_DIR / "metrics" / "context_aware_results.json"
    with open(results_path, 'w') as f:
        json.dump(output_results, f, indent=2)
    
    print(f"✓ Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n✅ Evaluation complete!")
    print(f"\n📊 Overall Performance:")
    for k, acc in results['overall']['accuracies'].items():
        print(f"  Top-{k}: {acc:.2f}%")
    
    if improvements:
        print(f"\n📈 Improvement over Baseline Transformer:")
        for k in [1, 10, 20]:
            imp = improvements[k]['improvement']
            print(f"  Top-{k}: +{imp:.1f}%")
    
    print(f"\n🎨 Visualizations saved to: {config.OUTPUT_DIR / 'visualizations'}/")
    print(f"\n📁 Files created:")
    print("  • context_performance.png - Performance by context")
    print("  • context_distribution.png - Context distribution pie chart")
    print("  • context_heatmap.png - Accuracy heatmap")
    if improvements:
        print("  • context_improvement.png - Improvement over baseline")
        print("  • full_model_comparison.png - Complete model comparison")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()