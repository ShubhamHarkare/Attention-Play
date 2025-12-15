"""
Comprehensive Performance Analysis and Visualization for Deep Learning Models
Compares GRU4Rec, Transformer, and Context-Aware Transformer

This script creates detailed visualizations showing:
1. Training progression comparison
2. Final performance comparison
3. Improvement analysis
4. Architecture efficiency metrics
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict

from TraningConfig import TrainingConfig

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


class DeepLearningComparator:
    """Comprehensive comparison of deep learning models"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = config.VIZ_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all results
        self.results = self.load_all_results()
    
    def load_all_results(self):
        """Load results from all trained models"""
        results = {
            'training_history': {},
            'test_performance': {},
            'model_info': {}
        }
        
        # Models to load
        models = ['GRU4Rec', 'Transformer', 'ContextAwareTransformer']
        
        for model_name in models:
            # Try to load checkpoint
            checkpoint_path = self.config.MODEL_DIR / f"{model_name}_best.pt"
            
            if checkpoint_path.exists():
                print(f"Loading {model_name}...")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # Extract training history
                results['training_history'][model_name] = {
                    'train_losses': checkpoint.get('train_losses', []),
                    'val_losses': checkpoint.get('val_losses', []),
                    'val_accuracies': checkpoint.get('val_accuracies', [])
                }
                
                # Extract final validation performance
                if checkpoint.get('val_accuracies'):
                    final_accs = checkpoint['val_accuracies'][-1]
                    results['test_performance'][model_name] = final_accs
                
                print(f"  ✓ Loaded {model_name}")
            else:
                print(f"  ⚠ {model_name} checkpoint not found at {checkpoint_path}")
        
        # Load test results if available
        try:
            test_results_path = self.config.OUTPUT_DIR / "metrics" / "deep_learning_results.json"
            if test_results_path.exists():
                with open(test_results_path, 'r') as f:
                    test_data = json.load(f)
                    # Update with actual test performance
                    results['test_performance'].update(test_data.get('test_results', {}))
        except Exception as e:
            print(f"  Note: Could not load test results: {e}")
        
        # Load context-aware results if available
        try:
            context_path = self.config.OUTPUT_DIR / "metrics" / "context_aware_results.json"
            if context_path.exists():
                with open(context_path, 'r') as f:
                    context_data = json.load(f)
                    context_accs = context_data['context_aware_results']['overall']['accuracies']
                    results['test_performance']['ContextAwareTransformer'] = context_accs
        except Exception as e:
            print(f"  Note: Could not load context-aware results: {e}")
        
        return results
    
    def plot_training_convergence(self, save_name="training_convergence.png"):
        """Compare training convergence speed and stability"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        history = self.results['training_history']
        colors = {'GRU4Rec': '#3498db', 'Transformer': '#2ecc71', 
                 'ContextAwareTransformer': '#9b59b6'}
        labels = {'GRU4Rec': 'GRU4Rec', 'Transformer': 'Transformer', 
                 'ContextAwareTransformer': 'Context-Aware'}
        
        # 1. Training Loss
        ax1 = axes[0, 0]
        for model, data in history.items():
            if data['train_losses']:
                epochs = range(1, len(data['train_losses']) + 1)
                ax1.plot(epochs, data['train_losses'], 
                        label=labels.get(model, model),
                        color=colors.get(model, 'gray'),
                        linewidth=2, marker='o', markersize=4, alpha=0.8)
        
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Training Loss', fontweight='bold')
        ax1.set_title('Training Loss Convergence', fontweight='bold', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Validation Loss
        ax2 = axes[0, 1]
        for model, data in history.items():
            if data['val_losses']:
                epochs = range(1, len(data['val_losses']) + 1)
                ax2.plot(epochs, data['val_losses'], 
                        label=labels.get(model, model),
                        color=colors.get(model, 'gray'),
                        linewidth=2, marker='s', markersize=4, alpha=0.8)
        
        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Validation Loss', fontweight='bold')
        ax2.set_title('Validation Loss Progression', fontweight='bold', fontsize=14)
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Top-10 Accuracy Evolution
        ax3 = axes[1, 0]
        for model, data in history.items():
            if data['val_accuracies']:
                epochs = range(1, len(data['val_accuracies']) + 1)
                top10_accs = [acc.get(10, 0) for acc in data['val_accuracies']]
                ax3.plot(epochs, top10_accs, 
                        label=labels.get(model, model),
                        color=colors.get(model, 'gray'),
                        linewidth=2.5, marker='o', markersize=5, alpha=0.8)
        
        ax3.set_xlabel('Epoch', fontweight='bold')
        ax3.set_ylabel('Top-10 Accuracy (%)', fontweight='bold')
        ax3.set_title('Top-10 Accuracy Evolution', fontweight='bold', fontsize=14)
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Learning Efficiency (Accuracy gain per epoch)
        ax4 = axes[1, 1]
        for model, data in history.items():
            if data['val_accuracies'] and len(data['val_accuracies']) > 1:
                top10_accs = [acc.get(10, 0) for acc in data['val_accuracies']]
                improvements = [top10_accs[i] - top10_accs[i-1] 
                              for i in range(1, len(top10_accs))]
                epochs = range(2, len(top10_accs) + 1)
                
                ax4.plot(epochs, improvements, 
                        label=labels.get(model, model),
                        color=colors.get(model, 'gray'),
                        linewidth=2, marker='o', markersize=4, alpha=0.7)
        
        ax4.axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        ax4.set_xlabel('Epoch', fontweight='bold')
        ax4.set_ylabel('Accuracy Gain (%)', fontweight='bold')
        ax4.set_title('Per-Epoch Accuracy Improvement', fontweight='bold', fontsize=14)
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_name}")
        plt.close()
    
    def plot_final_performance_comparison(self, save_name="dl_models_comparison.png"):
        """Bar chart comparing final test performance"""
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        test_perf = self.results['test_performance']
        models = ['GRU4Rec', 'Transformer', 'ContextAwareTransformer']
        labels = {'GRU4Rec': 'GRU4Rec', 'Transformer': 'Transformer', 
                 'ContextAwareTransformer': 'Context-Aware\nTransformer'}
        k_values = [1, 5, 10, 20]
        
        x = np.arange(len(k_values))
        width = 0.25
        colors = ['#3498db', '#2ecc71', '#9b59b6']
        
        for i, model in enumerate(models):
            if model in test_perf:
                accuracies = [test_perf[model].get(k, 0) for k in k_values]
                offset = (i - 1) * width
                
                bars = ax.bar(x + offset, accuracies, width, 
                            label=labels.get(model, model),
                            color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
                
                # Add value labels on bars
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{acc:.1f}%', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Top-K Metric', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
        ax.set_title('Deep Learning Models: Test Set Performance Comparison', 
                    fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Top-{k}' for k in k_values])
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add note about best model
        best_model = max(test_perf.items(), 
                        key=lambda x: x[1].get(10, 0) if isinstance(x[1], dict) else 0)
        best_acc = best_model[1].get(10, 0) if isinstance(best_model[1], dict) else 0
        
        ax.text(0.02, 0.98, 
               f'Best Model (Top-10): {labels.get(best_model[0], best_model[0])} ({best_acc:.2f}%)',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_name}")
        plt.close()
    
    def plot_improvement_cascade(self, save_name="improvement_cascade.png"):
        """Show progressive improvement from GRU → Transformer → Context-Aware"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        test_perf = self.results['test_performance']
        
        # Get Top-10 accuracies
        models_order = ['GRU4Rec', 'Transformer', 'ContextAwareTransformer']
        display_names = ['GRU4Rec', 'Transformer', 'Context-Aware\nTransformer']
        top10_accs = []
        
        for model in models_order:
            if model in test_perf:
                top10_accs.append(test_perf[model].get(10, 0))
            else:
                top10_accs.append(0)
        
        colors = ['#3498db', '#2ecc71', '#9b59b6']
        
        # Create bars
        bars = ax.bar(display_names, top10_accs, color=colors, 
                     alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, acc in zip(bars, top10_accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.2f}%', ha='center', va='bottom', 
                   fontsize=14, fontweight='bold')
        
        # Add improvement arrows and percentages
        for i in range(len(top10_accs) - 1):
            if top10_accs[i] > 0 and top10_accs[i+1] > 0:
                improvement = ((top10_accs[i+1] - top10_accs[i]) / top10_accs[i]) * 100
                abs_improvement = top10_accs[i+1] - top10_accs[i]
                
                # Draw arrow
                mid_y = (top10_accs[i] + top10_accs[i+1]) / 2
                ax.annotate('', xy=(i+1, top10_accs[i+1]), xytext=(i, top10_accs[i]),
                          arrowprops=dict(arrowstyle='->', lw=2, color='green', alpha=0.6))
                
                # Add improvement text
                ax.text(i + 0.5, mid_y, 
                       f'+{abs_improvement:.2f}%\n({improvement:+.1f}%)',
                       ha='center', va='center', fontsize=11, 
                       fontweight='bold', color='darkgreen',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', 
                                alpha=0.7, edgecolor='green', linewidth=1.5))
        
        ax.set_ylabel('Top-10 Accuracy (%)', fontweight='bold', fontsize=13)
        ax.set_title('Progressive Improvement in Deep Learning Models', 
                    fontweight='bold', fontsize=15, pad=20)
        ax.set_ylim(0, max(top10_accs) * 1.15)
        ax.grid(axis='y', alpha=0.3)
        
        # Add summary text box
        total_improvement = top10_accs[-1] - top10_accs[0] if len(top10_accs) > 1 else 0
        rel_improvement = (total_improvement / top10_accs[0] * 100) if top10_accs[0] > 0 else 0
        
        summary_text = f'Total Improvement:\n{total_improvement:.2f}% absolute\n{rel_improvement:+.1f}% relative'
        ax.text(0.98, 0.98, summary_text,
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightyellow', 
                        alpha=0.8, edgecolor='orange', linewidth=2))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_name}")
        plt.close()
    
    def plot_topk_progression(self, save_name="topk_progression.png"):
        """Line plot showing how each model performs across Top-K metrics"""
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        test_perf = self.results['test_performance']
        models = ['GRU4Rec', 'Transformer', 'ContextAwareTransformer']
        labels = {'GRU4Rec': 'GRU4Rec', 'Transformer': 'Transformer', 
                 'ContextAwareTransformer': 'Context-Aware Transformer'}
        k_values = [1, 5, 10, 20]
        
        colors = {'GRU4Rec': '#3498db', 'Transformer': '#2ecc71', 
                 'ContextAwareTransformer': '#9b59b6'}
        markers = {'GRU4Rec': 'o', 'Transformer': 's', 'ContextAwareTransformer': 'D'}
        
        for model in models:
            if model in test_perf:
                accuracies = [test_perf[model].get(k, 0) for k in k_values]
                
                ax.plot(k_values, accuracies, 
                       label=labels.get(model, model),
                       color=colors.get(model, 'gray'),
                       marker=markers.get(model, 'o'),
                       linewidth=3, markersize=12, alpha=0.8)
                
                # Add value labels
                for k, acc in zip(k_values, accuracies):
                    ax.annotate(f'{acc:.1f}%', 
                              xy=(k, acc), 
                              xytext=(0, 8), 
                              textcoords='offset points',
                              ha='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('K (Top-K Predictions)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
        ax.set_title('Model Performance Across Top-K Metrics', 
                    fontweight='bold', fontsize=14)
        ax.set_xticks(k_values)
        ax.set_xticklabels([f'Top-{k}' for k in k_values])
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_name}")
        plt.close()
    
    def create_comparison_table(self):
        """Create detailed comparison table"""
        
        test_perf = self.results['test_performance']
        history = self.results['training_history']
        
        data = []
        
        for model in ['GRU4Rec', 'Transformer', 'ContextAwareTransformer']:
            row = {
                'Model': model.replace('Transformer', ' Transformer').replace('Aware', '-Aware'),
            }
            
            # Test accuracies
            if model in test_perf:
                for k in [1, 5, 10, 20]:
                    row[f'Top-{k}'] = test_perf[model].get(k, 0)
            
            # Training info
            if model in history:
                row['Epochs Trained'] = len(history[model]['train_losses'])
                
                if history[model]['val_losses']:
                    row['Final Val Loss'] = history[model]['val_losses'][-1]
                    row['Best Val Loss'] = min(history[model]['val_losses'])
                
                if history[model]['val_accuracies']:
                    best_epoch = max(range(len(history[model]['val_accuracies'])),
                                   key=lambda i: history[model]['val_accuracies'][i].get(10, 0))
                    row['Best Epoch'] = best_epoch + 1
            
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    def generate_metrics_report(self):
        """Generate comprehensive metrics report"""
        
        print("\n" + "="*80)
        print("DEEP LEARNING MODELS: COMPREHENSIVE METRICS REPORT")
        print("="*80)
        
        test_perf = self.results['test_performance']
        
        # Overall Performance Summary
        print("\n" + "-"*80)
        print("TEST SET PERFORMANCE")
        print("-"*80)
        
        comparison_table = self.create_comparison_table()
        print("\n" + comparison_table.to_string(index=False))
        
        # Save table
        table_dir = self.config.OUTPUT_DIR / "metrics" / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        
        comparison_table.to_csv(table_dir / "dl_models_comparison.csv", index=False)
        print(f"\n✓ Saved comparison table to: {table_dir / 'dl_models_comparison.csv'}")
        
        # Improvement Analysis
        print("\n" + "-"*80)
        print("IMPROVEMENT ANALYSIS")
        print("-"*80)
        
        if 'GRU4Rec' in test_perf and 'Transformer' in test_perf:
            print("\nGRU4Rec → Transformer:")
            for k in [1, 10, 20]:
                gru_acc = test_perf['GRU4Rec'].get(k, 0)
                trans_acc = test_perf['Transformer'].get(k, 0)
                improvement = trans_acc - gru_acc
                rel_improvement = (improvement / gru_acc * 100) if gru_acc > 0 else 0
                print(f"  Top-{k}: {gru_acc:.2f}% → {trans_acc:.2f}% "
                      f"(+{improvement:.2f}%, {rel_improvement:+.1f}%)")
        
        if 'Transformer' in test_perf and 'ContextAwareTransformer' in test_perf:
            print("\nTransformer → Context-Aware Transformer:")
            for k in [1, 10, 20]:
                trans_acc = test_perf['Transformer'].get(k, 0)
                context_acc = test_perf['ContextAwareTransformer'].get(k, 0)
                improvement = context_acc - trans_acc
                rel_improvement = (improvement / trans_acc * 100) if trans_acc > 0 else 0
                print(f"  Top-{k}: {trans_acc:.2f}% → {context_acc:.2f}% "
                      f"(+{improvement:.2f}%, {rel_improvement:+.1f}%)")
        
        if 'GRU4Rec' in test_perf and 'ContextAwareTransformer' in test_perf:
            print("\nOverall: GRU4Rec → Context-Aware Transformer:")
            for k in [1, 10, 20]:
                gru_acc = test_perf['GRU4Rec'].get(k, 0)
                context_acc = test_perf['ContextAwareTransformer'].get(k, 0)
                improvement = context_acc - gru_acc
                rel_improvement = (improvement / gru_acc * 100) if gru_acc > 0 else 0
                print(f"  Top-{k}: {gru_acc:.2f}% → {context_acc:.2f}% "
                      f"(+{improvement:.2f}%, {rel_improvement:+.1f}%)")
        
        # Key Insights
        print("\n" + "-"*80)
        print("KEY INSIGHTS")
        print("-"*80)
        
        insights = []
        
        # Best model
        if test_perf:
            best_model = max(test_perf.items(), 
                           key=lambda x: x[1].get(10, 0) if isinstance(x[1], dict) else 0)
            best_acc = best_model[1].get(10, 0)
            insights.append(f"• Best Model: {best_model[0].replace('Transformer', ' Transformer')} "
                          f"with {best_acc:.2f}% Top-10 accuracy")
        
        # Transformer advantage
        if 'GRU4Rec' in test_perf and 'Transformer' in test_perf:
            gru_top10 = test_perf['GRU4Rec'].get(10, 0)
            trans_top10 = test_perf['Transformer'].get(10, 0)
            if trans_top10 > gru_top10:
                improvement = ((trans_top10 - gru_top10) / gru_top10 * 100)
                insights.append(f"• Transformer outperforms GRU4Rec by {improvement:.1f}% (Top-10)")
        
        # Context benefit
        if 'Transformer' in test_perf and 'ContextAwareTransformer' in test_perf:
            trans_top10 = test_perf['Transformer'].get(10, 0)
            context_top10 = test_perf['ContextAwareTransformer'].get(10, 0)
            if context_top10 > trans_top10:
                improvement = context_top10 - trans_top10
                insights.append(f"• Context conditioning provides +{improvement:.2f}% improvement")
        
        for insight in insights:
            print(f"\n{insight}")
        
        print("\n" + "="*80)


def main():
    """Generate all comparison visualizations and metrics"""
    
    config = TrainingConfig()
    
    print("="*80)
    print("DEEP LEARNING MODELS PERFORMANCE COMPARISON")
    print("="*80)
    
    comparator = DeepLearningComparator(config)
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    print("\n[1/5] Training convergence comparison...")
    comparator.plot_training_convergence()
    
    print("\n[2/5] Final performance comparison...")
    comparator.plot_final_performance_comparison()
    
    print("\n[3/5] Improvement cascade...")
    comparator.plot_improvement_cascade()
    
    print("\n[4/5] Top-K progression...")
    comparator.plot_topk_progression()
    
    print("\n[5/5] Generating metrics report...")
    comparator.generate_metrics_report()
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS AND METRICS GENERATED!")
    print("="*80)
    
    print(f"\n📁 Visualizations saved to: {config.VIZ_DIR}/")
    print("\nFiles created:")
    print("  1. training_convergence.png - Training history comparison")
    print("  2. dl_models_comparison.png - Final test performance bars")
    print("  3. improvement_cascade.png - Progressive improvement chart")
    print("  4. topk_progression.png - Performance across K values")
    print("  5. dl_models_comparison.csv - Detailed metrics table")
    
    print("\n💡 For your report:")
    print("  • Use improvement_cascade.png to show model evolution")
    print("  • Use dl_models_comparison.png for main results")
    print("  • Use training_convergence.png to show learning dynamics")
    print("  • Use topk_progression.png to analyze Top-K behavior")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()