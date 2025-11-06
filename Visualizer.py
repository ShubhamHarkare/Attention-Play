from config import Config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List,Dict
from collections import Counter
import seaborn as sns

class Visualizer:
    """Create visualizations for project poster"""
    
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = config.OUTPUT_DIR / "visualizations"
        
    def plot_dataset_statistics(self, playlists: List[Dict], audio_df: pd.DataFrame):
        """Plot dataset statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Playlist length distribution
        lengths = [len(p['tracks']) for p in playlists]
        axes[0, 0].hist(lengths, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Playlist Length')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Playlist Lengths')
        axes[0, 0].axvline(np.median(lengths), color='red', linestyle='--', 
                          label=f'Median: {np.median(lengths):.0f}')
        axes[0, 0].legend()
        
        # Audio feature distributions
        if 'valence' in audio_df.columns and 'energy' in audio_df.columns:
            axes[0, 1].scatter(audio_df['valence'], audio_df['energy'], 
                              alpha=0.3, s=1)
            axes[0, 1].set_xlabel('Valence')
            axes[0, 1].set_ylabel('Energy')
            axes[0, 1].set_title('Song Characteristics: Valence vs Energy')
        
        # Track popularity in playlists
        track_counts = Counter()
        for p in playlists:
            for t in p['tracks']:
                track_counts[t['track_uri']] += 1
        
        counts = list(track_counts.values())
        axes[1, 0].hist(counts, bins=50, edgecolor='black', alpha=0.7, log=True)
        axes[1, 0].set_xlabel('Number of Playlist Appearances')
        axes[1, 0].set_ylabel('Number of Tracks (log scale)')
        axes[1, 0].set_title('Track Popularity Distribution')
        
        # Feature correlation heatmap
        feature_cols = [col for col in self.config.AUDIO_FEATURES 
                       if col in audio_df.columns][:6]  # Top 6 features
        corr = audio_df[feature_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Audio Feature Correlations')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_statistics.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: dataset_statistics.png")
        plt.close()
    
    def plot_baseline_comparison(self, results: Dict):
        """Plot baseline model comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(results.keys())
        k_values = self.config.TOP_K_VALUES
        
        x = np.arange(len(k_values))
        width = 0.35
        
        for i, model in enumerate(models):
            accuracies = [results[model][k] for k in k_values]
            ax.bar(x + i * width, accuracies, width, label=model)
        
        ax.set_xlabel('Top-K')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Baseline Model Performance Comparison')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([f'Top-{k}' for k in k_values])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: baseline_comparison.png")
        plt.close()
    
    def plot_sequence_length_distribution(self, sequences_df: pd.DataFrame):
        """Plot distribution of sequence lengths"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        history_lengths = sequences_df['history'].apply(len)
        ax.hist(history_lengths, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('History Length (Number of Previous Songs)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Training Sequence Lengths')
        ax.axvline(history_lengths.median(), color='red', linestyle='--',
                  label=f'Median: {history_lengths.median():.0f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sequence_lengths.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: sequence_lengths.png")
        plt.close()
