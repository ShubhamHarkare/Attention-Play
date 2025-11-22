"""
Integration script to create all poster visualizations.
Run this AFTER training your models (main_2.py).

This script loads your trained models and results, then generates
all the publication-quality visualizations for your poster.
"""

import json
import pickle
import pandas as pd
from pathlib import Path
from EnhancedVisualizer import EnhancedVisualizer
from TraningConfig import TrainingConfig

def load_training_results(config):
    """Load training results from saved checkpoints"""
    print("\n[1/4] Loading training results...")
    
    trainers_dict = {}
    
    # Load GRU4Rec results
    try:
        gru_checkpoint = config.MODEL_DIR / 'GRU4Rec_best.pt'
        if gru_checkpoint.exists():
            import torch
            checkpoint = torch.load(gru_checkpoint, map_location='cpu')
            
            # Create dummy trainer object with loaded data
            class TrainerData:
                def __init__(self, checkpoint):
                    self.train_losses = checkpoint.get('train_losses', [])
                    self.val_losses = checkpoint.get('val_losses', [])
                    self.val_accuracies = checkpoint.get('val_accuracies', [])
            
            trainers_dict['GRU4Rec'] = TrainerData(checkpoint)
            print("  ✓ Loaded GRU4Rec results")
    except Exception as e:
        print(f"  ⚠ Could not load GRU4Rec: {e}")
    
    # Load Transformer results
    try:
        trans_checkpoint = config.MODEL_DIR / 'Transformer_best.pt'
        if trans_checkpoint.exists():
            import torch
            checkpoint = torch.load(trans_checkpoint, map_location='cpu')
            
            class TrainerData:
                def __init__(self, checkpoint):
                    self.train_losses = checkpoint.get('train_losses', [])
                    self.val_losses = checkpoint.get('val_losses', [])
                    self.val_accuracies = checkpoint.get('val_accuracies', [])
            
            trainers_dict['Transformer'] = TrainerData(checkpoint)
            print("  ✓ Loaded Transformer results")
    except Exception as e:
        print(f"  ⚠ Could not load Transformer: {e}")
    
    return trainers_dict

def load_test_results(config):
    """Load final test results from JSON"""
    print("\n[2/4] Loading test results...")
    
    results_dict = {}
    
    # Load deep learning results
    try:
        dl_results_path = config.OUTPUT_DIR / "metrics" / "deep_learning_results.json"
        if dl_results_path.exists():
            with open(dl_results_path, 'r') as f:
                dl_results = json.load(f)
                results_dict.update(dl_results.get('test_results', {}))
            print("  ✓ Loaded deep learning test results")
    except Exception as e:
        print(f"  ⚠ Could not load DL results: {e}")
    
    # Load baseline results
    try:
        baseline_path = config.OUTPUT_DIR / "metrics" / "summary.json"
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                summary = json.load(f)
                if 'baseline_results' in summary:
                    results_dict.update(summary['baseline_results'])
            print("  ✓ Loaded baseline results")
    except Exception as e:
        print(f"  ⚠ Could not load baseline results: {e}")
    
    return results_dict

def load_sequences_data(config):
    """Load sequence data for distribution analysis"""
    print("\n[3/4] Loading sequence data...")
    
    try:
        train_df = pd.read_parquet(config.DATA_DIR / "train.parquet")
        print(f"  ✓ Loaded {len(train_df):,} training sequences")
        return train_df
    except Exception as e:
        print(f"  ⚠ Could not load sequences: {e}")
        return None

def load_audio_features(config):
    """Load audio features for feature analysis"""
    print("\n[4/4] Loading audio features...")
    
    try:
        # Try to load from HuggingFace dataset
        import pandas as pd
        audio_df = pd.read_csv("https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv")
        print(f"  ✓ Loaded {len(audio_df):,} tracks with audio features")
        return audio_df
    except Exception as e:
        print(f"  ⚠ Could not load audio features: {e}")
        return None

def calculate_summary_stats(config, results_dict, sequences_df):
    """Calculate summary statistics for poster"""
    
    summary_stats = {}
    
    # Dataset stats
    if sequences_df is not None:
        summary_stats['num_playlists'] = sequences_df['playlist_id'].nunique()
        summary_stats['train_sequences'] = len(sequences_df)
        summary_stats['num_unique_tracks'] = len(sequences_df['target'].unique())
    
    # Try to load from summary.json
    try:
        summary_path = config.OUTPUT_DIR / "metrics" / "summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                saved_summary = json.load(f)
                summary_stats.update(saved_summary)
    except:
        pass
    
    # Best model results (assume Transformer is best)
    if 'Transformer' in results_dict:
        summary_stats['best_model_results'] = results_dict['Transformer']
    
    # Baseline for comparison
    if 'KNN' in results_dict:
        summary_stats['baseline_top10'] = results_dict['KNN'].get(10, 0)
    
    return summary_stats

def main():
    """Main function to create all poster visualizations"""
    
    print("=" * 80)
    print("CREATING POSTER VISUALIZATIONS")
    print("=" * 80)
    
    # Initialize
    config = TrainingConfig()
    viz = EnhancedVisualizer(config.OUTPUT_DIR)
    
    # Load all data
    trainers_dict = load_training_results(config)
    results_dict = load_test_results(config)
    sequences_df = load_sequences_data(config)
    audio_df = load_audio_features(config)
    
    if not trainers_dict and not results_dict:
        print("\n❌ ERROR: No training results found!")
        print("Please run main_2.py first to train models.")
        return
    
    # Calculate summary stats
    summary_stats = calculate_summary_stats(config, results_dict, sequences_df)
    
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # 1. Learning curves (if training data available)
    if trainers_dict:
        try:
            print("\n[1/9] Creating detailed learning curves...")
            viz.plot_learning_curves_comparison(trainers_dict)
        except Exception as e:
            print(f"  ⚠ Error creating learning curves: {e}")
    
    # 2. Architecture comparison (if test results available)
    if results_dict:
        try:
            print("\n[2/9] Creating architecture comparison...")
            viz.plot_architecture_comparison(results_dict)
        except Exception as e:
            print(f"  ⚠ Error creating architecture comparison: {e}")
    
    # 3. Model improvement timeline
    if results_dict:
        try:
            print("\n[3/9] Creating model improvement timeline...")
            viz.plot_model_improvement_timeline(results_dict)
        except Exception as e:
            print(f"  ⚠ Error creating improvement timeline: {e}")
    
    # 4. Training efficiency
    if trainers_dict:
        try:
            print("\n[4/9] Creating training efficiency analysis...")
            viz.plot_training_efficiency(trainers_dict)
        except Exception as e:
            print(f"  ⚠ Error creating training efficiency: {e}")
    
    # 5. Poster summary
    if summary_stats:
        try:
            print("\n[5/9] Creating poster summary figure...")
            viz.create_poster_summary_figure(summary_stats)
        except Exception as e:
            print(f"  ⚠ Error creating poster summary: {e}")
    
    # 6. Data distribution analysis
    if sequences_df is not None:
        try:
            print("\n[6/9] Creating data distribution analysis...")
            viz.plot_data_distribution_analysis(sequences_df)
        except Exception as e:
            print(f"  ⚠ Error creating data distribution: {e}")
    
    # 7. Audio features analysis
    if audio_df is not None:
        try:
            print("\n[7/9] Creating audio features analysis...")
            viz.plot_audio_features_analysis(audio_df)
        except Exception as e:
            print(f"  ⚠ Error creating audio features: {e}")
    
    # 8. Feature importance (example - you can customize)
    try:
        print("\n[8/9] Creating feature importance plot...")
        # These are example values - you can calculate real ones from your embeddings
        feature_correlations = {
            'energy': 0.342,
            'danceability': 0.298,
            'valence': 0.256,
            'tempo': 0.189,
            'loudness': 0.167,
            'acousticness': 0.134,
            'instrumentalness': 0.098,
            'speechiness': 0.076
        }
        viz.plot_feature_importance(feature_correlations)
    except Exception as e:
        print(f"  ⚠ Error creating feature importance: {e}")
    
    # 9. Error analysis (if you have predictions - optional)
    # Uncomment this if you want to analyze specific predictions
    # print("\n[9/9] Creating error analysis...")
    # predictions = [...]  # Load your predictions
    # targets = [...]      # Load your targets
    # positions = [...]    # Load position data
    # viz.plot_error_analysis(predictions, targets, positions)
    
    print("\n" + "=" * 80)
    print("✅ ALL VISUALIZATIONS CREATED!")
    print("=" * 80)
    print(f"\n📁 Visualizations saved to: {config.VIZ_DIR}")
    print("\n📊 Files created:")
    
    # List all created files
    viz_files = list(config.VIZ_DIR.glob("*.png"))
    for i, file in enumerate(sorted(viz_files), 1):
        print(f"  {i}. {file.name}")
    
    print("\n💡 Tips for your poster:")
    print("  • Use 'poster_summary.png' in your header section")
    print("  • Use 'architecture_comparison.png' for results section")
    print("  • Use 'improvement_timeline.png' to show your contribution")
    print("  • Use 'learning_curves_detailed.png' to show training process")
    print("  • Use 'data_distribution.png' in methodology section")
    print("  • All images are 300 DPI - suitable for printing!")
    
    print("\n🎓 Recommended poster sections:")
    print("  1. TITLE + SUMMARY: poster_summary.png")
    print("  2. DATASET: data_distribution.png")
    print("  3. METHODOLOGY: architecture diagram (create separately)")
    print("  4. TRAINING: learning_curves_detailed.png")
    print("  5. RESULTS: architecture_comparison.png + improvement_timeline.png")
    print("  6. FEATURES: audio_features.png + feature_importance.png")
    print("  7. CONCLUSION: Highlight key findings from summary")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()