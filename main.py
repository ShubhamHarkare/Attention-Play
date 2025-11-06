"""
CORRECTED main() function for AttentionPlay ML Pipeline
Replace your current main() function with this entire code block
"""

import json
import pickle
from BaselineModels import BaselineModels
from Evaluator import Evaluator
from TransformerDataPreparation import TransformerDataPreparation
from Visualizer import Visualizer
from config import Config
from dataloader import SpotifyDataLoader
from FeatureEngineer import FeatureEngineer


def main():
    """Run complete ML pipeline"""
    
    # Initialize
    config = Config()
    
    print("\n" + "=" * 80)
    print("STARTING ATTENTIONPLAY ML PIPELINE")
    print("=" * 80 + "\n")
    
    # Step 1-3: Load and preprocess data
    loader = SpotifyDataLoader(config)
    playlists_raw = loader.load_mpd_files()
    audio_df = loader.load_audio_features()
    playlists, audio_df_filtered, uri_to_id = loader.preprocess_data(playlists_raw, audio_df)
    
    # Step 4-5: Feature engineering
    engineer = FeatureEngineer(config)
    embeddings = engineer.create_audio_embeddings(audio_df_filtered, uri_to_id)
    transition_matrix = engineer.build_transition_matrix(playlists)
    
    # Step 6: Train baseline models
    baselines = BaselineModels(config)
    knn_model = baselines.train_knn(embeddings)
    
    # Step 7: Prepare Transformer data
    transformer_prep = TransformerDataPreparation(config)
    sequences_df = transformer_prep.create_sequences(playlists, uri_to_id)
    train_df, val_df, test_df = transformer_prep.train_val_test_split(sequences_df)
    transformer_prep.save_to_arrow(train_df, val_df, test_df, config.OUTPUT_DIR)
    
    # Evaluation
    print("\n" + "=" * 80)
    print("EVALUATING BASELINE MODELS")
    print("=" * 80)
    
    evaluator = Evaluator(config)
    evaluator.evaluate_baseline(baselines, test_df, "KNN")
    evaluator.evaluate_baseline(baselines, test_df, "Cosine")
    
    # Visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    viz = Visualizer(config)
    viz.plot_dataset_statistics(playlists, audio_df_filtered)
    viz.plot_baseline_comparison(evaluator.results)
    viz.plot_sequence_length_distribution(sequences_df)
    
    # Save models and metadata
    print("\n" + "=" * 80)
    print("SAVING MODELS AND METADATA")
    print("=" * 80)
    
    with open(config.OUTPUT_DIR / "models" / "baselines.pkl", 'wb') as f:
        pickle.dump({
            'knn_model': baselines,
            'embeddings': embeddings,
            'transition_matrix': transition_matrix,
            'uri_to_id': uri_to_id,
            'scaler': engineer.scaler
        }, f)
    print("✓ Saved baseline models")
    
    # Save summary statistics
    summary = {
        'num_playlists': len(playlists),
        'num_unique_tracks': len(embeddings),
        'num_sequences': len(sequences_df),
        'train_sequences': len(train_df),
        'val_sequences': len(val_df),
        'test_sequences': len(test_df),
        'baseline_results': evaluator.results
    }
    
    with open(config.OUTPUT_DIR / "metrics" / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print("✓ Saved summary statistics")
    
    # Print final summary (THIS GOES AT THE END, AFTER summary is created!)
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   • Processed {len(playlists):,} playlists")
    print(f"   • {len(embeddings):,} unique tracks with audio features")
    print(f"   • {len(sequences_df):,} training sequences generated")
    print(f"   • Train/Val/Test: {len(train_df):,}/{len(val_df):,}/{len(test_df):,}")
    print(f"\n📁 Output files saved to: {config.OUTPUT_DIR}")
    print(f"   • Arrow files: {config.OUTPUT_DIR / 'data'}/*.parquet")
    print(f"   • Models: {config.OUTPUT_DIR / 'models'}/baselines.pkl")
    print(f"   • Visualizations: {config.OUTPUT_DIR / 'visualizations'}/*.png")
    print(f"   • Metrics: {config.OUTPUT_DIR / 'metrics'}/summary.json")
    print("\n✅ Ready for Transformer training!")
    print("=" * 80)


if __name__ == "__main__":
    main()
