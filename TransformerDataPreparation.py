from config import Config
from typing import List,Dict,Tuple
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path

class TransformerDataPreparation:
    """Prepare sequential data for Transformer training"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def create_sequences(self, playlists: List[Dict], uri_to_id: Dict) -> pd.DataFrame:
        """Create sliding window sequences for next-song prediction"""
        print("\n[7/7] Creating sequences for Transformer training...")
        
        sequences = []
        
        for playlist in tqdm(playlists, desc="Creating sequences"):
            tracks = [t['track_uri'] for t in playlist['tracks']]
            playlist_name = playlist.get('name', '')
            
            # Sliding window approach
            for i in range(1, len(tracks)):
                sequence = {
                    'playlist_id': playlist['pid'],
                    'playlist_name': playlist_name,
                    'history': tracks[:i],  # All previous tracks
                    'target': tracks[i],     # Next track to predict
                    'position': i,
                    'playlist_length': len(tracks)
                }
                sequences.append(sequence)
        
        df = pd.DataFrame(sequences)
        print(f"Created {len(df):,} training sequences")
        
        return df
    
    def train_val_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data ensuring no playlist overlap between sets"""
        print("\nSplitting data into train/val/test...")
        
        # Get unique playlist IDs
        unique_playlists = df['playlist_id'].unique()
        np.random.seed(self.config.RANDOM_SEED)
        np.random.shuffle(unique_playlists)
        
        # Calculate split points
        n_total = len(unique_playlists)
        n_train = int(n_total * self.config.TRAIN_RATIO)
        n_val = int(n_total * self.config.VAL_RATIO)
        
        train_playlists = set(unique_playlists[:n_train])
        val_playlists = set(unique_playlists[n_train:n_train + n_val])
        test_playlists = set(unique_playlists[n_train + n_val:])
        
        # Split dataframe
        train_df = df[df['playlist_id'].isin(train_playlists)]
        val_df = df[df['playlist_id'].isin(val_playlists)]
        test_df = df[df['playlist_id'].isin(test_playlists)]
        
        print(f"Train: {len(train_df):,} sequences ({len(train_playlists):,} playlists)")
        print(f"Val: {len(val_df):,} sequences ({len(val_playlists):,} playlists)")
        print(f"Test: {len(test_df):,} sequences ({len(test_playlists):,} playlists)")
        
        return train_df, val_df, test_df
    
    def save_to_arrow(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                      test_df: pd.DataFrame, output_dir: Path):
        """Save datasets as Arrow/Parquet files for efficient loading"""
        print("\nSaving to Arrow format...")
        
        # Convert list columns to string (Arrow doesn't handle Python lists directly)
        for df in [train_df, val_df, test_df]:
            df['history_str'] = df['history'].apply(lambda x: '|'.join(x))
        
        # Save as Parquet (uses Arrow under the hood)
        train_df.to_parquet(output_dir / "data" / "train.parquet", index=False)
        val_df.to_parquet(output_dir / "data" / "val.parquet", index=False)
        test_df.to_parquet(output_dir / "data" / "test.parquet", index=False)
        
        print(f"✓ Saved to {output_dir / 'data'}/")
        print("  - train.parquet")
        print("  - val.parquet")
        print("  - test.parquet")

