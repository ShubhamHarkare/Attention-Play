from sklearn.preprocessing import StandardScaler
from config import Config
from collections import defaultdict,Counter
from typing import List,Dict
import pandas as pd
from tqdm import tqdm
class FeatureEngineer:
    '''
    Create embeddings and transition matrices for baseline models
    '''

    def __init__(self,config:Config):
        self.config = config
        self.scaler = StandardScaler()
        self.track_embeddings = {}
        self.transition_matrix = defaultdict(Counter)


    def create_audio_embeddings(self, audio_df: pd.DataFrame, uri_to_id: Dict) -> Dict:
        """Create normalized audio feature embeddings"""
        print("\n[4/7] Creating audio feature embeddings...")
        
        # Select and normalize audio features
        feature_cols = [col for col in self.config.AUDIO_FEATURES 
                       if col in audio_df.columns]
        print(f"Using features: {feature_cols}")
        
        X = audio_df[feature_cols].fillna(audio_df[feature_cols].median())
        X_normalized = self.scaler.fit_transform(X)
        
        # Create embedding dictionary (URI -> normalized feature vector)
        embeddings = {}
        for idx, row in audio_df.iterrows():
            track_id = row['id']
            uri = f"spotify:track:{track_id}"
            embeddings[uri] = X_normalized[audio_df.index.get_loc(idx)]
        
        print(f"Created embeddings for {len(embeddings):,} tracks")
        self.track_embeddings = embeddings
        return embeddings
    

    def build_transition_matrix(self, playlists: List[Dict]) -> Dict:
        """Build song-to-song transition matrix from playlists"""
        print("\n[5/7] Building transition matrix...")
        
        transition_counts = defaultdict(Counter)
        
        for playlist in tqdm(playlists, desc="Processing transitions"):
            tracks = [t['track_uri'] for t in playlist['tracks']]
            for i in range(len(tracks) - 1):
                current_track = tracks[i]
                next_track = tracks[i + 1]
                transition_counts[current_track][next_track] += 1
        
        # Convert counts to probabilities
        transition_probs = {}
        for source_track, targets in transition_counts.items():
            total = sum(targets.values())
            transition_probs[source_track] = {
                target: count / total 
                for target, count in targets.items()
            }
        
        print(f"Built transitions for {len(transition_probs):,} tracks")
        self.transition_matrix = transition_probs
        return transition_probs
