import re
from typing import Dict, List
import pandas as pd
from collections import Counter
import numpy as np

class ContextExtractor:
    """
    Extract context labels from playlist names using keyword matching.
    Implements Week 4 of the roadmap: context conditioning.
    """
    
    def __init__(self):
        # Define context keywords (expandable)
        self.context_keywords = {
            'workout': ['workout', 'gym', 'running', 'exercise', 'fitness', 'cardio', 
                       'training', 'run', 'jog', 'crossfit', 'weights', 'lifting'],
            'study': ['study', 'studying', 'focus', 'concentration', 'reading', 
                     'homework', 'exam', 'library', 'work', 'coding', 'programming'],
            'party': ['party', 'dance', 'club', 'clubbing', 'nightclub', 'edm', 
                     'pregame', 'drinking', 'night out', 'rave', 'festival'],
            'chill': ['chill', 'relax', 'relaxing', 'calm', 'mellow', 'lounge', 
                     'ambient', 'peaceful', 'zen', 'meditation', 'yoga'],
            'sleep': ['sleep', 'sleeping', 'bedtime', 'night', 'lullaby', 'rest'],
            'sad': ['sad', 'depressed', 'crying', 'heartbreak', 'breakup', 'lonely', 
                   'melancholy', 'emo', 'feelings'],
            'happy': ['happy', 'upbeat', 'cheerful', 'good vibes', 'positive', 
                     'feel good', 'sunshine', 'summer', 'fun']
        }
        
        # Add 'other' for unclassified playlists
        self.contexts = list(self.context_keywords.keys()) + ['other']
        self.context_to_idx = {ctx: idx for idx, ctx in enumerate(self.contexts)}
        self.idx_to_context = {idx: ctx for ctx, idx in self.context_to_idx.items()}
        
    def extract_context(self, playlist_name: str) -> str:
        """
        Extract context from a single playlist name.
        
        Args:
            playlist_name: Playlist name string
            
        Returns:
            Context label (e.g., 'workout', 'study', 'other')
        """
        if not playlist_name or pd.isna(playlist_name):
            return 'other'
        
        # Normalize: lowercase, remove special chars
        name_lower = playlist_name.lower()
        name_clean = re.sub(r'[^a-z0-9\s]', ' ', name_lower)
        
        # Check each context's keywords
        context_scores = {ctx: 0 for ctx in self.context_keywords.keys()}
        
        for context, keywords in self.context_keywords.items():
            for keyword in keywords:
                # Use word boundaries to avoid partial matches
                if re.search(r'\b' + re.escape(keyword) + r'\b', name_clean):
                    context_scores[context] += 1
        
        # Get context with highest score
        max_score = max(context_scores.values())
        if max_score > 0:
            return max(context_scores.items(), key=lambda x: x[1])[0]
        
        return 'other'
    
    def add_context_to_sequences(self, sequences_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add context column to sequences DataFrame.
        
        Args:
            sequences_df: DataFrame with 'playlist_name' column
            
        Returns:
            DataFrame with added 'context' and 'context_idx' columns
        """
        print("\n[Context Extraction] Extracting contexts from playlist names...")
        
        sequences_df['context'] = sequences_df['playlist_name'].apply(self.extract_context)
        sequences_df['context_idx'] = sequences_df['context'].map(self.context_to_idx)
        
        # Print statistics
        context_counts = sequences_df['context'].value_counts()
        print(f"\nContext distribution:")
        for ctx, count in context_counts.items():
            pct = count / len(sequences_df) * 100
            print(f"  {ctx:12s}: {count:7,} ({pct:5.1f}%)")
        
        return sequences_df
    
    def get_context_statistics(self, sequences_df: pd.DataFrame) -> Dict:
        """
        Get detailed statistics about context distribution.
        Used for analysis in Week 5.
        """
        stats = {}
        
        for context in self.contexts:
            ctx_df = sequences_df[sequences_df['context'] == context]
            if len(ctx_df) > 0:
                stats[context] = {
                    'count': len(ctx_df),
                    'avg_playlist_length': ctx_df['playlist_length'].mean(),
                    'avg_history_length': ctx_df['history'].apply(len).mean()
                }
        
        return stats
    
    def get_num_contexts(self) -> int:
        """Return number of context categories (for model architecture)"""
        return len(self.contexts)


# Example usage and testing
if __name__ == "__main__":
    extractor = ContextExtractor()
    
    # Test cases
    test_names = [
        "Workout Mix 2024",
        "Study Beats - Focus Music",
        "Party Hits 🎉",
        "Chill Vibes for Relaxing",
        "Late Night Sad Songs",
        "My Favorite Songs",  # Should be 'other'
        "Running Playlist",
        "Club Bangers",
        "Yoga & Meditation"
    ]
    
    print("Testing context extraction:")
    print("=" * 60)
    for name in test_names:
        context = extractor.extract_context(name)
        print(f"{name:35s} -> {context}")