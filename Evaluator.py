from collections import defaultdict
from config import Config
from typing import List
import pandas as pd
from tqdm import tqdm


class Evaluator:
    """Evaluate baseline models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = defaultdict(dict)
        
    def evaluate_top_k_accuracy(self, predictions: List[str], target: str, k: int) -> int:
        """Check if target is in top-k predictions"""
        return 1 if target in predictions[:k] else 0
    
    def evaluate_baseline(self, model, test_df: pd.DataFrame, model_name: str):
        """Evaluate a baseline model on test set"""
        print(f"\nEvaluating {model_name}...")
        
        top_k_hits = {k: 0 for k in self.config.TOP_K_VALUES}
        total = 0
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"{model_name}"):
            if len(row['history']) == 0:
                continue
            
            last_track = row['history'][-1]
            target = row['target']
            
            # Get recommendations based on model type
            if model_name == "KNN":
                recs = model.get_knn_recommendations(last_track, k=max(self.config.TOP_K_VALUES))
            elif model_name == "Cosine":
                recs = model.get_cosine_recommendations(last_track, k=max(self.config.TOP_K_VALUES))
            else:
                continue
            
            if not recs:
                continue
            
            predicted_tracks = [uri for uri, score in recs]
            
            # Calculate top-k accuracy
            for k in self.config.TOP_K_VALUES:
                top_k_hits[k] += self.evaluate_top_k_accuracy(predicted_tracks, target, k)
            
            total += 1
            
            # Sample limit for faster evaluation during development
            if total >= 10000:  # Evaluate on 10K samples
                break
        
        # Calculate accuracies
        accuracies = {k: (hits / total * 100) for k, hits in top_k_hits.items()}
        self.results[model_name] = accuracies
        
        print(f"\n{model_name} Results (evaluated on {total:,} samples):")
        for k, acc in accuracies.items():
            print(f"  Top-{k} Accuracy: {acc:.2f}%")
        
        return accuracies

