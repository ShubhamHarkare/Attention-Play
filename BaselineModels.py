from config import Config
from typing import Dict,List,Tuple
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class BaselineModels:
    """Traditional ML baselines: KNN and Cosine Similarity"""
    
    def __init__(self, config: Config):
        self.config = config
        self.knn_model = None
        self.track_index_to_uri = []
        self.track_uri_to_index = {}
        self.embedding_matrix = None
        
    def train_knn(self, embeddings: Dict) -> NearestNeighbors:
        """Train KNN model on audio embeddings"""
        print("\n[6/7] Training KNN baseline...")
        
        # Create matrix of embeddings
        track_uris = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[uri] for uri in track_uris])
        
        # Build index mappings
        self.track_index_to_uri = track_uris
        self.track_uri_to_index = {uri: idx for idx, uri in enumerate(track_uris)}
        self.embedding_matrix = embedding_matrix
        
        # Train KNN
        knn = NearestNeighbors(
            n_neighbors=self.config.KNN_NEIGHBORS + 1,  # +1 to exclude self
            algorithm='brute',
            n_jobs=-1
        )
        knn.fit(embedding_matrix)
        
        self.knn_model = knn
        print(f"Trained KNN with {len(track_uris):,} tracks")
        return knn
    
    def get_knn_recommendations(self, track_uri: str, k: int = 10) -> List[Tuple[str, float]]:
        """Get top-k similar tracks using KNN"""
        if track_uri not in self.track_uri_to_index:
            return []
        
        track_idx = self.track_uri_to_index[track_uri]
        distances, indices = self.knn_model.kneighbors(
            self.embedding_matrix[track_idx].reshape(1, -1),
            n_neighbors=k + 1
        )
        
        # Exclude the query track itself
        recommendations = []
        for dist, idx in zip(distances[0][1:], indices[0][1:]):
            recommended_uri = self.track_index_to_uri[idx]
            similarity = 1 - dist  # Convert distance to similarity
            recommendations.append((recommended_uri, similarity))
        
        return recommendations
    
    def get_cosine_recommendations(self, track_uri: str, k: int = 10) -> List[Tuple[str, float]]:
        """Get top-k similar tracks using cosine similarity"""
        if track_uri not in self.track_uri_to_index:
            return []
        
        track_idx = self.track_uri_to_index[track_uri]
        query_embedding = self.embedding_matrix[track_idx].reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding, self.embedding_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][1:k+1]  # Exclude self
        
        recommendations = [
            (self.track_index_to_uri[idx], similarities[idx])
            for idx in top_indices
        ]
        
        return recommendations