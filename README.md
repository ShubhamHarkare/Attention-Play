# AttentionPlay — Deep Learning Playlist Recommendation System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A research-oriented deep learning pipeline for playlist continuation and music recommendation using the Spotify Million Playlist Dataset. This project implements and compares multiple architectures from traditional baselines to state-of-the-art Transformer models.

## 📊 Project Overview
AttentionPlay tackles the sequential playlist continuation problem: given a user's listening history, predict the next track they'll enjoy. This is a challenging task with applications in music streaming platforms like Spotify, Apple Music, and YouTube Music. AttentionPlay considers the tracks in a playlist as sequences which helps in implementation of the transformer architecture

### *Key Features*
- **Multiple Model Architecture:** `KNearestNeighbors`,`CosineSimilarity`,`Gated-Recurrent Networks (GRU)` & `Transformer Architecture`
- **Large-Scale Dataset:** Used the [Spotify Million Playlist Dataset](https://www.kaggle.com/datasets/himanshuwagh/spotify-million) and [spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
- **Context-Aware Recommendation:** Fine-Tuned the model based on the mood of the playlist.

## 🎯 Results
<table>
<tr>
  <th>Model</th>
  <th>Top-1</th>
  <th>Top-5</th>
  <th>Top-10</th>
  <th>Top-20</th>
</tr>
<tr>
  <td>KNearestNeighbor</td>
  <td>0.18%</td>
  <td>0.96%</td>
  <td>1.85%</td>
  <td>3.49%</td>
</tr>
<tr>
  <td>CosineSimilarity</td>
  <td>0.17%</td>
  <td>0.95%</td>
  <td>1.87%</td>
  <td>3.35%</td>
</tr>
<tr>
  <td>GRU</td>
  <td>9.73%</td>
  <td>23.61%</td>
  <td>33.09%</td>
  <td>44.73%</td>
</tr>
<tr>
  <td><b>Transformer</b></td>
  <td><b>10.95%</b></td>
  <td><b>25.24%</b></td>
  <td><b>34.40%</b></td>
  <td><b>45.55%</b></td>
</tr>
</table>

### 🗝️ *Key-Findings*
1. Transformer achieves best performance across all Top-K metrics
2. ~18x improvement over baseline (Top-10: 34.3% vs 1.85%)
3. Attention mechanisms are critical for capturing playlist context
4. Deep models significantly outperform traditional ML baselines

[![Baseline Model](output/visualizations/baseline_comparison.png)]()
[![GRU and Transformer](output/visualizations/learning_curves_detailed.png)]()


## 🏋️‍♂️ Training Details
*Dataset Statistics:*
<table>
<tr>
  <th>Metric</th>
  <th>Value</th>
</tr>

<tr>
  <th>Total Playlists</th>
  <td>41,243 tracks</td>
</tr>
<tr>
  <th>Unique Playlists</th>
  <td>5,899 tracks</td>
</tr>
<tr>
  <th>Training Sequences</th>
  <td>3,414,231</td>
</tr>

<tr>
  <th>Validation Sequences</th>
  <td>729,960</td>
</tr>
<tr>
  <th>Test Sequences</th>
  <td>729,960</td>
</tr>
<tr>
  <th>Average Playlist Length</th>
  <td>12.6 songs</td>
</tr>
<tr>
  <th>Median Sequences</th>
  <td>7 songs</td>
</tr>

</table>

## 🤖 Model Architecture
1. *Transformer Configuarion*
  - Embedding Dimension: 256
  - Hidden Dimension: 512
  - Attention Heads: 8
  - Encoder Layers: 4
  - Dropout: 0.3
  - Model Parameters: 7,044,365
2. *Training HyperParameters*
  - Optimizer: AdamW (lr=0.0005, weight_decay=0.01)
  - Scheduler: Cosine Annealing with 3-epoch warmup
  - Loss Function: Cross-Entropy with 0.1 label smoothing
  - Batch Size: 128
  - Max Sequence Length: 50


## 📶 Visualizations

[![Visuals](output/visualizations/audio_features.png)]()

[![Visuals](output/visualizations/data_distribution.png)]()
[![Visuals](output/visualizations/dataset_statistics.png)]()
[![Visuals](output/visualizations/feature_importance.png)]()
[![Visuals](output/visualizations/training_efficiency.png)]()
[![Visuals](output/visualizations/transformer_test_results.png)]()


## 📘 Usage Examples

**Training Model**
```
from TraningConfig import TrainingConfig
from PlaylistTransformer import PlaylistTransformer
from Trainer import Trainer

# Initialize config
config = TrainingConfig()

# Create model
model = PlaylistTransformer(
    num_items=50000 + 2,  # vocab + PAD + UNK
    d_model=256,
    nhead=8,
    num_layer=4,
    dropout=0.3
)

# Train
trainer = Trainer(model, train_loader, val_loader, config, "Transformer")
results = trainer.train()
```

**Generating Recommendation**
```
from BaselineModels import BaselineModels

# Load trained KNN model
baselines = BaselineModels(config)
baselines.train_knn(embeddings)

# Get recommendations
seed_song = "spotify:track:6rqhFgbbKwnb9MLmUQDhG6"  # Shape of You
recommendations = baselines.get_knn_recommendations(seed_song, k=10)

# Display results
for track_uri, similarity in recommendations:
    print(f"{track_uri}: {similarity:.3f}")
```

**Context-Aware Recommendation**
```
from ContextAwareTransformer import ContextAwareTransformer

# Create context-aware model
model = ContextAwareTransformer(
    num_items=50000 + 2,
    num_contexts=8,  # workout, study, party, chill, sleep, sad, happy, other
    d_model=256
)

# Generate playlist with context
seed_songs = [song1, song2, song3]
context = 0  # workout
playlist = model.generate_playlist(
    seed_songs, 
    context_idx=context,
    max_length=15,
    temperature=1.0,
    top_k=50
)
```