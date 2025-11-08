# AttentionPlay Project Documentation

## Project Overview
AttentionPlay is a deep learning system for music playlist recommendation that uses both baseline models and transformer-based architectures. The project is split into two main pipelines:
1. Data preprocessing and baseline models (main.py)
2. Deep learning models training (main_2.py)

## Pipeline 1: Data Preprocessing and Baseline Models (main.py)

### Key Components:

1. **Data Loading and Preprocessing**
   - Loads Spotify Million Playlist Dataset (MPD) JSON files
   - Processes audio features from Spotify API
   - Filters playlists based on length constraints
   - Merges playlist data with audio features

2. **Feature Engineering**
   - Creates normalized audio embeddings
   - Builds transition matrices for sequential patterns
   - Processes audio features including:
     - Danceability
     - Energy
     - Loudness
     - Valence
     - Tempo

3. **Baseline Models**
   - K-Nearest Neighbors (KNN) for similar song recommendation
   - Cosine similarity-based recommendations
   - Transition probability-based suggestions

## Pipeline 2: Deep Learning Training (main_2.py)

### Key Components:

1. **Data Preparation**
   - Converts playlists into sequential training examples
   - Implements train/validation/test splitting
   - Creates efficient data loading pipelines using PyTorch DataLoader

2. **Model Architecture**
   - Transformer model with:
     - Multi-head self-attention
     - Positional encoding
     - Dropout regularization
     - Configurable embedding dimensions

3. **Training Process**
   - Implements batch processing
   - Uses cross-entropy loss
   - Supports early stopping
   - Handles device compatibility (CPU/GPU/MPS)

### Training Configuration
```python
Key Parameters:
- Embedding Dimension: 256
- Number of Attention Heads: 8
- Number of Transformer Layers: 4
- Dropout Rate: 0.1
- Batch Size: Configurable
- Learning Rate: Adaptive
```

## Data Flow

1. **Input Processing**
   ```
   Raw Playlists → Filtered Sequences → Training Examples
   ```

2. **Model Pipeline**
   ```
   Audio Features → Embeddings → Transformer → Predictions
   ```

## Evaluation Metrics

1. **Top-K Accuracy**
   - Measures if correct song is in top K predictions
   - K values: 1, 5, 10, 20

2. **Loss Tracking**
   - Training loss
   - Validation loss
   - Early stopping based on validation metrics

## Visualization Features

1. **Training Metrics**
   - Loss curves
   - Accuracy progression
   - Model comparison plots

2. **Attention Analysis**
   - Attention weight visualizations
   - Song relationship heatmaps

## Project Structure
```
Attention-Play/
├── main.py              # Preprocessing and baseline pipeline
├── main_2.py            # Deep learning training pipeline
├── PlaylistTransformer.py  # Transformer model implementation
├── models/              # Saved model checkpoints
├── data/               # Processed dataset files
├── metrics/            # Evaluation results
└── visualizations/     # Generated plots and figures
```

## Running the Project

1. **Preprocessing Pipeline**
   ```bash
   python main.py  # Processes data and trains baselines
   ```

2. **Deep Learning Training**
   ```bash
   python main_2.py  # Trains transformer model
   ```

## Results

The project compares performance between:
- KNN Baseline
- GRU4Rec (Sequential Model)
- Transformer (Attention-based)

Results are saved in:
- JSON metrics files
- Visualization plots
- Model checkpoints

## Future Improvements

1. **Model Enhancements**
   - Fine-tuning transformer architecture
   - Exploring different attention mechanisms
   - Implementing cross-attention for user preferences

2. **Data Processing**
   - Additional feature engineering
   - More sophisticated data augmentation
   - Handling cold-start problems

## Conclusion

AttentionPlay demonstrates the effectiveness of attention-based models for playlist continuation, showing improvements over traditional baselines while maintaining practical deployment capabilities.