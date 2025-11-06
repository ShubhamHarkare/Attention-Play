# AttentionPlay — Playlist Recommendation ML Pipeline

AttentionPlay is a research-oriented ML pipeline for playlist continuation and item recommendation. It includes data loading & preprocessing, baseline models based on audio features and transition matrices, and deep learning models (GRU4Rec and Transformer) with evaluation and visualization tools.

## Contents (key files)
- Project entry points:
  - [main.py](main.py) — end-to-end pipeline: preprocessing, baselines, Transformer data prep, evaluation, and visualizations.
  - [main_2.py](main_2.py) — focused deep-learning training pipeline (GRU4Rec & Transformer).
- Configuration:
  - [`Config`](config.py) — central configuration for the baseline pipeline ([config.py](config.py)).
  - [`TrainingConfig`](TraningConfig.py) — training-specific defaults for deep models ([TraningConfig.py](TraningConfig.py)).
- Data & loaders:
  - [`SpotifyDataLoader`](dataloader.py) — load and preprocess MPD JSONs and audio features ([dataloader.py](dataloader.py)).
  - [PlayListDataset.py](PlayListDataset.py) — dataset class used by the PyTorch data loaders.
  - [TransformerDataPreparation.py](TransformerDataPreparation.py) — sequence generation for Transformer training.
- Feature engineering & baselines:
  - [`FeatureEngineer.create_audio_embeddings`](FeatureEngineer.py) — create normalized audio embeddings from audio CSVs ([FeatureEngineer.py](FeatureEngineer.py)).
  - [BaselineModels.py](BaselineModels.py) — KNN / similarity-based baselines and helpers.
  - [check_vocab.py](check_vocab.py) — dataset vocabulary diagnostics.
- Models & training:
  - [`GRU4Rec`](GRU4Rec.py) — GRU-based sequential recommender ([GRU4Rec.py](GRU4Rec.py)).
  - [`PlaylistTransformer`](PlaylistTransformer.py) — Transformer model architecture ([PlaylistTransformer.py](PlaylistTransformer.py)).
  - [`Trainer`](Trainer.py) — training loop used for deep models ([Trainer.py](Trainer.py)).
- Evaluation & visualization:
  - [`Evaluator`](Evaluator.py) — baseline & deep model evaluation utilities ([Evaluator.py](Evaluator.py)).
  - [Visualizer.py](Visualizer.py) — baseline visualization helpers.
  - [`DeepLearningVisualizer.plot_training_curves`](DeepLearningVisualizer.py) — plots training/validation curves and attention visualization ([DeepLearningVisualizer.py](DeepLearningVisualizer.py)).
  - [DeepLearningVisualizer.py](DeepLearningVisualizer.py)
- Utilities & extras:
  - [FeatureEngineer.py](FeatureEngineer.py), [diagnostic.py](diagnostic.py), [report.md](report.md)
- Outputs:
  - `output/` — saved data, metrics, models, and visualizations
  - `output/models/`, `output/data/`, `output/metrics/`, `output/visualizations/`

## Quickstart

1. Create and activate a Python 3.8+ environment and install dependencies (example):
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
(If `requirements.txt` is not present, install typical packages: numpy, pandas, torch, scikit-learn, matplotlib, seaborn, tqdm, pyarrow.)

2. Configure paths and parameters in [`config.py`](config.py) or [`TraningConfig.py`](TraningConfig.py). Important settings:
   - `MPD_DIR` and `AUDIO_FEATURES_PATH` in [`Config`](config.py)
   - `OUTPUT_DIR`, `NUM_FILES_TO_PROCESS`, `MIN_PLAYLIST_LENGTH` in [`Config`](config.py)
   - training hyperparameters in [`TraningConfig.py`](TraningConfig.py)

3. Run the baseline + Transformer data prep pipeline:
```bash
python main.py
```

4. Run the deep learning training pipeline (GRU4Rec + Transformer):
```bash
python main_2.py
```

## Typical workflow
1. Use [`SpotifyDataLoader`](dataloader.py) to load raw MPD JSONs and audio feature CSVs.
2. Create audio embeddings with [`FeatureEngineer.create_audio_embeddings`](FeatureEngineer.py) and build transition matrices.
3. Train baseline models in [BaselineModels.py](BaselineModels.py) and evaluate with [`Evaluator`](Evaluator.py).
4. Prepare sequence data with [TransformerDataPreparation.py](TransformerDataPreparation.py).
5. Train deep models using [`Trainer`](Trainer.py) with [`GRU4Rec`](GRU4Rec.py) and [`PlaylistTransformer`](PlaylistTransformer.py).
6. Visualize results with [Visualizer.py](Visualizer.py) and [`DeepLearningVisualizer`](DeepLearningVisualizer.py).

## Outputs
- Preprocessed Arrow files: `output/data/*.parquet`
- Saved baseline models: `output/models/baselines.pkl`
- Deep model checkpoints: `output/models/*_best.pt`
- Metrics and summaries: `output/metrics/summary.json`, `output/metrics/deep_learning_results.json`
- Visualizations: `output/visualizations/*.png`

## Notes & tips
- Inspect `report.md` for a line-by-line project walkthrough and configuration guidance ([report.md](report.md)).
- Use [check_vocab.py](check_vocab.py) and diagnostic prints in [main_2.py](main_2.py) to verify vocabulary and UNK rates before training.
- GPU vs Apple MPS: DeepLearningVisualizer checks `torch.mps` availability; configure device behavior in model training if needed ([DeepLearningVisualizer.py](DeepLearningVisualizer.py)).


