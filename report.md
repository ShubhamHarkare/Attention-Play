# Week 1: Complete Line-by-Line Code Explanation
## AttentionPlay ML Pipeline - Every Single Line Explained

---

## Part 1: Imports and Setup

```python
import json
```
**What it does:** Imports Python's JSON library  
**Why we need it:** Spotify Million Playlist Dataset comes as JSON files (JavaScript Object Notation - a text format for storing data)  
**Example:** `{"name": "workout", "tracks": [...]}`

```python
import pandas as pd
```
**What it does:** Imports pandas, a library for working with tabular data  
**Why we need it:** We'll load audio features CSV and manipulate data in tables (DataFrames)  
**Think of it as:** Excel but in Python

```python
import numpy as np
```
**What it does:** Imports NumPy (Numerical Python)  
**Why we need it:** Fast mathematical operations on arrays/matrices  
**Example:** Calculating averages, normalizing features, array operations

```python
from pathlib import Path
```
**What it does:** Modern way to handle file paths  
**Why we need it:** Cross-platform file paths (works on Windows/Mac/Linux)  
**Example:** `Path("data/file.json")` instead of string concatenation

```python
from collections import defaultdict, Counter
```
**What it does:** Special dictionary types  
- `defaultdict`: Dictionary that creates missing keys automatically  
- `Counter`: Dictionary that counts occurrences  
**Why we need it:**  
- `defaultdict` for transition matrices (if song A→B transition doesn't exist, create it)
- `Counter` for counting how many times each song appears

```python
from typing import List, Dict, Tuple
```
**What it does:** Type hints for better code documentation  
**Why we need it:** Makes code clearer - tells you what type of data functions expect  
**Example:** `def process(songs: List[str]) -> Dict` means "this function takes a list of strings and returns a dictionary"

```python
import pickle
```
**What it does:** Serialize Python objects to files  
**Why we need it:** Save trained models and data structures to disk  
**Example:** Save your trained KNN model so you don't retrain it every time

```python
from tqdm import tqdm
```
**What it does:** Progress bars for loops  
**Why we need it:** When processing 100K playlists, you want to see progress  
**Example:** Shows `Processing: 45%|████████░░░░| 45000/100000`

```python
import warnings
warnings.filterwarnings('ignore')
```
**What it does:** Suppresses warning messages  
**Why we need it:** scikit-learn and pandas can be noisy with non-critical warnings  
**Note:** In production, you'd want to see warnings, but for learning it's cleaner

```python
from sklearn.neighbors import NearestNeighbors
```
**What it does:** KNN algorithm from scikit-learn  
**Why we need it:** First baseline model - finds songs similar to a given song  
**How it works:** Measures distance between song feature vectors

```python
from sklearn.preprocessing import StandardScaler
```
**What it does:** Normalizes data to have mean=0, std=1  
**Why we need it:** Audio features have different scales (tempo: 60-200, energy: 0-1)  
**Example:** Before: `tempo=120, energy=0.8` → After: `tempo=0.5, energy=0.3` (normalized)

```python
from sklearn.metrics.pairwise import cosine_similarity
```
**What it does:** Calculates similarity between vectors using cosine angle  
**Why we need it:** Second baseline - measures how "similar" two songs are  
**Formula:** Similarity = cos(θ) between two vectors, ranges from -1 to 1

```python
from sklearn.model_selection import train_test_split
```
**What it does:** Splits data into train/validation/test sets  
**Why we need it:** Machine learning golden rule - never test on training data!  
**Standard split:** 70% train, 15% validation, 15% test

```python
import matplotlib.pyplot as plt
import seaborn as sns
```
**What it does:** Plotting libraries  
**Why we need it:** Create visualizations for your poster  
- `matplotlib`: Low-level plotting (lines, bars, scatter)
- `seaborn`: High-level, prettier plots (heatmaps, distributions)

```python
import pyarrow as pa
import pyarrow.parquet as pq
```
**What it does:** Apache Arrow - columnar data format  
**Why we need it:** Save processed data efficiently for Transformer training  
**Why not CSV?** Parquet is 10x faster to load and takes less space

---

## Part 2: Configuration Class

```python
class Config:
    """Central configuration for the pipeline"""
```
**What it does:** Python class to hold ALL configuration in one place  
**Why we need it:** Change parameters in one location instead of hunting through code  
**Best practice:** Always centralize configuration

```python
    MPD_DIR = Path("data/spotify_million_playlist_dataset/data")
```
**What it does:** Path to your 1000 JSON files  
**YOU MUST CHANGE THIS:** Point to where YOUR JSON files are located  
**Example:** If your files are in `/Users/shubham/data/mpd/data`, change this path

```python
    AUDIO_FEATURES_PATH = Path("data/spotify_tracks_dataset.csv")
```
**What it does:** Path to Spotify audio features CSV from HuggingFace  
**Content:** Contains danceability, energy, valence, etc. for millions of songs  
**File size:** ~150MB

```python
    OUTPUT_DIR = Path("output")
```
**What it does:** Where all results will be saved  
**Creates:** `output/models/`, `output/visualizations/`, `output/data/`, `output/metrics/`

```python
    NUM_FILES_TO_PROCESS = 100
```
**What it does:** How many JSON files to load (out of 1000)  
**Why 100?** Balance between data size and processing time  
**Math:** 100 files × 1000 playlists/file = 100,000 playlists  
**For testing:** Start with `NUM_FILES_TO_PROCESS = 1` (1,000 playlists, ~5 min)

```python
    MIN_PLAYLIST_LENGTH = 5
```
**What it does:** Filter out playlists with fewer than 5 songs  
**Why?** Too short to learn sequential patterns  
**Example:** Playlist with 2 songs doesn't tell us much about transitions

```python
    MAX_PLAYLIST_LENGTH = 200
```
**What it does:** Filter out extremely long playlists  
**Why?** Computational efficiency and outliers (some playlists have 1000+ songs)  
**Transformer limitation:** Most models struggle with very long sequences

```python
    MIN_SONG_FREQUENCY = 10
```
**What it does:** Song must appear in at least 10 playlists to be included  
**Why?** Rare songs don't have enough data to learn patterns  
**Cold start problem:** We can't recommend songs no one has heard  
**Trade-off:** Higher value = smaller vocabulary but better learning

```python
    AUDIO_FEATURES = ['danceability', 'energy', 'loudness', ...]
```
**What it does:** List of Spotify audio features to use  
**What each means:**
- `danceability` (0-1): How suitable for dancing (rhythm, tempo, beat)
- `energy` (0-1): Intensity and activity (loud, fast, noisy = high energy)
- `loudness` (dB): Overall volume (-60 to 0 dB)
- `speechiness` (0-1): Presence of spoken words (podcasts = high)
- `acousticness` (0-1): Confidence song is acoustic (no electric instruments)
- `instrumentalness` (0-1): Predicts no vocals ("ooh" and "aah" count as instrumental)
- `liveness` (0-1): Detects audience presence (live recording)
- `valence` (0-1): Musical positivity (happy/cheerful vs sad/angry)
- `tempo` (BPM): Speed in beats per minute (60-200 typical)
- `duration_ms`: Length in milliseconds

```python
    KNN_NEIGHBORS = 20
```
**What it does:** How many similar songs to consider in KNN  
**Why 20?** Common default that balances specificity and diversity  
**Lower (5):** Very similar songs, might miss good recommendations  
**Higher (50):** More diverse but less similar

```python
    RANDOM_SEED = 42
```
**What it does:** Fixes randomness for reproducibility  
**Why 42?** Convention from "Hitchhiker's Guide to the Galaxy"  
**Importance:** Same seed = same results every time (critical for debugging)

```python
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
```
**What it does:** Data split percentages (must sum to 1.0)  
**Why this split?**
- **Train (70%):** Learn patterns
- **Validation (15%):** Tune hyperparameters, early stopping
- **Test (15%):** Final evaluation (NEVER look at this until the end)

```python
    TOP_K_VALUES = [1, 5, 10, 20]
```
**What it does:** Evaluate if correct song is in top-K predictions  
**Example:** If correct song is #7 in your predictions:
- Top-1: ❌ (not in top 1)
- Top-5: ❌ (not in top 5)
- Top-10: ✅ (IS in top 10)
- Top-20: ✅ (IS in top 20)

```python
    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (self.OUTPUT_DIR / "models").mkdir(exist_ok=True)
        ...
```
**What it does:** Constructor - runs when you create `Config()` object  
**Creates folders:** Makes all output directories if they don't exist  
- `parents=True`: Create parent folders too
- `exist_ok=True`: Don't error if folder already exists

---

## Part 3: SpotifyDataLoader Class

```python
class SpotifyDataLoader:
    """Handles loading and merging Spotify MPD and audio features"""
```
**What it does:** Responsible for loading data from disk into memory  
**Why a class?** Keeps all loading logic organized and reusable

```python
    def __init__(self, config: Config):
        self.config = config
        self.playlists = []
        self.track_uri_to_id = {}
        self.audio_features_df = None
```
**What it does:** Initialize the data loader  
**Variables:**
- `config`: Store configuration object
- `playlists`: Will hold all loaded playlists
- `track_uri_to_id`: Maps Spotify URIs to IDs (e.g., "spotify:track:ABC123" → "ABC123")
- `audio_features_df`: Will hold the audio features DataFrame

```python
    def load_mpd_files(self) -> List[Dict]:
```
**What it does:** Load JSON files from disk  
**Returns:** List of playlist dictionaries  
**Type hint:** `-> List[Dict]` means "returns a list of dictionaries"

```python
        json_files = sorted(list(self.config.MPD_DIR.glob("*.json")))
```
**What it does:** Find all JSON files in the directory  
**Breakdown:**
- `.glob("*.json")`: Find all files ending in `.json`
- `list()`: Convert generator to list
- `sorted()`: Sort alphabetically (mpd.slice.0-999.json, mpd.slice.1000-1999.json, ...)

```python
        if len(json_files) == 0:
            raise FileNotFoundError(f"No JSON files found in {self.config.MPD_DIR}")
```
**What it does:** Error handling - stop if no files found  
**Why?** Fail fast with clear message instead of cryptic error later  
**Common mistake:** Wrong path in Config

```python
        files_to_load = json_files[:self.config.NUM_FILES_TO_PROCESS]
```
**What it does:** Take only first N files  
**Example:** If `NUM_FILES_TO_PROCESS = 100`, take files 0-99  
**Python slice:** `[:100]` means "from start to position 100"

```python
        all_playlists = []
        for json_file in tqdm(files_to_load, desc="Loading JSON files"):
```
**What it does:** Loop through files with progress bar  
**tqdm:** Shows `Loading JSON files: 45%|████████░░░░| 45/100`

```python
            with open(json_file, 'r') as f:
                data = json.load(f)
                all_playlists.extend(data['playlists'])
```
**What it does:** Load JSON and extract playlists  
**Breakdown:**
- `with open(...)`: Opens file, automatically closes when done (even if error)
- `json.load(f)`: Parse JSON text into Python dictionary
- `data['playlists']`: Access the 'playlists' key (each file has structure `{"info": {...}, "playlists": [...]}`)
- `.extend()`: Add all playlists from this file to main list

```python
        print(f"Loaded {len(all_playlists):,} playlists")
```
**What it does:** Print total count with comma formatting  
**Example:** `Loaded 100,000 playlists` (not `Loaded 100000 playlists`)

---

## Part 4: Audio Features Loading

```python
    def load_audio_features(self) -> pd.DataFrame:
```
**What it does:** Load the audio features CSV  
**Returns:** pandas DataFrame (think Excel spreadsheet)

```python
        if not self.config.AUDIO_FEATURES_PATH.exists():
            raise FileNotFoundError(...)
```
**What it does:** Check if CSV file exists before trying to load  
**Common mistake:** Forgot to download the CSV from HuggingFace

```python
        df = pd.read_csv(self.config.AUDIO_FEATURES_PATH)
```
**What it does:** Load CSV into DataFrame  
**Result:** Table with columns like `id, danceability, energy, valence, ...`  
**Size:** ~2 million rows (songs)

```python
        print(f"Columns: {df.columns.tolist()}")
```
**What it does:** Print column names to verify format  
**Why?** Different datasets have different column names (`id` vs `track_id`)  
**Debugging:** If code fails later, check if column names match what we expect

---

## Part 5: Preprocessing and Merging

```python
    def preprocess_data(self, playlists: List[Dict], audio_df: pd.DataFrame):
```
**What it does:** Clean data and merge playlists with audio features  
**This is the MOST IMPORTANT function** - where all the magic happens

```python
        filtered_playlists = [
            p for p in playlists 
            if self.config.MIN_PLAYLIST_LENGTH <= len(p['tracks']) <= self.config.MAX_PLAYLIST_LENGTH
        ]
```
**What it does:** List comprehension - filter playlists by length  
**Reads as:** "Keep playlist p if it has between 5 and 200 tracks"  
**Example:** 
- Playlist with 3 songs: ❌ removed
- Playlist with 50 songs: ✅ kept
- Playlist with 500 songs: ❌ removed

```python
        track_counter = Counter()
        for playlist in filtered_playlists:
            for track in playlist['tracks']:
                track_counter[track['track_uri']] += 1
```
**What it does:** Count how many playlists each song appears in  
**Example output:** `Counter({'spotify:track:ABC': 523, 'spotify:track:XYZ': 45, ...})`  
**Why?** Need to know which songs are popular enough to include

```python
        frequent_tracks = {uri for uri, count in track_counter.items() 
                          if count >= self.config.MIN_SONG_FREQUENCY}
```
**What it does:** Set comprehension - keep only songs appearing 10+ times  
**Set:** Unordered collection of unique items (fast lookup)  
**Why set not list?** Checking `if uri in frequent_tracks` is O(1) for sets, O(n) for lists

```python
        if 'track_id' in audio_df.columns and 'id' not in audio_df.columns:
            audio_df = audio_df.rename(columns={'track_id': 'id'})
```
**What it does:** Standardize column names  
**Problem:** Some datasets use `track_id`, others use `id`  
**Solution:** Rename to consistent `id` column

```python
        uri_to_id = {}
        for _, row in tqdm(audio_df.iterrows(), total=len(audio_df), desc="Mapping URIs"):
```
**What it does:** Build mapping from Spotify URI to track ID  
**Example:** `"spotify:track:6rqhFgbbKwnb9MLmUQDhG6"` → `"6rqhFgbbKwnb9MLmUQDhG6"`  
**Why?** MPD uses URIs, audio features CSV uses IDs - need to connect them

```python
            if 'id' in row and pd.notna(row['id']):
                uri = f"spotify:track:{row['id']}"
                uri_to_id[uri] = row['id']
```
**What it does:** Create URI from ID and add to mapping  
**Check:** `pd.notna()` ensures ID isn't missing (NaN)  
**f-string:** `f"spotify:track:{row['id']}"` inserts ID into URI format

```python
        final_playlists = []
        for playlist in tqdm(filtered_playlists, desc="Filtering tracks"):
            filtered_tracks = [
                t for t in playlist['tracks'] 
                if t['track_uri'] in uri_to_id and t['track_uri'] in frequent_tracks
            ]
```
**What it does:** Keep only tracks that:
1. Have audio features (`in uri_to_id`)
2. Are frequent enough (`in frequent_tracks`)

**Example:** Playlist has 50 songs, but only 35 have audio features and are frequent → keep those 35

```python
            if len(filtered_tracks) >= self.config.MIN_PLAYLIST_LENGTH:
                playlist['tracks'] = filtered_tracks
                final_playlists.append(playlist)
```
**What it does:** After filtering tracks, check if playlist is still long enough  
**Example:** Started with 50 tracks, filtered to 3 tracks → discard entire playlist

---

## Part 6: Feature Engineering

```python
class FeatureEngineer:
    def create_audio_embeddings(self, audio_df: pd.DataFrame, uri_to_id: Dict) -> Dict:
```
**What it does:** Convert raw audio features into normalized vectors  
**Embedding:** Dense vector representation of a song (all features in one array)

```python
        feature_cols = [col for col in self.config.AUDIO_FEATURES 
                       if col in audio_df.columns]
```
**What it does:** Select only features that exist in the dataset  
**Why?** Some datasets might be missing certain features

```python
        X = audio_df[feature_cols].fillna(audio_df[feature_cols].median())
```
**What it does:** Handle missing values by filling with median  
**Why median not mean?** Median is robust to outliers  
**Example:** If `energy` is missing for a song, use median energy from all songs

```python
        X_normalized = self.scaler.fit_transform(X)
```
**What it does:** Normalize features to mean=0, std=1  
**Why?** So all features have equal importance  
**Example:**
- Before: `tempo=120, energy=0.8` (different scales)
- After: `tempo=0.5, energy=0.3` (same scale)

**Formula:** `(value - mean) / std`

```python
        embeddings = {}
        for idx, row in audio_df.iterrows():
            track_id = row['id']
            uri = f"spotify:track:{track_id}"
            embeddings[uri] = X_normalized[audio_df.index.get_loc(idx)]
```
**What it does:** Create dictionary mapping URI → normalized feature vector  
**Result:** `{"spotify:track:ABC": [0.5, -0.3, 1.2, ...], ...}`

---

## Part 7: Transition Matrix

```python
    def build_transition_matrix(self, playlists: List[Dict]) -> Dict:
```
**What it does:** Count how often song B follows song A  
**Use case:** KNN baseline can use "if people played A, they often play B next"

```python
        transition_counts = defaultdict(Counter)
```
**What it does:** Nested dictionary structure  
**Structure:** `{song_A: {song_B: 15, song_C: 8}, song_B: {song_D: 23}}`  
**Means:** After song_A, people played song_B 15 times and song_C 8 times

```python
        for playlist in tqdm(playlists, desc="Processing transitions"):
            tracks = [t['track_uri'] for t in playlist['tracks']]
            for i in range(len(tracks) - 1):
                current_track = tracks[i]
                next_track = tracks[i + 1]
                transition_counts[current_track][next_track] += 1
```
**What it does:** For each adjacent pair of songs, increment counter  
**Example:** Playlist = [A, B, C, D]  
- Transitions: A→B, B→C, C→D  
- Each transition increments its counter

```python
        transition_probs = {}
        for source_track, targets in transition_counts.items():
            total = sum(targets.values())
            transition_probs[source_track] = {
                target: count / total 
                for target, count in targets.items()
            }
```
**What it does:** Convert counts to probabilities  
**Example:** Song A was followed by:
- Song B: 15 times
- Song C: 5 times  
- Total: 20 times  
**Probabilities:** B: 15/20 = 0.75, C: 5/20 = 0.25

---

## Part 8: KNN Baseline

```python
class BaselineModels:
    def train_knn(self, embeddings: Dict) -> NearestNeighbors:
```
**What it does:** Train K-Nearest Neighbors model  
**KNN concept:** Find songs with similar feature vectors

```python
        track_uris = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[uri] for uri in track_uris])
```
**What it does:** Convert dictionary to matrix  
**Why?** scikit-learn KNN needs 2D numpy array  
**Shape:** (num_songs, num_features) e.g., (50000, 10)

```python
        self.track_index_to_uri = track_uris
        self.track_uri_to_index = {uri: idx for idx, uri in enumerate(track_uris)}
```
**What it does:** Create bidirectional mapping  
**Why?** KNN returns indices, we need to convert back to URIs  
**Example:** Index 42 → "spotify:track:ABC"

```python
        knn = NearestNeighbors(
            n_neighbors=self.config.KNN_NEIGHBORS + 1,
            metric='cosine',
            algorithm='brute',
            n_jobs=-1
        )
```
**What it does:** Configure KNN model  
**Parameters:**
- `n_neighbors`: Find 21 neighbors (we'll exclude the query song itself, leaving 20)
- `metric='cosine'`: Use cosine similarity (angle between vectors)
- `algorithm='brute'`: Check all songs (slow but accurate)
- `n_jobs=-1`: Use all CPU cores

```python
        knn.fit(embedding_matrix)
```
**What it does:** "Train" KNN (actually just stores the matrix)  
**KNN is lazy:** Doesn't learn anything, just remembers all training examples

---

## Part 9: Getting Recommendations

```python
    def get_knn_recommendations(self, track_uri: str, k: int = 10) -> List[Tuple[str, float]]:
```
**What it does:** Given a song, find K most similar songs  
**Returns:** List of (song_uri, similarity_score) tuples

```python
        if track_uri not in self.track_uri_to_index:
            return []
```
**What it does:** Handle unknown songs  
**Cold start:** If we haven't seen this song, can't recommend

```python
        track_idx = self.track_uri_to_index[track_uri]
        distances, indices = self.knn_model.kneighbors(
            self.embedding_matrix[track_idx].reshape(1, -1),
            n_neighbors=k + 1
        )
```
**What it does:** Find nearest neighbors  
**Input:** Reshape to (1, num_features) - KNN expects 2D array  
**Output:**
- `distances`: How far each neighbor is
- `indices`: Which songs are the neighbors

```python
        recommendations = []
        for dist, idx in zip(distances[0][1:], indices[0][1:]):
            recommended_uri = self.track_index_to_uri[idx]
            similarity = 1 - dist
            recommendations.append((recommended_uri, similarity))
```
**What it does:** Convert indices to URIs and distances to similarities  
**[1:]**: Skip first result (the query song itself)  
**similarity = 1 - dist**: Convert distance to similarity (closer = more similar)

---

## Part 10: Transformer Data Preparation

```python
class TransformerDataPreparation:
    def create_sequences(self, playlists: List[Dict], uri_to_id: Dict) -> pd.DataFrame:
```
**What it does:** Convert playlists into training examples  
**Key insight:** For Transformer, each training example is (history, target)

```python
        sequences = []
        for playlist in tqdm(playlists, desc="Creating sequences"):
            tracks = [t['track_uri'] for t in playlist['tracks']]
```
**What it does:** Extract track URIs in order

```python
            for i in range(1, len(tracks)):
                sequence = {
                    'playlist_id': playlist['pid'],
                    'playlist_name': playlist_name,
                    'history': tracks[:i],
                    'target': tracks[i],
                    'position': i,
                    'playlist_length': len(tracks)
                }
                sequences.append(sequence)
```
**What it does:** Sliding window approach  
**Example:** Playlist = [A, B, C, D]  
Creates 3 training examples:
1. history=[A], target=B
2. history=[A, B], target=C
3. history=[A, B, C], target=D

**Why this works?** Transformer learns: "given songs A,B,C, what comes next?"

```python
        df = pd.DataFrame(sequences)
```
**What it does:** Convert list of dictionaries to DataFrame  
**Result:** Table with columns [playlist_id, history, target, position, ...]

---

## Part 11: Train/Val/Test Split

```python
    def train_val_test_split(self, df: pd.DataFrame):
        unique_playlists = df['playlist_id'].unique()
        np.random.shuffle(unique_playlists)
```
**What it does:** Get all unique playlist IDs and randomize order  
**CRITICAL:** We split by PLAYLIST, not by sequence  
**Why?** If sequences from same playlist are in train and test, we're cheating!

```python
        n_total = len(unique_playlists)
        n_train = int(n_total * self.config.TRAIN_RATIO)
        n_val = int(n_total * self.config.VAL_RATIO)
```
**What it does:** Calculate split sizes  
**Example:** 10,000 playlists → 7,000 train, 1,500 val, 1,500 test

```python
        train_playlists = set(unique_playlists[:n_train])
        val_playlists = set(unique_playlists[n_train:n_train + n_val])
        test_playlists = set(unique_playlists[n_train + n_val:])
```
**What it does:** Partition playlist IDs into three non-overlapping sets  
**Sets ensure:** No playlist appears in multiple splits

```python
        train_df = df[df['playlist_id'].isin(train_playlists)]
```
**What it does:** Filter DataFrame to only sequences from train playlists  
**Boolean indexing:** `df[condition]` keeps only rows where condition is True

---

## Part 12: Arrow Export

```python
    def save_to_arrow(self, train_df, val_df, test_df, output_dir):
        for df in [train_df, val_df, test_df]:
            df['history_str'] = df['history'].apply(lambda x: '|'.join(x))
```
**What it does:** Convert list to string (Arrow doesn't handle Python lists)  
**Example:** `['song1', 'song2']` → `"song1|song2"`  
**Why |?** Delimiter that won't appear in Spotify URIs

```python
        train_df.to_parquet(output_dir / "data" / "train.parquet", index=False)
```
**What it does:** Save as Parquet (columnar format using Arrow)  
**Benefits:**
- 10x faster loading than CSV
- Smaller file size (compression)
- Preserves data types

---

## Part 13: Evaluation

```python
class Evaluator:
    def evaluate_top_k_accuracy(self, predictions: List[str], target: str, k: int):
        return 1 if target in predictions[:k] else 0
```
**What it does:** Check if target song is in top-K predictions  
**Example:** predictions=[A, B, C, D, E], target=C, k=5  
- `predictions[:5]` = [A, B, C, D, E]
- C is in this list → return 1 (correct!)

```python
    def evaluate_baseline(self, model, test_df, model_name):
        top_k_hits = {k: 0 for k in self.config.TOP_K_VALUES}
        total = 0
```
**What it does:** Initialize counters for each K value  
**Structure:** `{1: 0, 5: 0, 10: 0, 20: 0}`

```python
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            if len(row['history']) == 0:
                continue
            
            last_track = row['history'][-1]
            target = row['target']
```
**What it does:** For each test example, get last song in history and target  
**Strategy:** Use last song to generate recommendations

```python
            if model_name == "KNN":
                recs = model.get_knn_recommendations(last_track, k=max(self.config.TOP_K_VALUES))
```
**What it does:** Get top-20 recommendations (max K value)  
**Why max?** Get enough for all K values (1, 5, 10, 20)

```python
            predicted_tracks = [uri for uri, score in recs]
            
            for k in self.config.TOP_K_VALUES:
                top_k_hits[k] += self.evaluate_top_k_accuracy(predicted_tracks, target, k)
```
**What it does:** Check each K value and increment counter if correct

```python
        accurac