import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# For ML models
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# For visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# For Arrow file export
import pyarrow as pa
import pyarrow.parquet as pq



class Config:
    '''
    This class holds all the configurations like 
    input file path, output file path and hyperparameters for the models
    '''
    MPD_DIR = Path('archive-2/data')
    OUTPUT_DIR = Path('output')
    HF_DATASET_PATH = "hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv"
    NUM_FILES_TO_PROCESS = 100  #? How many files should we process in the dataset
    MIN_PLAYLIST_LENGTH = 5 #? Lower bound for number of songs in the playlist
    MAX_PLAYLIST_LENGTH = 200  #? Upper bound for number of songs in the playlist
    MIN_SONG_FREQUENCY = 10  #? How many times that song should be present in all playlists in order to be used.

    AUDIO_FEATURES = ['danceability', 'energy', 'loudness', 'speechiness', 
                     'acousticness', 'instrumentalness', 'liveness', 
                     'valence', 'tempo', 'duration_ms']
    
    KNN_NEIGHBORS = 20 #? Hyperparameter for NearestNeigbor classifier
    RANDOM_SEED = 42 #? For reproduciblity of code
    
    # Data splits
    TRAIN_RATIO = 0.70 #? 70% of data for training
    VAL_RATIO = 0.15 #? 15% Data for validation
    TEST_RATIO = 0.15 #? 15% data for testing
    
    # Evaluation
    TOP_K_VALUES = [1, 5, 10, 20]
    
    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (self.OUTPUT_DIR / "models").mkdir(exist_ok=True)
        (self.OUTPUT_DIR / "visualizations").mkdir(exist_ok=True)
        (self.OUTPUT_DIR / "data").mkdir(exist_ok=True)
        (self.OUTPUT_DIR / "metrics").mkdir(exist_ok=True)
