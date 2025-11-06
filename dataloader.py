from config import Config
from typing import List,Dict
from tqdm import tqdm
import json
import pandas as pd
from collections import Counter
class SpotifyDataLoader:
    '''
    Handles loading dataset and merging spotify MPD audio features
    '''
    def __init__(self,config:Config):
        self.config = config #? To store the configuration object
        self.playlist = [] #? To store all the playlists
        self.track_uri_to_id = {} #? map for TrackURI -> IDs
        self.audio_features_df = None #? Features dataframe


    def load_mpd_files(self) -> List[Dict]:
        '''
        Loading the Spotify MPD Dataset
        '''
        print('\n[1/7] Loading the spotify MPD Dataset')
        
        json_files = sorted(list(self.config.MPD_DIR.glob("*.json"))) #* Find all the files that are of the type ".json"

        if len(json_files) == 0:
            #! Raise error if file is not found for preprocessing
            raise FileNotFoundError(f'No JSON files are found in {self.config.MPD_DIR}')
        
        files_to_load = json_files[:self.config.NUM_FILES_TO_PROCESS]

        print(f'Found items {len(json_files)}, loading {len(files_to_load)} files..')

        all_playlists = []

        for json_file in tqdm(files_to_load, desc="Loading JSON files"):
            with open(json_file, 'r') as f: #TODO: Open each and every file in the path given and load it.
                data = json.load(f)
                all_playlists.extend(data['playlists']) #TODO: Extend the playlist in the all_playlists 
        
        print(f"Loaded {len(all_playlists):,} playlists")
        return all_playlists
    

    def load_audio_features(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.HF_DATASET_PATH)

        print(f"Loaded {len(df):,} tracks with audio features")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    

    def preprocess_data(self,playlists: List[Dict],audio_df: pd.DataFrame):
        '''
        Clean and merge playlist data with audio features
        '''

        filtered_playlists = [p for p in playlists if self.config.MIN_PLAYLIST_LENGTH <= len(p['tracks']) <= self.config.MAX_PLAYLIST_LENGTH]

        print(f"After length filtering: {len(filtered_playlists):,} playlists")


        # Extract all unique track URIs and count frequencies
        track_counter = Counter()
        for playlist in filtered_playlists:
            for track in playlist['tracks']:
                track_counter[track['track_uri']] += 1


        frequent_tracks = {uri for uri, count in track_counter.items() 
                          if count >= self.config.MIN_SONG_FREQUENCY}
        print(f"Tracks appearing in {self.config.MIN_SONG_FREQUENCY}+ playlists: {len(frequent_tracks):,}")


        print("Building track URI to ID mapping...")
        if 'track_id' in audio_df.columns and 'id' not in audio_df.columns:
            audio_df = audio_df.rename(columns={'track_id': 'id'})
        
        uri_to_id = {}
        for _, row in tqdm(audio_df.iterrows(), total=len(audio_df), desc="Mapping URIs"):
            if 'id' in row and pd.notna(row['id']):
                uri = f"spotify:track:{row['id']}"
                uri_to_id[uri] = row['id']
        
        print(f"Successfully mapped {len(uri_to_id):,} track URIs")

        final_playlists = []
        for playlist in tqdm(filtered_playlists, desc="Filtering tracks"):
            filtered_tracks = [
                t for t in playlist['tracks'] 
                if t['track_uri'] in uri_to_id and t['track_uri'] in frequent_tracks
            ]
            
            if len(filtered_tracks) >= self.config.MIN_PLAYLIST_LENGTH:
                playlist['tracks'] = filtered_tracks
                final_playlists.append(playlist)
        
        print(f"Final dataset: {len(final_playlists):,} playlists")
        
        # Build final track vocabulary
        final_track_uris = set()
        for playlist in final_playlists:
            for track in playlist['tracks']:
                final_track_uris.add(track['track_uri'])
        
        print(f"Final vocabulary: {len(final_track_uris):,} unique tracks")
        
        # Filter audio features to only include tracks in vocabulary
        final_track_ids = [uri_to_id[uri] for uri in final_track_uris if uri in uri_to_id]
        audio_df_filtered = audio_df[audio_df['id'].isin(final_track_ids)].copy()
        
        print(f"Audio features for {len(audio_df_filtered):,} tracks")
        
        self.playlists = final_playlists
        self.track_uri_to_id = uri_to_id
        self.audio_features_df = audio_df_filtered
        
        return final_playlists, audio_df_filtered, uri_to_id