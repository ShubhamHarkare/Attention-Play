import torch
from torch.utils.data import Dataset
import pandas as pd


class PlaylistDataset(Dataset):
    '''
    Dataset generator for the playlists dataset
    '''

    def __init__(self,data,vocab_size,max_sequence_length = 50):
        super().__init__()
        self.df = pd.read_parquet(data)
        self.vocab = vocab_size
        self.max_sequence_length = max_sequence_length


        # Reserve indices for special tokens
        self.PAD_IDX = 0
        self.UNK_IDX = 1

        # Build vocabulary with offset to avoid overlap
        self.track_to_idx = {track: idx + 2 for idx, track in enumerate(self.vocab)}
        self.idx_to_track = {idx: track for track, idx in self.track_to_idx.items()}


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row =  self.df.iloc[idx]
    

        if isinstance(row['history'],str):
            history = row['history'].split('|') if row['history'] else []
        else:
            history = row['history']
        target = row['target']

        history_indices = [
            self.track_to_idx.get(track, self.UNK_IDX) 
            for track in history
        ]
        
        target_idx = self.track_to_idx.get(target, self.UNK_IDX)
        
        # Truncate if too long
        if len(history_indices) > self.max_sequence_length:
            history_indices = history_indices[-self.max_sequence_length:]
        
        return {
            'history': history_indices,
            'target': target_idx,
            'seq_length': len(history_indices)
        }
    
    def collate_fn(self, batch):
        """Custom collate function for padding"""
        # Find max length in batch
        max_len = max([item['seq_length'] for item in batch])
        
        # Pad sequences
        padded_histories = []
        targets = []
        masks = []
        
        for item in batch:
            history = item['history']
            padding_length = max_len - len(history)
            
            # Pad history
            padded_history = history + [self.PAD_IDX] * padding_length
            padded_histories.append(padded_history)
            
            # Create mask (1 for real tokens, 0 for padding)
            mask = [1] * len(history) + [0] * padding_length
            masks.append(mask)
            
            targets.append(item['target'])
        
        return {
            'input_ids': torch.LongTensor(padded_histories),
            'targets': torch.LongTensor(targets),
            'attention_mask': torch.BoolTensor(masks)
        }

    

