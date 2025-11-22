import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU4Rec(nn.Module):
    """GRU optimized for your data profile"""
    
    def __init__(self, num_items, embedding_dim=256, hidden_dim=512, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout * 0.5)  # Less dropout on embeddings
        
        # 3-layer GRU (your data can support this)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            num_layers=3, 
            dropout=dropout
        )
        
        self.gru_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Attention over all timesteps (CRITICAL!)
        self.attention_score = nn.Linear(hidden_dim, 1)
        
        # Two-layer prediction head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc1_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, num_items)
        
        # Initialize weights
    #     self._init_weights()
    
    # def _init_weights(self):
    #     # Better initialization
    #     for name, param in self.named_parameters():
    #         if 'weight' in name:
    #             if 'gru' in name:
    #                 nn.init.orthogonal_(param)
    #             else:
    #                 nn.init.xavier_uniform_(param)
    #         elif 'bias' in name:
    #             nn.init.zeros_(param)
    
    def forward(self, x, mask=None):
        # Embedding
        emb = self.embedding(x)
        emb = self.embedding_dropout(emb)
        
        # GRU
        gru_out, _ = self.gru(emb)  # (batch, seq_len, hidden)
        gru_out = self.gru_dropout(gru_out)
        
        # Attention mechanism (use ALL history, not just last)
        attention_scores = self.attention_score(gru_out)  # (batch, seq_len, 1)
        
        if mask is not None:
            # Mask padding positions
            attention_scores = attention_scores.masked_fill(
                ~mask.unsqueeze(-1), float('-inf')
            )
        
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum of all timesteps
        context = (gru_out * attention_weights).sum(dim=1)  # (batch, hidden)
        
        # Normalize
        context = self.layer_norm(context)
        
        # Two-layer head
        hidden = F.relu(self.fc1(context))
        hidden = self.fc1_dropout(hidden)
        logits = self.fc2(hidden)
        
        return logits
    

    