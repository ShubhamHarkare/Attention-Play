import torch
import torch.nn as nn
import math
from PositionalEncoding import PositionalEncoding

class PlaylistTransformer(nn.Module):
    def __init__(self, num_items, d_model=256, nhead=8, num_layer=4, dropout=0.1, use_masks=True):
        super().__init__()
        self.embedding = nn.Embedding(num_items, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, 
            num_layers=num_layer
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_items)
        self.d_model = d_model
        self.use_masks = use_masks
    
    def generate_square_subsequent_mask(self, sz, device):
        """Generate causal mask for autoregressive training"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask
    
    def forward(self, x, src_key_padding_mask=None):
        batch_size, seq_len = x.size()
        
        emb = self.embedding(x) * math.sqrt(self.d_model)
        emb = self.pos_encoder(emb)
        emb = self.dropout(emb)
        
        # Check if we're on MPS and disable masks if needed
        is_mps = x.device.type == 'mps'
        
        if self.use_masks and not is_mps:
            # Use masks only if not on MPS
            attn_mask = self.generate_square_subsequent_mask(seq_len, x.device)
            
            if src_key_padding_mask is not None:
                src_key_padding_mask = ~src_key_padding_mask
            
            out = self.transformer(
                emb, 
                mask=attn_mask,
                src_key_padding_mask=src_key_padding_mask
            )
        else:
            # No masks for MPS compatibility
            out = self.transformer(emb)
        
        out = self.dropout(out)
        last_output = out[:, -1, :]
        logits = self.fc(last_output)
        
        return logits
    
    def get_attention_weights(self, x, src_key_padding_mask=None):
        """Extract attention weights for visualization"""
        seq_len = x.size(1)
        is_mps = x.device.type == 'mps'
        
        emb = self.embedding(x) * math.sqrt(self.d_model)
        emb = self.pos_encoder(emb)
        
        if self.use_masks and not is_mps:
            attn_mask = self.generate_square_subsequent_mask(seq_len, x.device)
            if src_key_padding_mask is not None:
                src_key_padding_mask = ~src_key_padding_mask
        else:
            attn_mask = None
            src_key_padding_mask = None
        
        attention_weights = []
        
        def hook_fn(module, input, output):
            if len(output) > 1 and output[1] is not None:
                attention_weights.append(output[1].detach())
        
        handles = []
        for layer in self.transformer.layers:
            handle = layer.self_attn.register_forward_hook(hook_fn)
            handles.append(handle)
        
        with torch.no_grad():
            if attn_mask is not None:
                _ = self.transformer(
                    emb, 
                    mask=attn_mask,
                    src_key_padding_mask=src_key_padding_mask
                )
            else:
                _ = self.transformer(emb)
        
        for handle in handles:
            handle.remove()
        
        return attention_weights