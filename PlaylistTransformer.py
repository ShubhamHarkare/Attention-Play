import torch
import torch.nn as nn
import math
from PositionalEncoding import PositionalEncoding
class PlaylistTransformer(nn.Module):

    def __init__(self,num_items,d_model = 256,nhead = 8,num_layer = 4,dropout = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_items,d_model,padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model=d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer=self.encoder_layer,num_layers=num_layer)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model,num_items)

        self.d_model = d_model




    def forward(self,x,src_key_padding_mask = None):
        emb = self.embedding(x) * math.sqrt(self.d_model)
        emb = self.pos_encoder(emb)
        emb = self.dropout(emb)


        if src_key_padding_mask is not None:
            src_key_padding_mask = ~src_key_padding_mask

        out = self.transformer(emb, src_key_padding_mask)
        out = self.dropout(out)


        last_output = out[:,-1,:]
        logits = self.fc(last_output)



    def get_attention_weights(self, x, src_key_padding_mask=None):
        """Extract attention weights for visualization"""
        emb = self.embedding(x) * math.sqrt(self.d_model)
        emb = self.pos_encoder(emb)
        
        if src_key_padding_mask is not None:
            src_key_padding_mask = ~src_key_padding_mask
        
        # Store attention weights
        attention_weights = []
        
        def hook_fn(module, input, output):
            # output[1] contains attention weights
            if len(output) > 1 and output[1] is not None:
                attention_weights.append(output[1].detach())
        
        
        handles = []
        for layer in self.transformer.layers:
            handle = layer.self_attn.register_forward_hook(hook_fn)
            handles.append(handle)
        
        # Forward pass
        with torch.no_grad():
            _ = self.transformer(emb, src_key_padding_mask=src_key_padding_mask)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return attention_weights
