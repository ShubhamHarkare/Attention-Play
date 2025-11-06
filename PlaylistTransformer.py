import torch
import torch.nn as nn
import math
from PositionalEncoding import PositionalEncoding

class PlaylistTransformer(nn.Module):
    """
    Enhanced Transformer for playlist recommendation with improved architecture
    """
    def __init__(self, num_items, d_model=256, nhead=8, num_layer=4, dropout=0.1):
        super().__init__()

        # Embedding layer with padding
        self.embedding = nn.Embedding(num_items, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        # Enhanced Transformer with layer normalization first
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN architecture for better training
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layer,
            norm=nn.LayerNorm(d_model)  # Final layer norm
        )

        # Improved prediction head with two-layer MLP
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()  # GELU works better than ReLU for transformers
        self.fc2 = nn.Linear(d_model, num_items)

        self.d_model = d_model

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Better weight initialization"""
        # Embedding initialization
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        nn.init.constant_(self.embedding.weight[0], 0)  # PAD token

        # Linear layer initialization
        for module in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)




    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: Input tensor (batch, seq_len)
            src_key_padding_mask: Mask for padding (batch, seq_len), True for valid tokens
        Returns:
            logits: Output predictions (batch, num_items)
        """
        # Embedding with scaling
        emb = self.embedding(x) * math.sqrt(self.d_model)

        # Add positional encoding (dropout is inside pos_encoder now)
        emb = self.pos_encoder(emb)

        # Invert mask: PyTorch expects True for positions to IGNORE
        if src_key_padding_mask is not None:
            src_key_padding_mask = ~src_key_padding_mask

        # Transformer encoding
        out = self.transformer(emb, src_key_padding_mask=src_key_padding_mask)

        # Use last non-padding token for prediction
        last_output = out[:, -1, :]

        # Two-layer prediction head
        hidden = self.fc1(last_output)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)

        return logits

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
