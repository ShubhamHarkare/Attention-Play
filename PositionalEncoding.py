import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Improved positional encoding with proper shape handling
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Proper shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        # Add positional encoding: broadcast across batch dimension
        x = x + self.pe[:seq_len].unsqueeze(0)
        return self.dropout(x)


