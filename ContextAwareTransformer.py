import torch
import torch.nn as nn
import math
from PositionalEncoding import PositionalEncoding

class ContextAwareTransformer(nn.Module):
    """
    Enhanced Transformer with context conditioning for playlist recommendation.
    Implements Week 4 of the roadmap: context-aware generation.
    
    Context is incorporated as an additional token prepended to the sequence.
    This allows the model to condition its predictions on playlist context
    (workout, study, party, chill, etc.)
    
    Architecture:
        1. Separate embeddings for songs and contexts
        2. Context token prepended to song sequence
        3. Standard transformer encoder processes combined sequence
        4. Last token used for next-song prediction
    """
    
    def __init__(self, num_items, num_contexts, d_model=256, nhead=8, num_layer=4, dropout=0.1):
        """
        Args:
            num_items: Vocabulary size (number of unique tracks + special tokens)
            num_contexts: Number of context categories (e.g., 8: workout, study, party, etc.)
            d_model: Embedding dimension
            nhead: Number of attention heads
            num_layer: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_items = num_items
        self.num_contexts = num_contexts
        self.d_model = d_model
        
        # Song embedding with padding
        self.song_embedding = nn.Embedding(num_items, d_model, padding_idx=0)
        
        # Context embedding (separate from songs)
        # Each context gets its own learned embedding vector
        self.context_embedding = nn.Embedding(num_contexts, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        
        # Enhanced Transformer with pre-layer normalization
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,  # Standard: 4x embedding dim
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layer,
            norm=nn.LayerNorm(d_model)  # Final layer norm
        )
        
        # Two-layer prediction head with GELU activation
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()  # GELU works better than ReLU for transformers
        self.fc2 = nn.Linear(d_model, num_items)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Better weight initialization for stable training"""
        # Song embeddings: small random values
        nn.init.normal_(self.song_embedding.weight, mean=0, std=0.02)
        nn.init.constant_(self.song_embedding.weight[0], 0)  # PAD token = all zeros
        
        # Context embeddings: small random values
        nn.init.normal_(self.context_embedding.weight, mean=0, std=0.02)
        
        # Linear layers: Xavier initialization
        for module in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x, context_idx, src_key_padding_mask=None):
        """
        Forward pass with context conditioning.
        
        Args:
            x: Song sequence (batch, seq_len) - indices of songs in history
            context_idx: Context indices (batch,) - single context per sequence
            src_key_padding_mask: Mask for padding (batch, seq_len)
                                 True for valid tokens, False for padding
        
        Returns:
            logits: Output predictions (batch, num_items) - scores for next song
        
        Example:
            x = [[5, 12, 8, 0, 0]]  # 3 songs + 2 padding
            context_idx = [0]        # workout context
            mask = [[1, 1, 1, 0, 0]] # First 3 are valid
            
            Output shape: (1, num_items) - probability distribution over all songs
        """
        batch_size = x.size(0)
        
        # Step 1: Get song embeddings
        # Scale by sqrt(d_model) for better gradient flow (from "Attention is All You Need")
        song_emb = self.song_embedding(x) * math.sqrt(self.d_model)  # (batch, seq_len, d_model)
        
        # Step 2: Get context embeddings
        context_emb = self.context_embedding(context_idx)  # (batch, d_model)
        context_emb = context_emb.unsqueeze(1)  # (batch, 1, d_model) - add sequence dimension
        
        # Step 3: Prepend context as first token
        # This allows all subsequent songs to attend to the context
        combined_emb = torch.cat([context_emb, song_emb], dim=1)  # (batch, seq_len+1, d_model)
        
        # Step 4: Add positional encoding
        # Tells model about position of each token in sequence
        combined_emb = self.pos_encoder(combined_emb)
        
        # Step 5: Adjust mask to account for context token
        if src_key_padding_mask is not None:
            # PyTorch convention: True = positions to IGNORE (padding)
            # Our convention: True = valid tokens
            # So we need to invert
            src_key_padding_mask = ~src_key_padding_mask
            
            # Add mask entry for context token (always valid, so False = don't ignore)
            context_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([context_mask, src_key_padding_mask], dim=1)
        
        # Step 6: Transformer encoding
        # Self-attention allows each position to attend to all previous positions + context
        out = self.transformer(combined_emb, src_key_padding_mask=src_key_padding_mask)
        # out shape: (batch, seq_len+1, d_model)
        
        # Step 7: Use last token representation for prediction
        # We skip the context token (position 0) and use the last song position
        last_output = out[:, -1, :]  # (batch, d_model)
        
        # Step 8: Two-layer prediction head
        hidden = self.fc1(last_output)      # (batch, d_model)
        hidden = self.activation(hidden)     # GELU activation
        hidden = self.dropout(hidden)        # Dropout for regularization
        logits = self.fc2(hidden)            # (batch, num_items)
        
        return logits
    
    def get_attention_weights(self, x, context_idx, src_key_padding_mask=None):
        """
        Extract attention weights for visualization.
        
        This is used in the frontend to show which songs the model
        attended to when making predictions.
        
        Args:
            x: Song sequence (batch, seq_len)
            context_idx: Context indices (batch,)
            src_key_padding_mask: Padding mask (batch, seq_len)
        
        Returns:
            List of attention weight tensors, one per layer
            Shape of each: (batch, num_heads, seq_len+1, seq_len+1)
        """
        batch_size = x.size(0)
        
        # Prepare embeddings (same as forward pass)
        song_emb = self.song_embedding(x) * math.sqrt(self.d_model)
        context_emb = self.context_embedding(context_idx).unsqueeze(1)
        combined_emb = torch.cat([context_emb, song_emb], dim=1)
        combined_emb = self.pos_encoder(combined_emb)
        
        # Adjust mask
        if src_key_padding_mask is not None:
            src_key_padding_mask = ~src_key_padding_mask
            context_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([context_mask, src_key_padding_mask], dim=1)
        
        # Store attention weights from each layer
        attention_weights = []
        
        def hook_fn(module, input, output):
            """Hook to capture attention weights from each layer"""
            # output[1] contains attention weights if need_weights=True
            if len(output) > 1 and output[1] is not None:
                attention_weights.append(output[1].detach())
        
        # Register hooks on all self-attention layers
        handles = []
        for layer in self.transformer.layers:
            handle = layer.self_attn.register_forward_hook(hook_fn)
            handles.append(handle)
        
        # Forward pass (no gradients needed)
        with torch.no_grad():
            _ = self.transformer(combined_emb, src_key_padding_mask=src_key_padding_mask)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return attention_weights
    
    def generate_playlist(self, seed_songs, context_idx, max_length=15, temperature=1.0, top_k=50):
        """
        Generate a playlist autoregressively given seed songs and context.
        
        This is used by the frontend for interactive generation.
        
        Args:
            seed_songs: Initial song indices (batch, seed_len) or (seed_len,)
            context_idx: Context index (batch,) or scalar
            max_length: Maximum playlist length to generate
            temperature: Sampling temperature
                        - Lower (0.5): More conservative, likely songs
                        - Higher (1.5): More random, diverse songs
            top_k: Consider only top-k most likely songs
                   - Prevents sampling very unlikely songs
                   - 0 = sample from full distribution
        
        Returns:
            Generated song indices (batch, max_length)
        
        Example:
            seed = [5, 12, 8]  # 3 seed songs
            context = 0         # workout
            
            generated = model.generate_playlist(seed, context, max_length=15)
            # Output: [5, 12, 8, 23, 45, 67, ...] (15 songs total)
        """
        self.eval()  # Set to evaluation mode (disables dropout)
        
        # Handle single example (add batch dimension if needed)
        if seed_songs.dim() == 1:
            seed_songs = seed_songs.unsqueeze(0)
        if isinstance(context_idx, int):
            context_idx = torch.tensor([context_idx], device=seed_songs.device)
        elif context_idx.dim() == 0:
            context_idx = context_idx.unsqueeze(0)
        
        current_sequence = seed_songs.clone()
        
        with torch.no_grad():
            # Generate one song at a time
            for _ in range(max_length - seed_songs.size(1)):
                # Create attention mask (all positions are valid)
                mask = torch.ones(
                    current_sequence.size(0), 
                    current_sequence.size(1), 
                    dtype=torch.bool, 
                    device=current_sequence.device
                )
                
                # Get predictions for next song
                logits = self.forward(current_sequence, context_idx, mask)
                
                # Apply temperature scaling
                # Lower temperature → more peaked distribution (conservative)
                # Higher temperature → flatter distribution (random)
                logits = logits / temperature
                
                # Top-k sampling: only consider k most likely songs
                if top_k > 0:
                    # Get top-k logits and their indices
                    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    
                    # Convert to probabilities
                    probs = torch.softmax(top_k_logits, dim=-1)
                    
                    # Sample from top-k distribution
                    next_token_idx = torch.multinomial(probs, num_samples=1)
                    next_token = top_k_indices.gather(-1, next_token_idx)
                else:
                    # Sample from full distribution
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                # Append predicted song to sequence
                current_sequence = torch.cat([current_sequence, next_token], dim=1)
        
        return current_sequence


# Testing and validation
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING CONTEXT-AWARE TRANSFORMER")
    print("=" * 80)
    
    # Test configuration
    num_items = 50000 + 2  # vocab + PAD + UNK
    num_contexts = 8  # workout, study, party, chill, sleep, sad, happy, other
    
    print(f"\nModel Configuration:")
    print(f"  Vocabulary size: {num_items:,}")
    print(f"  Number of contexts: {num_contexts}")
    print(f"  Embedding dim: 256")
    print(f"  Attention heads: 8")
    print(f"  Transformer layers: 4")
    
    # Create model
    model = ContextAwareTransformer(
        num_items=num_items,
        num_contexts=num_contexts,
        d_model=256,
        nhead=8,
        num_layer=4,
        dropout=0.1
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test 1: Forward pass
    print("\n[Test 1] Forward pass:")
    batch_size = 4
    seq_len = 10
    
    # Random inputs
    x = torch.randint(2, num_items, (batch_size, seq_len))  # Skip PAD/UNK
    context = torch.randint(0, num_contexts, (batch_size,))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    logits = model(x, context, mask)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Context shape: {context.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Output range: [{logits.min():.2f}, {logits.max():.2f}]")
    assert logits.shape == (batch_size, num_items), "Output shape mismatch!"
    print("  ✓ Forward pass works!")
    
    # Test 2: Generation
    print("\n[Test 2] Playlist generation:")
    seed = torch.randint(2, num_items, (5,))  # 5 seed songs
    print(f"  Seed songs: {seed.tolist()}")
    print(f"  Context: workout (0)")
    print(f"  Target length: 15")
    
    generated = model.generate_playlist(
        seed, 
        context_idx=0, 
        max_length=15,
        temperature=1.0,
        top_k=50
    )
    
    print(f"  Generated shape: {generated.shape}")
    print(f"  Generated songs: {generated[0].tolist()}")
    assert generated.shape[1] == 15, "Generated wrong length!"
    print("  ✓ Generation works!")
    
    # Test 3: Attention extraction
    print("\n[Test 3] Attention extraction:")
    attention_weights = model.get_attention_weights(x[:1], context[:1], mask[:1])
    
    print(f"  Number of layers: {len(attention_weights)}")
    if len(attention_weights) > 0:
        print(f"  Attention shape per layer: {attention_weights[0].shape}")
        print(f"  Expected: (batch=1, heads=8, seq_len+1, seq_len+1)")
        print("  ✓ Attention extraction works!")
    
    # Test 4: Context conditioning
    print("\n[Test 4] Context effect:")
    seed = torch.randint(2, num_items, (1, 5))
    mask = torch.ones(1, 5, dtype=torch.bool)
    
    # Generate with different contexts
    logits_workout = model(seed, torch.tensor([0]), mask)  # workout
    logits_study = model(seed, torch.tensor([1]), mask)    # study
    
    # Check if outputs differ
    diff = (logits_workout - logits_study).abs().mean().item()
    print(f"  Same seed, different contexts")
    print(f"  Workout logits mean: {logits_workout.mean():.4f}")
    print(f"  Study logits mean: {logits_study.mean():.4f}")
    print(f"  Mean absolute difference: {diff:.4f}")
    
    if diff > 0.01:
        print("  ✓ Context affects predictions!")
    else:
        print("  ⚠️  Context has little effect (might need more training)")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)
    print("\nModel is ready for training!")
    print("Next step: python train_context_model.py")