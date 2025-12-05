"""
Transformer Encoder

This module implements the Transformer Encoder as described in 
"An Introduction to Transformers" by Richard E. Turner.

The encoder processes the input sequence and produces a continuous
representation that captures contextual information from the entire
sequence. Each encoder layer applies self-attention followed by a
feed-forward network, with residual connections and layer normalization.

References:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - "An Introduction to Transformers" (Turner, 2024)
"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForwardNetwork


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder Block.
    
    Each encoder block consists of:
    1. Multi-Head Self-Attention
    2. Add & Normalize (residual connection + layer normalization)
    3. Position-wise Feed-Forward Network
    4. Add & Normalize
    
    The architecture follows the Pre-LN (Pre-Layer Normalization) variant
    which is more stable during training:
    
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    
    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward hidden dimension
        dropout (float): Dropout probability. Default: 0.1
    
    Example:
        >>> encoder_block = TransformerEncoderBlock(
        ...     d_model=512, num_heads=8, d_ff=2048, dropout=0.1
        ... )
        >>> x = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)
        >>> output = encoder_block(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            dropout=dropout
        )
        
        # Feed-Forward Network
        self.feed_forward = FeedForwardNetwork(
            d_model=d_model, 
            d_ff=d_ff, 
            dropout=dropout
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Process input through the encoder block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            src_mask: Optional source mask for padding
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Pre-LN Self-Attention with residual connection
        normalized = self.norm1(x)
        attn_output, _ = self.self_attention(
            query=normalized, 
            key=normalized, 
            value=normalized, 
            mask=src_mask
        )
        x = x + self.dropout(attn_output)
        
        # Pre-LN Feed-Forward with residual connection
        normalized = self.norm2(x)
        ff_output = self.feed_forward(normalized)
        x = x + self.dropout(ff_output)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer Encoder Blocks.
    
    The full encoder is a stack of N identical encoder blocks.
    Each block can attend to all positions in the input sequence,
    allowing it to build rich contextual representations.
    
    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward hidden dimension
        num_layers (int): Number of encoder blocks
        dropout (float): Dropout probability. Default: 0.1
    
    Example:
        >>> encoder = TransformerEncoder(
        ...     d_model=512, num_heads=8, d_ff=2048, num_layers=6, dropout=0.1
        ... )
        >>> x = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)
        >>> output = encoder(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        num_layers: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Stack of encoder blocks
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        x: torch.Tensor, 
        src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Process input through all encoder layers.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            src_mask: Optional source mask for padding
        
        Returns:
            Encoded representation of shape (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return self.norm(x)
