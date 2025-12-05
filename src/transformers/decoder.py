"""
Transformer Decoder

This module implements the Transformer Decoder as described in 
"An Introduction to Transformers" by Richard E. Turner.

The decoder generates the output sequence one token at a time,
using both self-attention (on the output produced so far) and
cross-attention (attending to the encoder's output).

Key differences from the encoder:
1. Masked self-attention prevents attending to future positions
2. Cross-attention allows attending to the encoder's output

References:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - "An Introduction to Transformers" (Turner, 2024)
"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForwardNetwork


class TransformerDecoderBlock(nn.Module):
    """
    Single Transformer Decoder Block.
    
    Each decoder block consists of:
    1. Masked Multi-Head Self-Attention (prevents looking ahead)
    2. Add & Normalize
    3. Multi-Head Cross-Attention (attends to encoder output)
    4. Add & Normalize
    5. Position-wise Feed-Forward Network
    6. Add & Normalize
    
    The causal mask ensures that position i can only attend to
    positions <= i, maintaining autoregressive properties.
    
    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward hidden dimension
        dropout (float): Dropout probability. Default: 0.1
    
    Example:
        >>> decoder_block = TransformerDecoderBlock(
        ...     d_model=512, num_heads=8, d_ff=2048, dropout=0.1
        ... )
        >>> x = torch.randn(2, 8, 512)  # (batch, tgt_seq_len, d_model)
        >>> memory = torch.randn(2, 10, 512)  # encoder output
        >>> output = decoder_block(x, memory)
        >>> print(output.shape)  # torch.Size([2, 8, 512])
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Masked Self-Attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            dropout=dropout
        )
        
        # Cross-Attention (attends to encoder output)
        self.cross_attention = MultiHeadAttention(
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
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(p=dropout)
    
    @staticmethod
    def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate a causal (look-ahead) mask for self-attention.
        
        The mask prevents attending to future positions.
        
        Args:
            seq_len: Sequence length
            device: Device to create the mask on
        
        Returns:
            Boolean mask of shape (seq_len, seq_len) where True 
            indicates positions to mask out
        """
        # Create upper triangular mask (True for positions to mask)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()
    
    def forward(
        self, 
        x: torch.Tensor, 
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Process input through the decoder block.
        
        Args:
            x: Target sequence of shape (batch, tgt_seq_len, d_model)
            memory: Encoder output of shape (batch, src_seq_len, d_model)
            tgt_mask: Optional mask for target self-attention
            memory_mask: Optional mask for cross-attention
        
        Returns:
            Output tensor of shape (batch, tgt_seq_len, d_model)
        """
        seq_len = x.size(1)
        
        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_causal_mask(seq_len, x.device)
        
        # Pre-LN Masked Self-Attention with residual
        normalized = self.norm1(x)
        self_attn_output, _ = self.self_attention(
            query=normalized,
            key=normalized,
            value=normalized,
            mask=tgt_mask
        )
        x = x + self.dropout(self_attn_output)
        
        # Pre-LN Cross-Attention with residual
        normalized = self.norm2(x)
        cross_attn_output, _ = self.cross_attention(
            query=normalized,
            key=memory,
            value=memory,
            mask=memory_mask
        )
        x = x + self.dropout(cross_attn_output)
        
        # Pre-LN Feed-Forward with residual
        normalized = self.norm3(x)
        ff_output = self.feed_forward(normalized)
        x = x + self.dropout(ff_output)
        
        return x


class TransformerDecoder(nn.Module):
    """
    Stack of Transformer Decoder Blocks.
    
    The full decoder is a stack of N identical decoder blocks.
    It generates the output sequence autoregressively, attending
    to both its own previous outputs and the encoder's representation.
    
    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward hidden dimension
        num_layers (int): Number of decoder blocks
        dropout (float): Dropout probability. Default: 0.1
    
    Example:
        >>> decoder = TransformerDecoder(
        ...     d_model=512, num_heads=8, d_ff=2048, num_layers=6, dropout=0.1
        ... )
        >>> x = torch.randn(2, 8, 512)  # target sequence
        >>> memory = torch.randn(2, 10, 512)  # encoder output
        >>> output = decoder(x, memory)
        >>> print(output.shape)  # torch.Size([2, 8, 512])
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
        
        # Stack of decoder blocks
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
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
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Process input through all decoder layers.
        
        Args:
            x: Target sequence of shape (batch, tgt_seq_len, d_model)
            memory: Encoder output of shape (batch, src_seq_len, d_model)
            tgt_mask: Optional mask for target self-attention
            memory_mask: Optional mask for cross-attention
        
        Returns:
            Decoded representation of shape (batch, tgt_seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        
        return self.norm(x)
