"""
Positional Encoding for Transformers

This module implements positional encoding as described in 
"An Introduction to Transformers" by Richard E. Turner.

Since the Transformer architecture doesn't have any inherent notion of
position (unlike RNNs or CNNs), we must inject information about the
relative or absolute position of tokens in the sequence.

The sinusoidal positional encoding uses sine and cosine functions of
different frequencies to create unique position embeddings.

References:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - "An Introduction to Transformers" (Turner, 2024)
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    
    Adds positional information to the input embeddings using sine and
    cosine functions of different frequencies:
    
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    where pos is the position and i is the dimension.
    
    This encoding has several advantages:
    1. It allows the model to easily learn to attend by relative positions
    2. It can extrapolate to sequence lengths longer than those seen during training
    3. The encoding is deterministic and doesn't require learning
    
    Args:
        d_model (int): The dimension of the model embeddings
        max_len (int): Maximum sequence length to precompute. Default: 5000
        dropout (float): Dropout probability. Default: 0.1
    
    Example:
        >>> pe = PositionalEncoding(d_model=512, max_len=1000, dropout=0.1)
        >>> x = torch.randn(2, 50, 512)  # (batch, seq_len, d_model)
        >>> output = pe(x)
        >>> print(output.shape)  # torch.Size([2, 50, 512])
    """
    
    def __init__(
        self, 
        d_model: int, 
        max_len: int = 5000, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the division term
        # exp(2i * -log(10000) / d_model) = 1/10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model) with
            positional encoding added
        """
        seq_len = x.size(1)
        
        # Add positional encoding (broadcasting over batch dimension)
        x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding.
    
    An alternative to sinusoidal encoding where positions are represented
    by learned embedding vectors. This is used in models like BERT and GPT.
    
    Pros:
    - Can learn task-specific positional patterns
    - Potentially more flexible
    
    Cons:
    - Limited to maximum sequence length seen during training
    - Requires additional parameters
    
    Args:
        d_model (int): The dimension of the model embeddings
        max_len (int): Maximum sequence length. Default: 512
        dropout (float): Dropout probability. Default: 0.1
    
    Example:
        >>> pe = LearnedPositionalEncoding(d_model=512, max_len=512, dropout=0.1)
        >>> x = torch.randn(2, 50, 512)  # (batch, seq_len, d_model)
        >>> output = pe(x)
        >>> print(output.shape)  # torch.Size([2, 50, 512])
    """
    
    def __init__(
        self, 
        d_model: int, 
        max_len: int = 512, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Learnable position embeddings
        self.position_embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model) with
            positional encoding added
        """
        seq_len = x.size(1)
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        # Add positional embeddings
        x = x + self.position_embedding(positions)
        
        return self.dropout(x)
