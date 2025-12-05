"""
Feed-Forward Network for Transformers

This module implements the position-wise feed-forward network component
of the Transformer architecture as described in "An Introduction to 
Transformers" by Richard E. Turner.

The feed-forward network applies the same two-layer neural network to
each position independently and identically. This consists of two linear
transformations with a ReLU (or GELU) activation in between.

References:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - "An Introduction to Transformers" (Turner, 2024)
"""

import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    This module implements the feed-forward sublayer of a Transformer:
    
        FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
    Or with GELU activation:
    
        FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
    
    The network expands the dimensionality to d_ff (typically 4x d_model)
    before projecting back to d_model. This expansion allows the network
    to learn more complex transformations.
    
    Args:
        d_model (int): Input and output dimension
        d_ff (int): Hidden layer dimension (typically 4 * d_model)
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation function ('relu' or 'gelu'). Default: 'relu'
    
    Example:
        >>> ffn = FeedForwardNetwork(d_model=512, d_ff=2048, dropout=0.1)
        >>> x = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)
        >>> output = ffn(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply position-wise feed-forward network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # First linear transformation with activation
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second linear transformation
        x = self.linear2(x)
        
        return x
