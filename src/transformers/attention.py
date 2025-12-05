"""
Attention Mechanisms for Transformers

This module implements the core attention mechanisms described in 
"An Introduction to Transformers" by Richard E. Turner.

Key concepts:
1. Scaled Dot-Product Attention: The fundamental attention operation
2. Multi-Head Attention: Parallel attention with different learned projections

References:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - "An Introduction to Transformers" (Turner, 2024)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    This is the fundamental building block of the Transformer architecture.
    Given Query (Q), Key (K), and Value (V) matrices, it computes:
    
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    The scaling factor 1/sqrt(d_k) is crucial to prevent the dot products
    from growing too large in magnitude, which would push the softmax into
    regions of extremely small gradients.
    
    Args:
        dropout (float): Dropout probability applied to attention weights.
                        Default: 0.0 (no dropout)
    
    Attributes:
        dropout (nn.Dropout): Dropout layer for regularization
    
    Example:
        >>> attention = ScaledDotProductAttention(dropout=0.1)
        >>> Q = torch.randn(2, 8, 10, 64)  # (batch, heads, seq_len, d_k)
        >>> K = torch.randn(2, 8, 10, 64)
        >>> V = torch.randn(2, 8, 10, 64)
        >>> output, weights = attention(Q, K, V)
        >>> print(output.shape)  # torch.Size([2, 8, 10, 64])
        >>> print(weights.shape)  # torch.Size([2, 8, 10, 10])
    """
    
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (..., seq_len_q, d_k)
            key: Key tensor of shape (..., seq_len_k, d_k)
            value: Value tensor of shape (..., seq_len_k, d_v)
            mask: Optional mask tensor. Positions with True are masked out.
                  Shape should be broadcastable to (..., seq_len_q, seq_len_k)
        
        Returns:
            output: Attention output of shape (..., seq_len_q, d_v)
            attention_weights: Attention weights of shape (..., seq_len_q, seq_len_k)
        """
        d_k = query.size(-1)
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # Shape: (..., seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided (for padding or causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout for regularization
        attention_weights = self.dropout(attention_weights)
        
        # Compute weighted sum of values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Instead of performing a single attention function, this layer projects
    the queries, keys, and values h times with different learned linear
    projections. On each of these projected versions, attention is performed
    in parallel, yielding h different outputs that are concatenated and
    linearly projected again.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    
    This allows the model to jointly attend to information from different
    representation subspaces at different positions.
    
    Args:
        d_model (int): The dimension of the model (embedding size)
        num_heads (int): Number of parallel attention heads
        dropout (float): Dropout probability. Default: 0.0
    
    Attributes:
        d_model (int): Model dimension
        num_heads (int): Number of heads
        d_k (int): Dimension of each head (d_model // num_heads)
        W_q (nn.Linear): Query projection layer
        W_k (nn.Linear): Key projection layer
        W_v (nn.Linear): Value projection layer
        W_o (nn.Linear): Output projection layer
        attention (ScaledDotProductAttention): The base attention mechanism
    
    Example:
        >>> mha = MultiHeadAttention(d_model=512, num_heads=8, dropout=0.1)
        >>> x = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)
        >>> output, weights = mha(x, x, x)  # Self-attention
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=dropout)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.
        
        Args:
            query: Query tensor of shape (batch, seq_len_q, d_model)
            key: Key tensor of shape (batch, seq_len_k, d_model)
            value: Value tensor of shape (batch, seq_len_k, d_model)
            mask: Optional mask tensor for attention scores
        
        Returns:
            output: Attention output of shape (batch, seq_len_q, d_model)
            attention_weights: Attention weights of shape 
                              (batch, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)
        
        # 1. Linear projections
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. Reshape to (batch, num_heads, seq_len, d_k)
        # This splits d_model into num_heads parallel attention heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Apply attention
        if mask is not None:
            # Handle different mask shapes
            if mask.dim() == 2:
                # (seq_len_q, seq_len_k) -> (1, 1, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                # (batch, seq_len_q, seq_len_k) -> (batch, 1, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(1)
        
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        
        # 4. Concatenate heads: reshape back to (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # 5. Final linear projection
        output = self.W_o(attn_output)
        
        return output, attention_weights
