#!/usr/bin/env python3
"""
Example 1: Understanding Scaled Dot-Product Attention

This example demonstrates the core attention mechanism that underlies
the Transformer architecture. We'll explore:

1. How queries, keys, and values work together
2. Why scaling by sqrt(d_k) is important
3. How attention weights are computed and visualized
4. The effect of masking on attention patterns

Based on "An Introduction to Transformers" by Richard E. Turner

Run this example:
    python -m src.examples.example1_attention
"""

import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transformers.attention import ScaledDotProductAttention


def demonstrate_basic_attention():
    """
    Demonstrate basic scaled dot-product attention.
    
    Attention computes: softmax(QK^T / sqrt(d_k)) * V
    
    This allows each position to attend to all other positions,
    with learned attention weights determining the importance.
    """
    print("=" * 60)
    print("Basic Scaled Dot-Product Attention")
    print("=" * 60)
    
    # Parameters
    batch_size = 1
    seq_len = 4  # Short sequence for visualization
    d_k = 8  # Dimension of key/query vectors
    
    # Create simple query, key, value tensors
    # In practice, these come from linear projections of input
    torch.manual_seed(42)
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)
    
    print(f"\nShape of Q, K, V: ({batch_size}, {seq_len}, {d_k})")
    
    # Step 1: Compute attention scores (before scaling)
    scores_unscaled = torch.matmul(Q, K.transpose(-2, -1))
    print(f"\nUnscaled attention scores (QK^T):")
    print(scores_unscaled[0].numpy().round(2))
    
    # Step 2: Scale by sqrt(d_k)
    scores_scaled = scores_unscaled / math.sqrt(d_k)
    print(f"\nScaled attention scores (QK^T / sqrt({d_k})):")
    print(scores_scaled[0].numpy().round(2))
    
    # Step 3: Apply softmax
    attention_weights = F.softmax(scores_scaled, dim=-1)
    print(f"\nAttention weights (after softmax):")
    print(attention_weights[0].numpy().round(3))
    print(f"Each row sums to: {attention_weights[0].sum(dim=-1).numpy().round(3)}")
    
    # Step 4: Compute weighted sum
    output = torch.matmul(attention_weights, V)
    print(f"\nOutput shape: {output.shape}")
    
    return attention_weights[0].numpy()


def demonstrate_scaling_importance():
    """
    Demonstrate why scaling by sqrt(d_k) is crucial.
    
    Without scaling, large d_k values lead to very peaked softmax
    distributions, causing vanishing gradients during training.
    """
    print("\n" + "=" * 60)
    print("Why Scaling Matters")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Compare different dimensions
    dimensions = [8, 64, 512]
    
    for d_k in dimensions:
        # Random unit vectors
        q = torch.randn(1, d_k)
        k = torch.randn(10, d_k)  # 10 keys
        
        # Unscaled dot product
        scores_unscaled = torch.matmul(q, k.T)
        
        # Scaled dot product
        scores_scaled = scores_unscaled / math.sqrt(d_k)
        
        # Softmax distributions
        weights_unscaled = F.softmax(scores_unscaled, dim=-1)
        weights_scaled = F.softmax(scores_scaled, dim=-1)
        
        print(f"\nd_k = {d_k}:")
        print(f"  Unscaled variance: {scores_unscaled.var().item():.2f}")
        print(f"  Scaled variance:   {scores_scaled.var().item():.2f}")
        print(f"  Max unscaled attention: {weights_unscaled.max():.4f}")
        print(f"  Max scaled attention:   {weights_scaled.max():.4f}")
    
    print("\n→ Without scaling, larger d_k leads to more peaked attention")
    print("  (one position gets almost all the weight)")


def demonstrate_masking():
    """
    Demonstrate causal (look-ahead) masking.
    
    In autoregressive models like GPT, we must prevent tokens
    from attending to future positions. This is done by masking.
    """
    print("\n" + "=" * 60)
    print("Causal Masking for Autoregressive Models")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    seq_len = 4
    d_k = 8
    
    Q = torch.randn(1, seq_len, d_k)
    K = torch.randn(1, seq_len, d_k)
    V = torch.randn(1, seq_len, d_k)
    
    # Create causal mask
    # True means "mask this position" (don't attend to it)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    print("\nCausal mask (True = masked, can't attend):")
    print(causal_mask.int().numpy())
    
    # Compute attention without mask
    attention = ScaledDotProductAttention()
    output_no_mask, weights_no_mask = attention(Q, K, V, mask=None)
    
    # Compute attention with causal mask
    output_masked, weights_masked = attention(Q, K, V, mask=causal_mask)
    
    print("\nAttention weights WITHOUT mask:")
    print(weights_no_mask[0].numpy().round(3))
    
    print("\nAttention weights WITH causal mask:")
    print(weights_masked[0].numpy().round(3))
    print("\n→ Note: Each row can only attend to current and previous positions")
    
    return weights_no_mask[0].numpy(), weights_masked[0].numpy()


def visualize_attention_patterns(save_path: str = None):
    """
    Create visualization of attention patterns.
    """
    # Run demonstrations and get attention weights
    print("\n" + "=" * 60)
    print("Generating Attention Visualizations")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    seq_len = 6
    d_k = 8
    
    Q = torch.randn(1, seq_len, d_k)
    K = torch.randn(1, seq_len, d_k)
    V = torch.randn(1, seq_len, d_k)
    
    attention = ScaledDotProductAttention()
    
    # Without mask
    _, weights_full = attention(Q, K, V)
    
    # With causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    _, weights_causal = attention(Q, K, V, mask=causal_mask)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot full attention
    im1 = axes[0].imshow(weights_full[0].numpy(), cmap='Blues', aspect='equal')
    axes[0].set_title('Full Attention\n(Bidirectional)', fontsize=12)
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    axes[0].set_xticks(range(seq_len))
    axes[0].set_yticks(range(seq_len))
    plt.colorbar(im1, ax=axes[0], label='Attention Weight')
    
    # Plot causal attention
    im2 = axes[1].imshow(weights_causal[0].numpy(), cmap='Blues', aspect='equal')
    axes[1].set_title('Causal Attention\n(Unidirectional/Autoregressive)', fontsize=12)
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    axes[1].set_xticks(range(seq_len))
    axes[1].set_yticks(range(seq_len))
    plt.colorbar(im2, ax=axes[1], label='Attention Weight')
    
    plt.suptitle('Attention Patterns in Transformers', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    
    plt.close()


def demonstrate_self_vs_cross_attention():
    """
    Explain the difference between self-attention and cross-attention.
    """
    print("\n" + "=" * 60)
    print("Self-Attention vs Cross-Attention")
    print("=" * 60)
    
    print("""
SELF-ATTENTION (used in both Encoder and Decoder):
    - Q, K, V all come from the SAME sequence
    - Each token attends to all tokens in its own sequence
    - Example: Understanding context in "The cat sat on the mat"
    
    Input:  [The, cat, sat, on, the, mat]
            ↓ ↓ ↓ ↓ ↓ ↓
    Q, K, V all derived from this same sequence

CROSS-ATTENTION (used in Decoder only):
    - Q comes from the decoder (target sequence)
    - K, V come from the encoder (source sequence)
    - Decoder tokens attend to encoder representation
    - Example: Translation - French decoder attends to English encoder
    
    Encoder: [The, cat, sat] → Memory representation
                              ↘
    Decoder: [Le, chat] → Q     Cross-attention with K, V from encoder
""")
    
    # Demonstrate with code
    torch.manual_seed(42)
    
    attention = ScaledDotProductAttention()
    
    # Self-attention: same sequence for Q, K, V
    seq = torch.randn(1, 4, 8)  # (batch, seq_len, d_k)
    output_self, weights_self = attention(seq, seq, seq)
    
    print("Self-attention weights shape:", weights_self.shape)
    print("  → (batch, seq_len, seq_len)")
    
    # Cross-attention: Q from one sequence, K/V from another
    query_seq = torch.randn(1, 3, 8)   # Decoder sequence (3 tokens)
    memory_seq = torch.randn(1, 5, 8)  # Encoder sequence (5 tokens)
    
    output_cross, weights_cross = attention(query_seq, memory_seq, memory_seq)
    
    print("\nCross-attention weights shape:", weights_cross.shape)
    print("  → (batch, decoder_len, encoder_len)")
    print("  Each decoder position attends to all encoder positions")


def main():
    """Run all attention demonstrations."""
    print("\n" + "#" * 60)
    print("# EXAMPLE 1: Scaled Dot-Product Attention Deep Dive")
    print("#" * 60)
    
    # Run demonstrations
    demonstrate_basic_attention()
    demonstrate_scaling_importance()
    demonstrate_masking()
    demonstrate_self_vs_cross_attention()
    
    # Generate visualization
    visualize_attention_patterns("/tmp/attention_patterns.png")
    
    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
1. ATTENTION FORMULA: softmax(QK^T / sqrt(d_k)) * V
   - Q (Query): What am I looking for?
   - K (Key): What do I have to offer?
   - V (Value): What information do I provide?

2. SCALING: Division by sqrt(d_k) prevents vanishing gradients
   - Without scaling, softmax becomes too peaked for large d_k
   - This is why it's called "Scaled" Dot-Product Attention

3. MASKING: Essential for autoregressive generation
   - Causal mask prevents attending to future positions
   - Ensures the model can only use past context

4. ATTENTION PATTERNS: 
   - Full attention: Bidirectional (BERT-style encoders)
   - Causal attention: Unidirectional (GPT-style decoders)
""")


if __name__ == "__main__":
    main()
