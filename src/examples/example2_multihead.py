#!/usr/bin/env python3
"""
Example 2: Multi-Head Attention and Representation Learning

This example explores Multi-Head Attention, a key innovation that allows
the model to jointly attend to information from different representation
subspaces at different positions.

Key concepts covered:
1. Why multiple heads are better than one
2. How heads learn different attention patterns
3. Visualizing attention across heads
4. The role of learned projections

Based on "An Introduction to Transformers" by Richard E. Turner

Run this example:
    python -m src.examples.example2_multihead
"""

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transformers.attention import MultiHeadAttention, ScaledDotProductAttention


def explain_multihead_intuition():
    """
    Explain the intuition behind multi-head attention.
    """
    print("=" * 60)
    print("Multi-Head Attention: The Intuition")
    print("=" * 60)
    
    print("""
WHY MULTIPLE HEADS?
───────────────────
Consider the sentence: "The cat sat on the mat because it was tired."

Different aspects to understand:
1. SYNTACTIC HEAD: "it" → relates to "cat" (subject reference)
2. SEMANTIC HEAD: "tired" → relates to "sat" (action-state relation)  
3. POSITIONAL HEAD: Nearby words provide local context

A single attention head can only learn ONE pattern.
Multiple heads let the model learn MULTIPLE patterns simultaneously!

MATHEMATICAL FORMULATION:
─────────────────────────
Instead of: Attention(Q, K, V) with d_model dimensions

We do: 
    head_i = Attention(QW_i^Q, KW_i^K, VW_i^V) with d_k = d_model/h dimensions
    MultiHead = Concat(head_1, ..., head_h) W^O

Each head has its own learned projections (W_i^Q, W_i^K, W_i^V),
allowing it to focus on different patterns.
""")


def demonstrate_head_specialization():
    """
    Show how different heads can specialize in different patterns.
    """
    print("\n" + "=" * 60)
    print("Head Specialization: Different Heads, Different Patterns")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Create multi-head attention
    d_model = 64
    num_heads = 4
    seq_len = 8
    batch_size = 1
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    
    # Create input sequence
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Perform self-attention
    output, attention_weights = mha(x, x, x)
    
    print(f"\nConfiguration:")
    print(f"  - d_model: {d_model}")
    print(f"  - num_heads: {num_heads}")
    print(f"  - d_k per head: {d_model // num_heads}")
    print(f"  - seq_len: {seq_len}")
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"  → (batch, num_heads, seq_len_q, seq_len_k)")
    
    # Analyze each head's attention pattern
    print("\nAttention Diversity (variance across heads):")
    for pos in range(min(4, seq_len)):
        weights_at_pos = attention_weights[0, :, pos, :]  # All heads at position pos
        variance = weights_at_pos.var(dim=0).mean().item()
        print(f"  Position {pos}: variance = {variance:.4f}")
    
    return attention_weights


def visualize_multiple_heads(attention_weights: torch.Tensor, save_path: str = None):
    """
    Visualize attention patterns for each head.
    """
    print("\n" + "=" * 60)
    print("Visualizing Attention Patterns Across Heads")
    print("=" * 60)
    
    num_heads = attention_weights.size(1)
    seq_len = attention_weights.size(2)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for head_idx in range(min(4, num_heads)):
        weights = attention_weights[0, head_idx].numpy()
        
        im = axes[head_idx].imshow(weights, cmap='viridis', aspect='equal')
        axes[head_idx].set_title(f'Head {head_idx + 1}', fontsize=12)
        axes[head_idx].set_xlabel('Key Position')
        axes[head_idx].set_ylabel('Query Position')
        axes[head_idx].set_xticks(range(seq_len))
        axes[head_idx].set_yticks(range(seq_len))
        plt.colorbar(im, ax=axes[head_idx])
    
    plt.suptitle('Multi-Head Attention: Different Heads Learn Different Patterns', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.close()


def demonstrate_projection_matrices():
    """
    Examine the learned projection matrices in multi-head attention.
    """
    print("\n" + "=" * 60)
    print("Understanding Projection Matrices")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    d_model = 64
    num_heads = 4
    d_k = d_model // num_heads
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    print(f"\nProjection Matrix Dimensions:")
    print(f"  W_q: {mha.W_q.weight.shape} (maps {d_model} → {d_model})")
    print(f"  W_k: {mha.W_k.weight.shape} (maps {d_model} → {d_model})")
    print(f"  W_v: {mha.W_v.weight.shape} (maps {d_model} → {d_model})")
    print(f"  W_o: {mha.W_o.weight.shape} (maps {d_model} → {d_model})")
    
    print(f"\nConceptually, W_q is split into {num_heads} heads:")
    print(f"  Each head's W_q_i: ({d_k}, {d_model})")
    
    # Show that the full projection contains all heads
    W_q = mha.W_q.weight.detach()
    
    print(f"\nThe full W_q matrix can be viewed as {num_heads} stacked head projections:")
    for i in range(num_heads):
        start, end = i * d_k, (i + 1) * d_k
        print(f"  Head {i+1}: rows {start}-{end}")


def demonstrate_computational_efficiency():
    """
    Show that multi-head attention is computationally efficient.
    """
    print("\n" + "=" * 60)
    print("Computational Efficiency of Multi-Head Attention")
    print("=" * 60)
    
    print("""
KEY INSIGHT: Despite having multiple heads, the computation is efficient!

Single Large Attention:
    - d_model = 512
    - Attention dimension: 512 × 512 = 262,144 per position pair
    
Multi-Head Attention (8 heads):
    - d_k = 512 / 8 = 64 per head
    - Total per position pair: 8 × (64 × 64) = 32,768
    - WAIT... that's LESS than single head!
    
Actually, the comparison is:
    - Single head: O(n² × d_model)
    - Multi-head:  O(n² × d_model) ← SAME!
    
The magic: heads are computed in PARALLEL, and the reduced d_k
per head keeps total FLOPs the same as a single large head.
""")
    
    # Demonstrate with timing
    import time
    
    d_model = 512
    seq_len = 128
    batch_size = 8
    
    # Compare single-head equivalent vs multi-head
    print("\nBenchmark (small-scale demonstration):")
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Multi-head attention
    mha = MultiHeadAttention(d_model=d_model, num_heads=8, dropout=0.0)
    
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = mha(x, x, x)
    mha_time = (time.time() - start) / 10
    
    # Single-head attention (d_k = d_model)
    single_attn = ScaledDotProductAttention(dropout=0.0)
    W_q = nn.Linear(d_model, d_model)
    W_k = nn.Linear(d_model, d_model)
    W_v = nn.Linear(d_model, d_model)
    
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            Q = W_q(x)
            K = W_k(x)
            V = W_v(x)
            _ = single_attn(Q, K, V)
    single_time = (time.time() - start) / 10
    
    print(f"  Multi-head (8 heads): {mha_time*1000:.2f} ms")
    print(f"  Single large head:    {single_time*1000:.2f} ms")
    print(f"  → Similar computational cost, but multiple attention patterns!")


def demonstrate_head_analysis():
    """
    Analyze what different heads might be learning.
    """
    print("\n" + "=" * 60)
    print("Analyzing Head Behavior (Simulated Example)")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Create a sequence with different patterns
    seq_len = 8
    d_model = 64
    num_heads = 4
    
    # Simulate different types of relationships in data
    print("""
In trained models, researchers have found heads that specialize in:

1. POSITIONAL HEADS: Attend to specific relative positions
   - "Attend to previous token"
   - "Attend to next token"
   
2. SYNTACTIC HEADS: Attend based on grammatical structure
   - Subject-verb agreement
   - Pronoun resolution
   
3. SEMANTIC HEADS: Attend based on meaning
   - Word similarity
   - Entity tracking
   
4. RARE PATTERN HEADS: Handle edge cases
   - Long-distance dependencies
   - Special constructions
""")
    
    # Create synthetic attention patterns to illustrate
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Head 1: Local attention (previous token)
    attn1 = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        if i > 0:
            attn1[i, i-1] = 0.7
            attn1[i, i] = 0.3
        else:
            attn1[i, i] = 1.0
    
    # Head 2: Forward attention (next tokens)
    attn2 = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(i, min(i+3, seq_len)):
            attn2[i, j] = 0.5 ** (j - i)
        attn2[i] = attn2[i] / attn2[i].sum()
    
    # Head 3: Long-range (first and last positions)
    attn3 = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        attn3[i, 0] = 0.4
        attn3[i, -1] = 0.4
        attn3[i, i] = 0.2
    
    # Head 4: Uniform (equal attention)
    attn4 = torch.ones(seq_len, seq_len) / seq_len
    
    patterns = [
        (attn1, "Local (Previous Token)"),
        (attn2, "Forward Looking"),
        (attn3, "Long-Range (First/Last)"),
        (attn4, "Uniform/Averaging")
    ]
    
    for idx, (attn, title) in enumerate(patterns):
        ax = axes[idx // 2, idx % 2]
        im = ax.imshow(attn.numpy(), cmap='Blues', aspect='equal', vmin=0, vmax=1)
        ax.set_title(f'Synthetic Head: {title}', fontsize=11)
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Synthetic Examples: Types of Attention Patterns Heads Can Learn', 
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('/tmp/head_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization of synthetic head patterns saved to /tmp/head_patterns.png")


def main():
    """Run all multi-head attention demonstrations."""
    print("\n" + "#" * 60)
    print("# EXAMPLE 2: Multi-Head Attention Deep Dive")
    print("#" * 60)
    
    # Run demonstrations
    explain_multihead_intuition()
    attention_weights = demonstrate_head_specialization()
    visualize_multiple_heads(attention_weights, "/tmp/multihead_attention.png")
    demonstrate_projection_matrices()
    demonstrate_computational_efficiency()
    demonstrate_head_analysis()
    
    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
1. MULTIPLE HEADS = MULTIPLE PATTERNS
   - Each head learns its own projection (W_q, W_k, W_v)
   - Allows the model to capture different types of relationships
   - Some heads may focus on syntax, others on semantics

2. COMPUTATIONAL EFFICIENCY
   - Despite multiple heads, total computation is similar
   - d_k per head = d_model / num_heads
   - Heads computed in parallel

3. HEAD SPECIALIZATION
   - Different heads naturally learn different patterns
   - Some attend locally, others globally
   - This diversity improves model expressiveness

4. OUTPUT COMBINATION
   - Head outputs are concatenated
   - Final linear projection W_o combines the information
   - This allows complex cross-head interactions
""")


if __name__ == "__main__":
    main()
