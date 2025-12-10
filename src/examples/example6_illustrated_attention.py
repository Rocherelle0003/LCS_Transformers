#!/usr/bin/env python3
"""
Example 6: The Illustrated Transformer - Visual Deep Dive

This example provides detailed visualizations inspired by Jay Alammar's
"The Illustrated Transformer" (https://jalammar.github.io/illustrated-transformer/).

Key concepts covered:
1. Step-by-step attention visualization with real words
2. Query-Key-Value intuition with analogies
3. Multi-head attention visualization across different heads
4. Complete encoding pipeline visualization

Run this example:
    python -m src.examples.example6_illustrated_attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transformers import MultiHeadAttention, PositionalEncoding


def illustrate_qkv_analogy():
    """
    Explain Query-Key-Value using real-world analogies.
    
    Inspired by "The Illustrated Transformer" visualization style.
    """
    print("=" * 70)
    print("Query-Key-Value: The Library Analogy")
    print("=" * 70)
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    THE LIBRARY ANALOGY                                ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  Imagine you're in a library looking for information:                  ║
║                                                                        ║
║   QUERY (Q) = Your question                                         ║
║      "I need information about machine learning"                       ║
║                                                                        ║
║   KEY (K) = Book titles/labels on the shelf                         ║
║      ["Deep Learning", "Cooking 101", "ML Fundamentals", "Poetry"]    ║
║                                                                        ║
║   VALUE (V) = Actual book contents                                  ║
║      [DL content, Cooking content, ML content, Poetry content]        ║
║                                                                        ║
║   ATTENTION MECHANISM:                                             ║
║      1. Compare your QUERY with each KEY (dot product)                ║
║      2. Higher match = higher attention weight                         ║
║      3. Use weights to blend VALUES                                    ║
║                                                                        ║
║  Result: You get mostly "Deep Learning" and "ML Fundamentals"         ║
║          content, with little from "Cooking" or "Poetry"               ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Demonstrate with actual tensors
    torch.manual_seed(42)
    
    # Simulate 4 "books" with 8-dimensional representations
    d_k = 8
    
    # Query: "machine learning question"
    query = torch.randn(1, 1, d_k)
    
    # Keys: Different topic labels
    # Make ML-related keys similar to query
    key_dl = query + 0.1 * torch.randn(1, 1, d_k)  # Deep Learning (similar)
    key_cooking = torch.randn(1, 1, d_k)  # Cooking (different)
    key_ml = query + 0.2 * torch.randn(1, 1, d_k)  # ML (very similar)
    key_poetry = torch.randn(1, 1, d_k)  # Poetry (different)
    
    keys = torch.cat([key_dl, key_cooking, key_ml, key_poetry], dim=1)
    
    # Values: Content representations
    values = torch.randn(1, 4, d_k)
    
    # Compute attention
    scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(d_k)
    attention_weights = F.softmax(scores, dim=-1)
    
    topics = ["Deep Learning", "Cooking 101", "ML Fundamentals", "Poetry"]
    print("\nAttention Weights (how relevant each book is to your query):")
    print("-" * 50)
    for topic, weight in zip(topics, attention_weights[0, 0].tolist()):
        bar = "█" * int(weight * 40)
        print(f"  {topic:20s}: {weight:.3f} {bar}")
    
    print("\n→ Notice how ML-related topics get more attention!")


def illustrate_sentence_attention():
    """
    Show attention patterns for a real sentence.
    
    Demonstrates how words attend to each other in self-attention.
    """
    print("\n" + "=" * 70)
    print("Self-Attention on a Real Sentence")
    print("=" * 70)
    
    # Example sentence
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    
    print(f"\nSentence: '{' '.join(sentence)}'")
    print("\nIn self-attention, each word asks:")
    print("  'Which other words should I pay attention to?'")
    
    torch.manual_seed(42)
    
    # Create mock embeddings (in practice, these come from embedding layer)
    seq_len = len(sentence)
    d_model = 64
    
    # Create attention module
    attention = MultiHeadAttention(d_model=d_model, num_heads=4, dropout=0.0)
    
    # Random embeddings for demonstration
    embeddings = torch.randn(1, seq_len, d_model)
    
    # Get attention weights
    with torch.no_grad():
        _, attention_weights = attention(embeddings, embeddings, embeddings)
    
    # Average across heads for visualization
    avg_attention = attention_weights[0].mean(dim=0).numpy()
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap
    im = axes[0].imshow(avg_attention, cmap='Blues', aspect='equal')
    axes[0].set_xticks(range(seq_len))
    axes[0].set_yticks(range(seq_len))
    axes[0].set_xticklabels(sentence, fontsize=10)
    axes[0].set_yticklabels(sentence, fontsize=10)
    axes[0].set_xlabel('Attending TO (Key)', fontsize=11)
    axes[0].set_ylabel('Attending FROM (Query)', fontsize=11)
    axes[0].set_title('Self-Attention Heatmap\n(Average across heads)', fontsize=12)
    plt.colorbar(im, ax=axes[0])
    
    # Per-head visualization
    axes[1].set_title('Attention by Head', fontsize=12)
    x = np.arange(seq_len)
    width = 0.2
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for head_idx in range(4):
        # Show attention from "cat" (position 1) to all other words
        cat_attention = attention_weights[0, head_idx, 1, :].numpy()
        axes[1].bar(x + head_idx * width, cat_attention, width, 
                   label=f'Head {head_idx+1}', color=colors[head_idx])
    
    axes[1].set_xlabel('Word', fontsize=11)
    axes[1].set_ylabel('Attention Weight', fontsize=11)
    axes[1].set_xticks(x + width * 1.5)
    axes[1].set_xticklabels(sentence, fontsize=10)
    axes[1].legend()
    axes[1].set_title("Where does 'cat' attend?\n(Each head's perspective)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig('src/tmp/sentence_attention.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved to src/tmp/sentence_attention.png")
    print("\nKey observations:")
    print("  • Each row shows where one word attends")
    print("  • Different heads learn different patterns")
    print("  • 'cat' might attend to 'sat' (action) or 'mat' (location)")


def illustrate_positional_encoding_patterns():
    """
    Visualize positional encoding patterns in detail.
    
    Shows the wave-like nature and how positions are uniquely encoded.
    """
    print("\n" + "=" * 70)
    print("Positional Encoding: Unique Position Fingerprints")
    print("=" * 70)
    
    d_model = 128
    max_len = 100
    
    pe = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
    pe_matrix = pe.pe[0, :50, :64].numpy()  # First 50 positions, 64 dimensions
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Full positional encoding heatmap
    im1 = axes[0, 0].imshow(pe_matrix.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 0].set_xlabel('Position', fontsize=11)
    axes[0, 0].set_ylabel('Dimension', fontsize=11)
    axes[0, 0].set_title('Positional Encoding Matrix\n(Each column is a position)', fontsize=12)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. Wave patterns for different dimensions
    positions = np.arange(50)
    for dim in [0, 4, 8, 16, 32]:
        axes[0, 1].plot(positions, pe_matrix[:, dim], label=f'Dim {dim}', linewidth=2)
    axes[0, 1].set_xlabel('Position', fontsize=11)
    axes[0, 1].set_ylabel('Encoding Value', fontsize=11)
    axes[0, 1].set_title('Wave Patterns at Different Dimensions\n(Lower dims = higher frequency)', fontsize=12)
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-1.1, 1.1)
    
    # 3. Position similarity matrix
    similarity = np.dot(pe_matrix, pe_matrix.T)
    im3 = axes[1, 0].imshow(similarity, cmap='viridis', aspect='equal')
    axes[1, 0].set_xlabel('Position', fontsize=11)
    axes[1, 0].set_ylabel('Position', fontsize=11)
    axes[1, 0].set_title('Position Similarity Matrix\n(Nearby positions are more similar)', fontsize=12)
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 4. 3D-like visualization of position vectors
    # Show a few position vectors as bars
    positions_to_show = [0, 5, 10, 20, 40]
    x = np.arange(32)  # First 32 dimensions
    width = 0.15
    
    for i, pos in enumerate(positions_to_show):
        axes[1, 1].bar(x + i * width, pe_matrix[pos, :32], width, 
                       label=f'Pos {pos}', alpha=0.8)
    
    axes[1, 1].set_xlabel('Dimension', fontsize=11)
    axes[1, 1].set_ylabel('Encoding Value', fontsize=11)
    axes[1, 1].set_title('Position Vectors (First 32 dims)\n(Each position has unique pattern)', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('src/tmp/positional_encoding_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved to src/tmp/positional_encoding_patterns.png")
    
    print("""
KEY INSIGHTS:
─────────────
1. Each position has a UNIQUE encoding (like a fingerprint)
2. Lower dimensions oscillate faster (capture fine differences)
3. Higher dimensions oscillate slower (capture coarse patterns)
4. Nearby positions have SIMILAR encodings
5. The model can learn RELATIVE positions from these patterns
   (PE[pos+k] can be expressed as linear combination of PE[pos])
""")


def illustrate_encoder_pipeline():
    """
    Show the complete encoder pipeline step by step.
    """
    print("\n" + "=" * 70)
    print("Complete Encoder Pipeline (The Illustrated Transformer)")
    print("=" * 70)
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              ENCODER STEP-BY-STEP                                     ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  INPUT: "The cat sat"                                                  ║
║    ↓                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────┐  ║
║  │ 1. TOKENIZATION                                                 │  ║
║  │    "The" → 101, "cat" → 2087, "sat" → 3421                     │  ║
║  └─────────────────────────────────────────────────────────────────┘  ║
║    ↓                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────┐  ║
║  │ 2. EMBEDDING LOOKUP                                             │  ║
║  │    101 → [0.2, -0.1, 0.8, ...] (512 dims)                      │  ║
║  │    Each token gets a dense vector representation                │  ║
║  └─────────────────────────────────────────────────────────────────┘  ║
║    ↓                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────┐  ║
║  │ 3. ADD POSITIONAL ENCODING                                      │  ║
║  │    Embedding + PE = Final input to encoder                      │  ║
║  │    Now the model knows: "cat" is at position 1                  │  ║
║  └─────────────────────────────────────────────────────────────────┘  ║
║    ↓                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────┐  ║
║  │ 4. SELF-ATTENTION LAYER                                         │  ║
║  │    Each word looks at ALL words                                 │  ║
║  │    "cat" attends to "The" (article) and "sat" (action)         │  ║
║  │                                                                 │  ║
║  │    [The] ←──┐                                                  │  ║
║  │    [cat] ←──┼──── attention connections                        │  ║
║  │    [sat] ←──┘                                                  │  ║
║  └─────────────────────────────────────────────────────────────────┘  ║
║    ↓                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────┐  ║
║  │ 5. RESIDUAL + LAYER NORM                                        │  ║
║  │    output = LayerNorm(input + attention_output)                 │  ║
║  └─────────────────────────────────────────────────────────────────┘  ║
║    ↓                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────┐  ║
║  │ 6. FEED-FORWARD NETWORK                                         │  ║
║  │    FFN(x) = ReLU(x·W1 + b1)·W2 + b2                            │  ║
║  │    Each position processed independently                        │  ║
║  └─────────────────────────────────────────────────────────────────┘  ║
║    ↓                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────┐  ║
║  │ 7. RESIDUAL + LAYER NORM                                        │  ║
║  │    output = LayerNorm(attention_output + ffn_output)            │  ║
║  └─────────────────────────────────────────────────────────────────┘  ║
║    ↓                                                                   ║
║  OUTPUT: Contextual embeddings for each word                          ║
║    [The_ctx] [cat_ctx] [sat_ctx]                                      ║
║                                                                        ║
║  These embeddings now UNDERSTAND CONTEXT!                              ║
║  "cat" knows it's THE cat that SAT                                     ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def illustrate_multi_head_specialization():
    """
    Show how different heads can specialize in different patterns.
    """
    print("\n" + "=" * 70)
    print("Multi-Head Attention: Specialized Perspectives")
    print("=" * 70)
    
    # Create synthetic attention patterns that heads might learn
    seq_len = 8
    sentence = ["<s>", "The", "quick", "brown", "fox", "jumps", "over", "lazy"]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Define different types of patterns heads might learn
    patterns = [
        ("Positional: Previous", create_previous_pattern(seq_len)),
        ("Positional: Next", create_next_pattern(seq_len)),
        ("Syntactic: Adjective-Noun", create_adjective_noun_pattern(seq_len)),
        ("Global: Start Token", create_start_token_pattern(seq_len)),
        ("Local: Window", create_window_pattern(seq_len, 2)),
        ("Semantic: Related Words", create_semantic_pattern(seq_len)),
        ("Copy: Diagonal", create_copy_pattern(seq_len)),
        ("Long-range: Distant", create_distant_pattern(seq_len)),
    ]
    
    for idx, (title, pattern) in enumerate(patterns):
        ax = axes[idx // 4, idx % 4]
        im = ax.imshow(pattern, cmap='Blues', aspect='equal', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        if idx >= 4:
            ax.set_xticklabels(sentence, rotation=45, ha='right', fontsize=7)
        else:
            ax.set_xticklabels([])
        if idx % 4 == 0:
            ax.set_yticklabels(sentence, fontsize=7)
        else:
            ax.set_yticklabels([])
    
    plt.suptitle('Different Attention Heads Learn Different Patterns\n(Each head sees the sentence differently)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('src/tmp/multi_head_specialization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved to src/tmp/multi_head_specialization.png")
    print("""
HEAD SPECIALIZATION EXAMPLES:
─────────────────────────────
1. POSITIONAL HEADS: Attend to relative positions
   - Previous: Look at the token before
   - Next: Look at the token after
   
2. SYNTACTIC HEADS: Capture grammatical structure
   - Adjective-Noun: "quick" → "fox", "brown" → "fox"
   
3. SEMANTIC HEADS: Related meanings
   - "fox" and "jumps" are related (subject-verb)
   
4. GLOBAL HEADS: Aggregate sequence information
   - Start token collects overall meaning
   
5. COPY HEADS: Preserve original information
   - Strong diagonal = identity operation
""")


# Helper functions for pattern generation
def create_previous_pattern(seq_len):
    pattern = np.eye(seq_len, k=-1)
    pattern[0, 0] = 1  # First token attends to itself
    return pattern

def create_next_pattern(seq_len):
    pattern = np.eye(seq_len, k=1)
    pattern[-1, -1] = 1  # Last token attends to itself
    return pattern

def create_adjective_noun_pattern(seq_len):
    pattern = np.zeros((seq_len, seq_len))
    # Positions 2,3 (quick, brown) attend to position 4 (fox)
    pattern[2, 4] = 0.9
    pattern[3, 4] = 0.9
    # Add some self-attention
    np.fill_diagonal(pattern, 0.3)
    return pattern

def create_start_token_pattern(seq_len):
    pattern = np.zeros((seq_len, seq_len))
    pattern[:, 0] = 0.7  # All attend to start
    pattern[0, :] = 0.8  # Start attends to all
    np.fill_diagonal(pattern, 0.5)
    return pattern

def create_window_pattern(seq_len, window):
    pattern = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(max(0, i-window), min(seq_len, i+window+1)):
            pattern[i, j] = 0.8 if abs(i-j) <= window else 0
    return pattern

def create_semantic_pattern(seq_len):
    pattern = np.zeros((seq_len, seq_len))
    # fox (4) and jumps (5) are semantically related
    pattern[4, 5] = 0.8
    pattern[5, 4] = 0.8
    # over (6) and lazy (7)
    pattern[6, 7] = 0.6
    np.fill_diagonal(pattern, 0.4)
    return pattern

def create_copy_pattern(seq_len):
    return np.eye(seq_len) * 0.9

def create_distant_pattern(seq_len):
    pattern = np.zeros((seq_len, seq_len))
    # Last token attends to first few
    pattern[-1, :3] = 0.7
    # First tokens attend to last
    pattern[:2, -1] = 0.6
    np.fill_diagonal(pattern, 0.3)
    return pattern


def main():
    """Run all illustrated transformer demonstrations."""
    print("\n" + "#" * 70)
    print("# EXAMPLE 6: The Illustrated Transformer - Visual Deep Dive")
    print("#" * 70)
    print("\nInspired by: https://jalammar.github.io/illustrated-transformer/")
    
    # Run demonstrations
    illustrate_qkv_analogy()
    illustrate_sentence_attention()
    illustrate_positional_encoding_patterns()
    illustrate_encoder_pipeline()
    illustrate_multi_head_specialization()
    
    print("\n" + "=" * 70)
    print("Key Takeaways from The Illustrated Transformer:")
    print("=" * 70)
    print("""
1. QUERY-KEY-VALUE is like searching in a library
   - Query = your question
   - Keys = labels to match against
   - Values = actual content to retrieve

2. SELF-ATTENTION lets each word "look at" all other words
   - Creates context-aware representations
   - Different patterns for different relationships

3. POSITIONAL ENCODING gives position information
   - Sine/cosine waves at different frequencies
   - Nearby positions have similar encodings

4. MULTI-HEAD ATTENTION is like having multiple perspectives
   - Each head can specialize in different patterns
   - Syntactic, semantic, positional, etc.

5. THE ENCODER builds rich, contextual representations
   - Input: word embeddings + positions
   - Output: context-aware embeddings
""")


if __name__ == "__main__":
    main()
