#!/usr/bin/env python3
"""
Example 5: Transformer Variants and Modern Architectures

This example explores various Transformer variants and architectural
innovations that have emerged since the original "Attention Is All You Need":

1. Encoder-only (BERT-style)
2. Decoder-only (GPT-style)
3. Efficient attention mechanisms
4. Architectural innovations

Based on "An Introduction to Transformers" by Richard E. Turner

Run this example:
    python -m src.examples.example5_variants
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transformers import (
    TransformerEncoder,
    TransformerDecoder,
    MultiHeadAttention,
    PositionalEncoding,
    FeedForwardNetwork
)


class EncoderOnlyTransformer(nn.Module):
    """
    Encoder-only Transformer (BERT-style).
    
    Used for tasks that require understanding the full context:
    - Text classification
    - Named entity recognition
    - Sentence similarity
    - Question answering (extractive)
    
    Key characteristic: Bidirectional attention (each token can attend to all others)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        d_ff: int = 3072,
        num_layers: int = 12,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Encoder layers
        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Encode
        return self.encoder(x, attention_mask)


class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only Transformer (GPT-style).
    
    Used for autoregressive generation tasks:
    - Text generation
    - Code completion
    - Creative writing
    - Chatbots
    
    Key characteristic: Causal attention (each token can only attend to previous tokens)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        d_ff: int = 3072,
        num_layers: int = 12,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Decoder layers (using decoder without cross-attention)
        self.layers = nn.ModuleList([
            CausalSelfAttentionBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Position indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device), 
            diagonal=1
        ).bool()
        
        # Through decoder layers
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        x = self.norm(x)
        return self.output_projection(x)
    
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Get predictions for last position
            logits = self(input_ids)[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Truncate if too long
            if input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]
        
        return input_ids


class CausalSelfAttentionBlock(nn.Module):
    """Decoder block with only causal self-attention (no cross-attention)."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-LN self-attention
        normalized = self.norm1(x)
        attn_out, _ = self.self_attention(normalized, normalized, normalized, mask)
        x = x + self.dropout(attn_out)
        
        # Pre-LN feed-forward
        normalized = self.norm2(x)
        ff_out = self.feed_forward(normalized)
        x = x + self.dropout(ff_out)
        
        return x


def explain_variants():
    """Explain the main Transformer variants."""
    print("=" * 70)
    print("Transformer Variants: Encoder, Decoder, and Encoder-Decoder")
    print("=" * 70)
    
    print("""
┌────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER VARIANTS                             │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. ENCODER-ONLY (BERT-style)                                       │
│     ─────────────────────────                                       │
│     Structure: Stack of encoder blocks                              │
│     Attention: Bidirectional (full context)                         │
│                                                                      │
│     [Input tokens] → Encoder → [Contextualized representations]     │
│                                                                      │
│     Use cases:                                                       │
│     • Classification (sentiment, topic)                             │
│     • NER (named entity recognition)                                │
│     • Sentence similarity                                            │
│     • Fill-in-the-blank (masked language modeling)                  │
│                                                                      │
│     Examples: BERT, RoBERTa, DistilBERT, ALBERT                     │
│                                                                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  2. DECODER-ONLY (GPT-style)                                        │
│     ─────────────────────────                                       │
│     Structure: Stack of decoder blocks (no cross-attention)         │
│     Attention: Causal (can only see past)                           │
│                                                                      │
│     [Input tokens] → Decoder → [Next token prediction]              │
│                                                                      │
│     Use cases:                                                       │
│     • Text generation                                                │
│     • Code completion                                                │
│     • Chatbots / dialogue                                            │
│     • Any autoregressive task                                        │
│                                                                      │
│     Examples: GPT-2, GPT-3, GPT-4, LLaMA, Claude                    │
│                                                                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  3. ENCODER-DECODER (T5-style)                                      │
│     ──────────────────────────                                      │
│     Structure: Full original Transformer architecture               │
│     Attention: Bidirectional (encoder) + Causal (decoder)           │
│                                                                      │
│     [Source] → Encoder → [Memory] ←─┐                               │
│                                      │ cross-attention               │
│     [Target] → Decoder ────────────┘ → [Output]                     │
│                                                                      │
│     Use cases:                                                       │
│     • Machine translation                                            │
│     • Summarization                                                  │
│     • Question answering (generative)                                │
│     • Any sequence-to-sequence task                                  │
│                                                                      │
│     Examples: T5, BART, mT5, FLAN-T5                                │
│                                                                      │
└────────────────────────────────────────────────────────────────────┘
""")


def demonstrate_encoder_only():
    """Demonstrate encoder-only architecture."""
    print("\n" + "=" * 70)
    print("Encoder-Only: BERT-style Classification")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Create small model
    vocab_size = 1000
    d_model = 128
    num_classes = 3
    
    model = EncoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        d_ff=512,
        num_layers=2,
        max_seq_len=64,
        dropout=0.1
    )
    
    # Add classification head
    classifier = nn.Linear(d_model, num_classes)
    
    # Example input
    input_ids = torch.randint(0, vocab_size, (2, 20))  # batch=2, seq_len=20
    
    with torch.no_grad():
        # Get encoded representations
        encoded = model(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Encoded shape: {encoded.shape}")
        
        # Use [CLS] token representation for classification (first token)
        cls_repr = encoded[:, 0, :]
        print(f"CLS representation shape: {cls_repr.shape}")
        
        # Classify
        logits = classifier(cls_repr)
        print(f"Classification logits shape: {logits.shape}")
        print(f"Predicted classes: {logits.argmax(dim=-1).tolist()}")
    
    print("""
BERT-STYLE CLASSIFICATION:
    1. Encode full sequence with bidirectional attention
    2. Extract [CLS] token representation (position 0)
    3. Pass through classification head
    4. Apply softmax for probabilities
    
The [CLS] token aggregates information from the whole sequence
through bidirectional self-attention.
""")


def demonstrate_decoder_only():
    """Demonstrate decoder-only architecture."""
    print("\n" + "=" * 70)
    print("Decoder-Only: GPT-style Generation")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Create small model
    vocab_size = 100
    
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        d_ff=256,
        num_layers=2,
        max_seq_len=32,
        dropout=0.1
    )
    
    # Example input (prompt)
    prompt = torch.randint(0, vocab_size, (1, 5))  # 5 token prompt
    
    print(f"Prompt tokens: {prompt[0].tolist()}")
    
    with torch.no_grad():
        # Forward pass
        logits = model(prompt)
        print(f"Output logits shape: {logits.shape}")
        
        # Next token prediction
        next_token_logits = logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1)
        print(f"Next token prediction: {next_token.item()}")
        
        # Generation
        generated = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=50)
        print(f"Generated sequence: {generated[0].tolist()}")
    
    print("""
GPT-STYLE GENERATION:
    1. Input prompt tokens
    2. Causal attention (each token only sees previous)
    3. Predict next token distribution
    4. Sample or take argmax
    5. Append to sequence, repeat
    
Temperature: Controls randomness (lower = more deterministic)
Top-k: Only sample from top k most likely tokens
""")


def explain_efficient_attention():
    """Explain efficient attention mechanisms."""
    print("\n" + "=" * 70)
    print("Efficient Attention Mechanisms")
    print("=" * 70)
    
    print("""
PROBLEM: Standard attention is O(n²) in sequence length
    - For n=1000: 1,000,000 attention scores
    - For n=10000: 100,000,000 attention scores
    - Quickly becomes memory-bound
    
SOLUTIONS:

1. SPARSE ATTENTION (e.g., Longformer, BigBird)
   ─────────────────────────────────────────
   • Only compute attention for selected positions
   • Local attention: sliding window
   • Global attention: special tokens attend to all
   • Strided patterns: skip positions
   
   Complexity: O(n × k) where k is pattern size
   
   ┌───────────────────────────────────────┐
   │ Local Window (sliding attention)      │
   │                                        │
   │  Token:  1  2  3  4  5  6  7  8       │
   │  Attends: [1,2] [1,2,3] [2,3,4] ...   │
   │                                        │
   │ Each token only attends to neighbors   │
   └───────────────────────────────────────┘

2. LINEAR ATTENTION (e.g., Performer, Linear Transformer)
   ──────────────────────────────────────────────────
   • Approximate softmax(QK^T)V differently
   • Use kernel feature maps: φ(Q)φ(K)^T V
   • Can compute as φ(Q)(φ(K)^T V) = O(n × d²)
   
   Complexity: O(n × d²) instead of O(n² × d)

3. FLASH ATTENTION
   ────────────────
   • Same mathematical result as standard attention
   • Hardware-aware algorithm (GPU memory hierarchy)
   • Tiles the computation to avoid materialization
   • 2-4x faster, uses less memory
   
   Key insight: Fuse attention computation to avoid
   reading/writing intermediate matrices to HBM

4. MULTI-QUERY / GROUPED-QUERY ATTENTION
   ────────────────────────────────────
   • Share K, V projections across heads
   • Reduces memory bandwidth requirements
   • Used in PaLM, LLaMA 2, etc.
   
   Multi-Query: All heads share same K, V
   Grouped-Query: Groups of heads share K, V
""")


def explain_architectural_innovations():
    """Explain architectural innovations in modern Transformers."""
    print("\n" + "=" * 70)
    print("Architectural Innovations")
    print("=" * 70)
    
    print("""
1. PRE-NORM vs POST-NORM
   ─────────────────────
   Original (Post-Norm): x + Attention(LayerNorm(x)) ❌
   Modern (Pre-Norm):    x + Attention(LayerNorm(x)) ✓
   
   Pre-Norm is more stable for deep networks.

2. ROTARY POSITION EMBEDDING (RoPE)
   ─────────────────────────────
   • Encodes position in the attention operation itself
   • Rotates Q and K by position-dependent angle
   • Better extrapolation to longer sequences
   • Used in LLaMA, GPT-NeoX, etc.

3. GATED LINEAR UNITS (GLU)
   ─────────────────────
   Standard FFN: GELU(xW1)W2
   GLU:         (xW1 ⊗ sigmoid(xW_gate))W2
   
   The gate provides finer control over information flow.

4. SwiGLU ACTIVATION
   ──────────────────
   Combines Swish activation with GLU:
   SwiGLU(x) = Swish(xW1) ⊗ (xW_gate)
   
   Used in LLaMA, PaLM, and other modern LLMs.

5. RMSNORM vs LAYERNORM
   ─────────────────────
   LayerNorm: Normalizes mean and variance
   RMSNorm:   Only normalizes by RMS (no mean subtraction)
   
   RMSNorm is simpler and often works as well.

6. PARALLEL ATTENTION + FFN
   ────────────────────────
   Standard:  x → Attn → FFN → output
   Parallel:  x → [Attn + FFN] → output
   
   Can be 15% faster with similar quality (GPT-J, PaLM).

7. MIXTURE OF EXPERTS (MoE)
   ─────────────────────────
   • Replace FFN with multiple "expert" FFNs
   • Router selects which expert(s) to use
   • Increases parameters without proportional compute
   • Used in Mixtral, Switch Transformer, etc.
""")


def visualize_attention_patterns():
    """Visualize different attention patterns."""
    print("\n" + "=" * 70)
    print("Attention Patterns Comparison")
    print("=" * 70)
    
    seq_len = 16
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    patterns = [
        ("Full Bidirectional\n(BERT encoder)", torch.ones(seq_len, seq_len)),
        ("Causal\n(GPT decoder)", torch.tril(torch.ones(seq_len, seq_len))),
        ("Local Window (k=4)\n(Longformer)", create_local_mask(seq_len, 4)),
        ("Strided\n(Sparse)", create_strided_mask(seq_len, 4)),
        ("Global + Local\n(BigBird)", create_global_local_mask(seq_len, 4, 2)),
        ("Cross-Attention\n(Encoder-Decoder)", torch.ones(seq_len, seq_len))
    ]
    
    for idx, (title, mask) in enumerate(patterns):
        ax = axes[idx // 3, idx % 3]
        im = ax.imshow(mask.numpy(), cmap='Blues', aspect='equal', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_xticks(range(0, seq_len, 4))
        ax.set_yticks(range(0, seq_len, 4))
    
    plt.suptitle('Attention Patterns in Different Transformer Variants', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('src/tmp/attention_variants.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Attention patterns saved to src/tmp/attention_variants.png")


def create_local_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """Create local window attention mask."""
    mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = 1
    return mask


def create_strided_mask(seq_len: int, stride: int) -> torch.Tensor:
    """Create strided attention mask."""
    mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        # Attend to every stride-th position
        mask[i, ::stride] = 1
        # Plus local
        mask[i, max(0, i-1):i+2] = 1
    return mask


def create_global_local_mask(seq_len: int, window_size: int, num_global: int) -> torch.Tensor:
    """Create global + local attention mask."""
    mask = create_local_mask(seq_len, window_size)
    # Global tokens (first num_global positions)
    mask[:, :num_global] = 1  # All attend to global
    mask[:num_global, :] = 1  # Global attends to all
    return mask


def main():
    """Run all variant demonstrations."""
    print("\n" + "#" * 70)
    print("# EXAMPLE 5: Transformer Variants and Modern Architectures")
    print("#" * 70)
    
    # Explain variants
    explain_variants()
    
    # Demonstrate each variant
    demonstrate_encoder_only()
    demonstrate_decoder_only()
    
    # Explain efficient attention
    explain_efficient_attention()
    
    # Explain architectural innovations
    explain_architectural_innovations()
    
    # Visualize patterns
    visualize_attention_patterns()
    
    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print("""
1. VARIANT SELECTION
   - Encoder-only: Understanding/classification tasks
   - Decoder-only: Generation tasks (dominant for LLMs)
   - Encoder-decoder: Sequence-to-sequence tasks

2. EFFICIENCY MATTERS
   - Quadratic attention limits sequence length
   - Many solutions: sparse, linear, Flash Attention
   - Trade-offs between speed, memory, and quality

3. MODERN ARCHITECTURES
   - Pre-norm for stability
   - RoPE for better position encoding
   - SwiGLU for better FFN
   - MoE for scaling parameters

4. THE TREND
   - Decoder-only dominates for large language models
   - Efficiency innovations enable longer contexts
   - Architecture simplification (fewer components)
""")


if __name__ == "__main__":
    main()
