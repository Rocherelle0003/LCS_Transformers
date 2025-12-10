#!/usr/bin/env python3
"""
Example 3: The Complete Transformer Architecture

This example provides a deep dive into the complete Transformer
architecture, showing how all components work together:

1. Encoder-Decoder structure
2. Residual connections and Layer Normalization
3. Information flow through the network
4. Training vs Inference behavior

Based on "An Introduction to Transformers" by Richard E. Turner

Run this example:
    python -m src.examples.example3_architecture
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

from src.transformers import (
    TransformerEncoderBlock,
    TransformerEncoder,
    TransformerDecoderBlock,
    TransformerDecoder,
    Transformer,
    PositionalEncoding,
    MultiHeadAttention,
    FeedForwardNetwork
)


def explain_architecture():
    """
    Explain the overall Transformer architecture.
    """
    print("=" * 70)
    print("The Complete Transformer Architecture")
    print("=" * 70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                     TRANSFORMER ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  INPUT SEQUENCE           OUTPUT SEQUENCE (shifted right)            │
│       ↓                          ↓                                   │
│  ┌─────────┐               ┌─────────┐                               │
│  │Embedding│               │Embedding│                               │
│  └────┬────┘               └────┬────┘                               │
│       │                         │                                    │
│  ┌────┴────┐               ┌────┴────┐                               │
│  │Pos Enc  │               │Pos Enc  │                               │
│  └────┬────┘               └────┬────┘                               │
│       │                         │                                    │
│  ┌────┴────────────────────┐   │                                    │
│  │                          │   │                                    │
│  │   ENCODER                │   │     DECODER                        │
│  │   ┌─────────────┐        │   │     ┌─────────────────┐            │
│  │   │Self-Attention│ ←Nx  │   │     │Masked Self-Attn │←           │
│  │   └──────┬──────┘       │   │     └────────┬────────┘ │          │
│  │          ↓               │   │              ↓          │          │
│  │   ┌─────────────┐       │   │     ┌─────────────────┐ │          │
│  │   │ Feed Forward │      │   │     │ Cross-Attention │ │ Nx       │
│  │   └──────┬──────┘       │   │     └────────┬────────┘ │          │
│  │          │               │   │              ↓          │          │
│  └──────────┼───────────────┘   │     ┌─────────────────┐ │          │
│             │                    │     │  Feed Forward   │ │          │
│             │                    │     └────────┬────────┘→          │
│             │                    │              ↓                     │
│             └────────────────────┼──→ (cross-attention)              │
│                                  │              ↓                     │
│                                  └─────┬────────┘                     │
│                                        ↓                              │
│                                 ┌──────────────┐                      │
│                                 │    Linear    │                      │
│                                 └──────┬───────┘                      │
│                                        ↓                              │
│                                 ┌──────────────┐                      │
│                                 │   Softmax    │                      │
│                                 └──────────────┘                      │
│                                        ↓                              │
│                                OUTPUT PROBABILITIES                   │
└─────────────────────────────────────────────────────────────────────┘

KEY COMPONENTS:
1. Embedding + Positional Encoding: Convert tokens to vectors with position info
2. Encoder Stack (Nx): Process source sequence with self-attention
3. Decoder Stack (Nx): Generate target with masked self-attn + cross-attn
4. Linear + Softmax: Produce output token probabilities
""")


def demonstrate_encoder_block():
    """
    Demonstrate the encoder block in detail.
    """
    print("\n" + "=" * 70)
    print("Encoder Block: Self-Attention + Feed-Forward")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Parameters
    d_model = 256
    num_heads = 8
    d_ff = 1024
    seq_len = 10
    batch_size = 2
    
    # Create encoder block
    encoder_block = TransformerEncoderBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1
    )
    encoder_block.eval()  # Disable dropout for demonstration
    
    print(f"\nEncoder Block Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_ff (hidden): {d_ff}")
    print(f"  d_k per head: {d_model // num_heads}")
    
    # Input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass with intermediate outputs
    with torch.no_grad():
        # Step 1: Layer norm before self-attention (Pre-LN)
        x_norm1 = encoder_block.norm1(x)
        print(f"After LayerNorm 1: {x_norm1.shape}")
        
        # Step 2: Self-attention
        attn_out, attn_weights = encoder_block.self_attention(x_norm1, x_norm1, x_norm1)
        print(f"Self-attention output: {attn_out.shape}")
        print(f"Attention weights: {attn_weights.shape}")
        
        # Step 3: Residual connection
        x_after_attn = x + attn_out
        print(f"After residual connection: {x_after_attn.shape}")
        
        # Step 4: Layer norm before FFN
        x_norm2 = encoder_block.norm2(x_after_attn)
        print(f"After LayerNorm 2: {x_norm2.shape}")
        
        # Step 5: Feed-forward network
        ffn_out = encoder_block.feed_forward(x_norm2)
        print(f"FFN output: {ffn_out.shape}")
        
        # Step 6: Final residual connection
        output = x_after_attn + ffn_out
        print(f"Final output: {output.shape}")
    
    print("""
ENCODER BLOCK FLOW (Pre-LN variant):
    x → LayerNorm → Self-Attention → (+x) → LayerNorm → FFN → (+previous) → output
    
Note: We use Pre-LN (LayerNorm before sublayers) which is more stable for training.
The original paper used Post-LN (LayerNorm after residual connections).
""")


def demonstrate_decoder_block():
    """
    Demonstrate the decoder block in detail.
    """
    print("\n" + "=" * 70)
    print("Decoder Block: Masked Self-Attn + Cross-Attn + Feed-Forward")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Parameters
    d_model = 256
    num_heads = 8
    d_ff = 1024
    src_len = 12  # Source/encoder sequence length
    tgt_len = 8   # Target/decoder sequence length
    batch_size = 2
    
    # Create decoder block
    decoder_block = TransformerDecoderBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1
    )
    decoder_block.eval()
    
    print(f"\nDecoder Block Configuration:")
    print(f"  Source sequence length: {src_len}")
    print(f"  Target sequence length: {tgt_len}")
    
    # Inputs
    tgt = torch.randn(batch_size, tgt_len, d_model)  # Target sequence
    memory = torch.randn(batch_size, src_len, d_model)  # Encoder output
    
    print(f"\nTarget input shape: {tgt.shape}")
    print(f"Memory (encoder output) shape: {memory.shape}")
    
    # Generate causal mask
    causal_mask = decoder_block.generate_causal_mask(tgt_len, tgt.device)
    print(f"\nCausal mask shape: {causal_mask.shape}")
    print("Causal mask (True = masked):")
    print(causal_mask.int().numpy())
    
    # Forward pass
    with torch.no_grad():
        output = decoder_block(tgt, memory)
        print(f"\nOutput shape: {output.shape}")
    
    print("""
DECODER BLOCK FLOW:
    1. Masked Self-Attention: Target attends to itself (causally masked)
    2. Cross-Attention: Target queries attend to encoder keys/values
    3. Feed-Forward: Position-wise transformation
    
The causal mask ensures position i can only attend to positions 0...i
(cannot look into the future during generation).
""")


def demonstrate_full_transformer():
    """
    Demonstrate the complete Transformer model.
    """
    print("\n" + "=" * 70)
    print("Complete Transformer: End-to-End")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Model configuration (smaller for demonstration)
    config = {
        'src_vocab_size': 1000,
        'tgt_vocab_size': 1000,
        'd_model': 256,
        'num_heads': 8,
        'd_ff': 1024,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dropout': 0.1
    }
    
    # Create model
    model = Transformer(**config)
    model.eval()
    
    print("\nModel Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    
    # Create input
    batch_size = 2
    src_len = 20
    tgt_len = 15
    
    src = torch.randint(0, config['src_vocab_size'], (batch_size, src_len))
    tgt = torch.randint(0, config['tgt_vocab_size'], (batch_size, tgt_len))
    
    print(f"\nSource tokens shape: {src.shape}")
    print(f"Target tokens shape: {tgt.shape}")
    
    # Forward pass
    with torch.no_grad():
        # Full forward pass
        output = model(src, tgt)
        print(f"Output logits shape: {output.shape}")
        print(f"  → (batch, tgt_len, vocab_size)")
        
        # Convert to probabilities
        probs = torch.softmax(output, dim=-1)
        print(f"\nProbability distribution at each position sums to: {probs[0, 0].sum():.4f}")
        
        # Get predicted tokens
        predicted = output.argmax(dim=-1)
        print(f"Predicted token indices: {predicted[0].tolist()[:10]}...")
    
    print("""
TRAINING vs INFERENCE:
──────────────────────
During TRAINING:
    - Teacher forcing: Use ground truth target tokens as decoder input
    - Parallel computation for all target positions
    - Loss computed over all positions simultaneously

During INFERENCE:
    - Autoregressive generation: Predict one token at a time
    - Each new prediction becomes input for next step
    - Much slower than training (sequential)
""")


def demonstrate_residual_and_normalization():
    """
    Explain residual connections and layer normalization.
    """
    print("\n" + "=" * 70)
    print("Residual Connections and Layer Normalization")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    d_model = 64
    seq_len = 4
    batch_size = 1
    
    # Sample input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print("\n1. RESIDUAL CONNECTIONS")
    print("-" * 40)
    print("""
Purpose: Enable deep networks by providing gradient highways
    
    output = x + F(x)
    
Benefits:
    - Gradient flows directly through addition
    - Model can learn identity mapping (F(x) = 0)
    - Enables training of very deep networks (100+ layers)
""")
    
    # Simulate sublayer transformation
    sublayer = nn.Linear(d_model, d_model)
    with torch.no_grad():
        F_x = sublayer(x)
        output_no_residual = F_x
        output_with_residual = x + F_x
        
        print(f"Input norm: {x.norm():.4f}")
        print(f"Output without residual norm: {output_no_residual.norm():.4f}")
        print(f"Output with residual norm: {output_with_residual.norm():.4f}")
    
    print("\n2. LAYER NORMALIZATION")
    print("-" * 40)
    print("""
Purpose: Stabilize training by normalizing activations

    LayerNorm(x) = (x - μ) / σ * γ + β
    
Where μ and σ are computed across the feature dimension (d_model)
for each position independently.

Difference from Batch Normalization:
    - LayerNorm: Normalizes across features for each sample
    - BatchNorm: Normalizes across samples for each feature
    - LayerNorm works with variable sequence lengths and batch size 1
""")
    
    # Demonstrate layer normalization
    layer_norm = nn.LayerNorm(d_model)
    
    with torch.no_grad():
        x_norm = layer_norm(x)
        
        # Check statistics after normalization
        print(f"\nBefore LayerNorm:")
        print(f"  Mean: {x.mean(dim=-1).numpy().round(3)}")
        print(f"  Std: {x.std(dim=-1).numpy().round(3)}")
        
        print(f"\nAfter LayerNorm:")
        print(f"  Mean: {x_norm.mean(dim=-1).numpy().round(3)}")
        print(f"  Std: {x_norm.std(dim=-1).numpy().round(3)}")


def demonstrate_positional_encoding_detail():
    """
    Deep dive into positional encoding.
    """
    print("\n" + "=" * 70)
    print("Positional Encoding: Giving Position Information")
    print("=" * 70)
    
    d_model = 64
    max_len = 100
    
    pe = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
    
    # Get the positional encoding matrix
    pe_matrix = pe.pe[0, :50, :].numpy()  # First 50 positions
    
    print("""
SINUSOIDAL POSITIONAL ENCODING:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Key Properties:
1. Each position has a unique encoding
2. Encodings are bounded (-1 to 1)
3. Model can learn relative positions (PE[pos+k] can be represented 
   as a linear function of PE[pos])
4. Can extrapolate to longer sequences than seen during training
""")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot encoding values
    im1 = axes[0].imshow(pe_matrix.T, aspect='auto', cmap='RdBu', 
                          vmin=-1, vmax=1)
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Embedding Dimension')
    axes[0].set_title('Positional Encoding Values')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot a few dimensions across positions
    positions = np.arange(50)
    for dim in [0, 1, 4, 5, 16, 17]:
        axes[1].plot(positions, pe_matrix[:, dim], label=f'dim {dim}')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Positional Encoding for Selected Dimensions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('src/tmp/positional_encoding.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved to src/tmp/positional_encoding.png")
    print("""
OBSERVATIONS:
- Even dimensions (0, 4, 16, ...): Sine waves with decreasing frequency
- Odd dimensions (1, 5, 17, ...): Cosine waves with decreasing frequency
- Low dimensions: High frequency (capture fine position differences)
- High dimensions: Low frequency (capture coarse position information)
""")


def main():
    """Run all architecture demonstrations."""
    print("\n" + "#" * 70)
    print("# EXAMPLE 3: The Complete Transformer Architecture")
    print("#" * 70)
    
    # Run demonstrations
    explain_architecture()
    demonstrate_encoder_block()
    demonstrate_decoder_block()
    demonstrate_residual_and_normalization()
    demonstrate_positional_encoding_detail()
    demonstrate_full_transformer()
    
    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print("""
1. ENCODER-DECODER STRUCTURE
   - Encoder: Processes source sequence with bidirectional self-attention
   - Decoder: Generates target with causal self-attention + cross-attention
   - Cross-attention connects decoder to encoder output

2. RESIDUAL CONNECTIONS
   - Enable training of deep networks (gradient highways)
   - output = x + SubLayer(x)
   - Present after both attention and FFN sublayers

3. LAYER NORMALIZATION
   - Stabilizes training by normalizing activations
   - Applied before sublayers (Pre-LN) for better stability
   - Normalizes across features, not batch

4. POSITIONAL ENCODING
   - Adds position information to embeddings
   - Sinusoidal encoding allows extrapolation
   - Critical since attention is position-invariant

5. TRAINING vs INFERENCE
   - Training: Teacher forcing, parallel computation
   - Inference: Autoregressive, sequential generation
""")


if __name__ == "__main__":
    main()
