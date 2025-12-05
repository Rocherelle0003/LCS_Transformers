#!/usr/bin/env python3
"""
Example 7: The Annotated Transformer - Step-by-Step Implementation

This example follows the style of "The Annotated Transformer" 
(https://nlp.seas.harvard.edu/2018/04/03/attention.html) by providing
detailed, annotated implementations with mathematical formulations.

Key concepts covered:
1. Layer-by-layer implementation with equations
2. Training a small Transformer on copy task
3. Label smoothing and learning rate scheduling
4. Visualization of training dynamics

Run this example:
    python -m src.examples.example7_annotated_transformer
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# PART 1: ATTENTION WITH DETAILED EQUATIONS
# ============================================================================

def attention_detailed(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention' with detailed comments.
    
    Mathematical formulation:
    ────────────────────────
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    Where:
    - Q ∈ ℝ^(n×d_k) : Query matrix (n queries of dimension d_k)
    - K ∈ ℝ^(m×d_k) : Key matrix (m keys of dimension d_k)  
    - V ∈ ℝ^(m×d_v) : Value matrix (m values of dimension d_v)
    - √d_k         : Scaling factor to prevent gradient vanishing
    
    The scaling factor is crucial because:
    - For large d_k, dot products grow large in magnitude
    - Large values push softmax into regions with tiny gradients
    - Scaling keeps the variance of scores ≈ 1
    
    Args:
        query: Tensor of shape (..., seq_len_q, d_k)
        key: Tensor of shape (..., seq_len_k, d_k)
        value: Tensor of shape (..., seq_len_k, d_v)
        mask: Optional mask tensor
        dropout: Optional dropout module
    
    Returns:
        output: Weighted sum of values, shape (..., seq_len_q, d_v)
        attention_weights: Attention distribution, shape (..., seq_len_q, seq_len_k)
    """
    d_k = query.size(-1)
    
    # Step 1: Compute raw attention scores
    # scores_ij = q_i · k_j (dot product between query i and key j)
    # Shape: (..., seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Step 2: Scale by √d_k
    # This keeps the variance of scores ≈ 1 regardless of d_k
    # Without scaling: Var(q·k) = d_k (assuming unit variance inputs)
    # With scaling: Var(q·k/√d_k) = 1
    scores = scores / math.sqrt(d_k)
    
    # Step 3: Apply mask (if provided)
    # Mask positions get -inf, so softmax gives them 0 weight
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Step 4: Apply softmax to get attention weights
    # Each row sums to 1: Σ_j attention_weights_ij = 1
    attention_weights = F.softmax(scores, dim=-1)
    
    # Step 5: Optional dropout on attention weights
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # Step 6: Compute weighted sum of values
    # output_i = Σ_j attention_weights_ij × v_j
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


# ============================================================================
# PART 2: MULTI-HEAD ATTENTION WITH EQUATIONS
# ============================================================================

class MultiHeadAttentionAnnotated(nn.Module):
    """
    Multi-Head Attention with detailed mathematical annotations.
    
    Mathematical formulation:
    ────────────────────────
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    Parameters:
    - W_i^Q ∈ ℝ^(d_model × d_k) : Query projection for head i
    - W_i^K ∈ ℝ^(d_model × d_k) : Key projection for head i
    - W_i^V ∈ ℝ^(d_model × d_v) : Value projection for head i
    - W^O ∈ ℝ^(h·d_v × d_model) : Output projection
    
    where d_k = d_v = d_model / h
    
    Why multiple heads?
    ──────────────────
    1. Different heads can attend to different positions
    2. Each head learns different relationship types
    3. Heads can specialize (syntax, semantics, position)
    4. Ensemble effect: combining multiple views
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # W^Q, W^K, W^V projections (stacked for efficiency)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # W^O output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        self.attn_weights = None  # Store for visualization
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass with shape annotations.
        
        Input shapes:
            query: (batch, seq_len_q, d_model)
            key:   (batch, seq_len_k, d_model)
            value: (batch, seq_len_k, d_model)
        
        Output shape:
            output: (batch, seq_len_q, d_model)
        """
        batch_size = query.size(0)
        
        # 1. Linear projections: (batch, seq_len, d_model) → (batch, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. Reshape for multi-head: (batch, seq_len, d_model) → (batch, h, seq_len, d_k)
        # This splits the d_model dimension into h heads of dimension d_k
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Apply attention on all heads in parallel
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
        
        attn_output, self.attn_weights = attention_detailed(
            Q, K, V, mask=mask, dropout=self.dropout
        )
        
        # 4. Concatenate heads: (batch, h, seq_len, d_k) → (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # 5. Final linear projection
        output = self.W_o(attn_output)
        
        return output


# ============================================================================
# PART 3: LEARNING RATE SCHEDULING (NOAM SCHEDULER)
# ============================================================================

def get_noam_scheduler(optimizer, d_model: int, warmup_steps: int = 4000):
    """
    Noam learning rate scheduler as described in "Attention is All You Need".
    
    Mathematical formulation:
    ────────────────────────
    lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))
    
    This creates a learning rate that:
    1. Increases linearly during warmup (step < warmup_steps)
    2. Decreases proportionally to 1/√step after warmup
    
    The warmup prevents unstable training at the beginning when
    the model parameters are randomly initialized.
    """
    def lr_lambda(step):
        # Avoid division by zero
        step = max(step, 1)
        
        # Noam formula
        arg1 = step ** (-0.5)
        arg2 = step * (warmup_steps ** (-1.5))
        
        return (d_model ** (-0.5)) * min(arg1, arg2)
    
    return LambdaLR(optimizer, lr_lambda)


def visualize_lr_schedule():
    """Visualize the Noam learning rate schedule."""
    print("=" * 70)
    print("Noam Learning Rate Schedule")
    print("=" * 70)
    
    d_models = [256, 512, 1024]
    warmup_steps = 4000
    steps = np.arange(1, 20001)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for d_model in d_models:
        lrs = []
        for step in steps:
            arg1 = step ** (-0.5)
            arg2 = step * (warmup_steps ** (-1.5))
            lr = (d_model ** (-0.5)) * min(arg1, arg2)
            lrs.append(lr)
        
        ax.plot(steps, lrs, label=f'd_model={d_model}')
    
    ax.axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.5, label='Warmup ends')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Noam Learning Rate Schedule\n(Warmup + Inverse Square Root Decay)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/noam_lr_schedule.png', dpi=150)
    plt.close()
    
    print("\nVisualization saved to /tmp/noam_lr_schedule.png")
    print("""
The Noam schedule has two phases:
1. WARMUP (step < 4000): LR increases linearly
   - Allows model to stabilize before large updates
   
2. DECAY (step >= 4000): LR decreases as 1/√step
   - Gradually reduces step size for fine convergence
   
Larger d_model → smaller peak LR (inversely proportional to √d_model)
""")


# ============================================================================
# PART 4: LABEL SMOOTHING
# ============================================================================

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing as described in the Transformer paper.
    
    Mathematical formulation:
    ────────────────────────
    Instead of hard targets (1 for correct class, 0 for others):
        y_hard = [0, 0, 1, 0, 0]  (one-hot)
    
    Use smoothed targets:
        y_smooth = [ε/K, ε/K, 1-ε+ε/K, ε/K, ε/K]
    
    where:
    - ε is the smoothing factor (e.g., 0.1)
    - K is the number of classes
    
    Benefits:
    ─────────
    1. Prevents over-confident predictions
    2. Acts as regularization
    3. Improves calibration
    4. Can improve generalization
    """
    
    def __init__(self, vocab_size: int, smoothing: float = 0.1, pad_idx: int = 0):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.pad_idx = pad_idx
        
        # Smoothed distribution: ε/K for non-target classes
        self.confidence = 1.0 - smoothing
        self.smoothing_value = smoothing / (vocab_size - 2)  # -2 for target and pad
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            logits: Model predictions (batch*seq_len, vocab_size)
            target: Target indices (batch*seq_len,)
        """
        # Create smoothed target distribution
        smooth_target = torch.full_like(logits, self.smoothing_value)
        smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_target[:, self.pad_idx] = 0
        
        # Mask padding positions
        non_pad_mask = target != self.pad_idx
        smooth_target = smooth_target * non_pad_mask.unsqueeze(1)
        
        # KL divergence loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_target * log_probs).sum(dim=-1)
        
        return loss[non_pad_mask].mean()


def demonstrate_label_smoothing():
    """Visualize the effect of label smoothing."""
    print("\n" + "=" * 70)
    print("Label Smoothing Visualization")
    print("=" * 70)
    
    vocab_size = 10
    target = 3  # True class
    
    # Hard labels
    hard_labels = torch.zeros(vocab_size)
    hard_labels[target] = 1.0
    
    # Smoothed labels (ε = 0.1)
    smoothing = 0.1
    smooth_labels = torch.full((vocab_size,), smoothing / (vocab_size - 1))
    smooth_labels[target] = 1.0 - smoothing + smoothing / (vocab_size - 1)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(vocab_size)
    
    # Hard labels
    axes[0].bar(x, hard_labels.numpy(), color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Target Probability')
    axes[0].set_title('Hard Labels (No Smoothing)\nTarget class = 3')
    axes[0].set_xticks(x)
    axes[0].set_ylim(0, 1.1)
    
    # Smoothed labels
    colors = ['coral' if i != target else 'steelblue' for i in range(vocab_size)]
    axes[1].bar(x, smooth_labels.numpy(), color=colors, edgecolor='black')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Target Probability')
    axes[1].set_title(f'Label Smoothing (ε={smoothing})\nNon-target classes get small probability')
    axes[1].set_xticks(x)
    axes[1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('/tmp/label_smoothing.png', dpi=150)
    plt.close()
    
    print("\nVisualization saved to /tmp/label_smoothing.png")
    print(f"""
Hard Labels:
  Class 3: 1.0
  Others:  0.0

Smoothed Labels (ε={smoothing}):
  Class 3: {smooth_labels[target]:.4f}
  Others:  {smooth_labels[0]:.4f} each

Effect: Model learns to be less overconfident
        Better calibration and generalization
""")


# ============================================================================
# PART 5: COPY TASK DEMONSTRATION
# ============================================================================

def generate_copy_data(batch_size: int, seq_len: int, vocab_size: int):
    """
    Generate data for the copy task.
    
    The copy task is a simple sanity check for sequence-to-sequence models:
    - Input:  [BOS, a, b, c, d, EOS]
    - Output: [BOS, a, b, c, d, EOS]
    
    The model must learn to simply copy the input sequence.
    """
    # Random sequence (excluding special tokens 0=PAD, 1=BOS, 2=EOS)
    data = torch.randint(3, vocab_size, (batch_size, seq_len))
    
    # Add BOS at start
    bos = torch.full((batch_size, 1), 1)
    
    # Source: [BOS, tokens]
    src = torch.cat([bos, data], dim=1)
    
    # Target: [BOS, tokens] (shifted for teacher forcing)
    tgt = torch.cat([bos, data], dim=1)
    
    return src, tgt


def demonstrate_copy_task():
    """Show the copy task setup."""
    print("\n" + "=" * 70)
    print("The Copy Task: Transformer Sanity Check")
    print("=" * 70)
    
    print("""
THE COPY TASK
─────────────
A simple diagnostic task where the model must copy input to output.

Example:
    Input:  [<BOS>, 7, 3, 9, 2, 5, <EOS>]
    Target: [<BOS>, 7, 3, 9, 2, 5, <EOS>]

This tests:
1. Can the model learn the identity function?
2. Does attention properly route information?
3. Are gradients flowing correctly?

A model that can't learn this has fundamental issues!
""")
    
    # Generate example
    src, tgt = generate_copy_data(batch_size=1, seq_len=5, vocab_size=10)
    
    print(f"Source sequence: {src[0].tolist()}")
    print(f"Target sequence: {tgt[0].tolist()}")
    print("\n→ They should be identical!")


# ============================================================================
# PART 6: GEOMETRIC SHAPES CLASSIFICATION (More Complex Example)
# ============================================================================

class GeometricShapeDataset:
    """
    Dataset for classifying geometric shapes from their properties.
    
    This is a more nuanced example as requested - going beyond simple
    triangles and rectangles to include:
    - Parallelograms
    - Trapezoids  
    - Rhombi
    - Regular polygons (pentagon, hexagon, etc.)
    """
    
    SHAPES = [
        'triangle', 'rectangle', 'square', 'parallelogram', 
        'rhombus', 'trapezoid', 'pentagon', 'hexagon',
        'heptagon', 'octagon', 'circle', 'ellipse'
    ]
    
    PROPERTIES = [
        'num_sides', 'num_right_angles', 'num_equal_sides',
        'num_parallel_pairs', 'is_regular', 'has_curved_edges',
        'is_convex', 'num_axes_of_symmetry'
    ]
    
    # Shape property definitions
    SHAPE_PROPERTIES = {
        'triangle':      [3, 0, 0, 0, 0, 0, 1, 0],      # General triangle
        'rectangle':     [4, 4, 2, 2, 0, 0, 1, 2],      # 4 right angles, 2 pairs equal
        'square':        [4, 4, 4, 2, 1, 0, 1, 4],      # Regular rectangle
        'parallelogram': [4, 0, 2, 2, 0, 0, 1, 0],      # 2 parallel pairs
        'rhombus':       [4, 0, 4, 2, 0, 0, 1, 2],      # All sides equal
        'trapezoid':     [4, 0, 0, 1, 0, 0, 1, 0],      # 1 parallel pair
        'pentagon':      [5, 0, 5, 0, 1, 0, 1, 5],      # Regular pentagon
        'hexagon':       [6, 0, 6, 3, 1, 0, 1, 6],      # Regular hexagon
        'heptagon':      [7, 0, 7, 0, 1, 0, 1, 7],      # Regular heptagon
        'octagon':       [8, 0, 8, 4, 1, 0, 1, 8],      # Regular octagon
        'circle':        [0, 0, 0, 0, 1, 1, 1, 999],    # Infinite symmetry
        'ellipse':       [0, 0, 0, 0, 0, 1, 1, 2],      # 2 axes of symmetry
    }
    
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
        self.data = []
        self.labels = []
        
        for _ in range(num_samples):
            # Random shape
            shape_idx = np.random.randint(len(self.SHAPES))
            shape = self.SHAPES[shape_idx]
            
            # Get properties and add noise
            props = np.array(self.SHAPE_PROPERTIES[shape], dtype=np.float32)
            props = props + np.random.normal(0, 0.1, size=props.shape)
            
            self.data.append(props)
            self.labels.append(shape_idx)
        
        self.data = np.stack(self.data)
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return self.num_samples
    
    def get_batch(self, batch_size: int):
        indices = np.random.randint(0, len(self), batch_size)
        return (
            torch.tensor(self.data[indices]),
            torch.tensor(self.labels[indices])
        )


def demonstrate_geometric_shapes():
    """Demonstrate the geometric shapes classification task."""
    print("\n" + "=" * 70)
    print("Geometric Shapes Classification: A Nuanced Example")
    print("=" * 70)
    
    dataset = GeometricShapeDataset(num_samples=100)
    
    print("""
TASK: Classify geometric shapes from their properties
──────────────────────────────────────────────────────

Properties used:
1. Number of sides
2. Number of right angles  
3. Number of equal sides
4. Number of parallel pairs
5. Is regular (all sides/angles equal)
6. Has curved edges
7. Is convex
8. Number of axes of symmetry

Shapes to classify:
""")
    
    for i, shape in enumerate(GeometricShapeDataset.SHAPES):
        props = GeometricShapeDataset.SHAPE_PROPERTIES[shape]
        print(f"  {i:2d}. {shape:15s}: {props}")
    
    print("""
This is more nuanced than simple triangle/rectangle because:
- Multiple shapes share properties (rhombus vs square)
- Some properties are noisy/imprecise
- Requires understanding property relationships
""")
    
    # Sample batch
    X, y = dataset.get_batch(5)
    print("\nSample batch:")
    for i in range(5):
        shape = GeometricShapeDataset.SHAPES[y[i]]
        print(f"  Properties: {X[i].numpy().round(2)} → {shape}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all annotated transformer demonstrations."""
    print("\n" + "#" * 70)
    print("# EXAMPLE 7: The Annotated Transformer - Detailed Implementation")
    print("#" * 70)
    print("\nInspired by: https://nlp.seas.harvard.edu/2018/04/03/attention.html")
    
    # Demonstrate components
    print("\n" + "=" * 70)
    print("PART 1: Detailed Attention Implementation")
    print("=" * 70)
    
    # Test attention
    torch.manual_seed(42)
    Q = torch.randn(1, 4, 8)
    K = torch.randn(1, 4, 8)
    V = torch.randn(1, 4, 8)
    
    output, weights = attention_detailed(Q, K, V)
    print(f"\nAttention test:")
    print(f"  Input shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Weights sum to 1: {weights[0, 0].sum():.4f}")
    
    # Multi-head attention
    mha = MultiHeadAttentionAnnotated(d_model=64, num_heads=4)
    x = torch.randn(2, 10, 64)
    out = mha(x, x, x)
    print(f"\nMulti-head attention test:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    
    # Learning rate schedule
    visualize_lr_schedule()
    
    # Label smoothing
    demonstrate_label_smoothing()
    
    # Copy task
    demonstrate_copy_task()
    
    # Geometric shapes
    demonstrate_geometric_shapes()
    
    print("\n" + "=" * 70)
    print("Key Takeaways from The Annotated Transformer:")
    print("=" * 70)
    print("""
1. ATTENTION MECHANICS
   - Scaling by √d_k is crucial for gradient stability
   - Multi-head splits d_model into h parallel attention operations
   
2. LEARNING RATE SCHEDULE
   - Warmup prevents early training instability
   - Inverse square root decay for fine convergence
   
3. LABEL SMOOTHING
   - Prevents overconfident predictions
   - Acts as implicit regularization
   
4. THE COPY TASK
   - Simple sanity check for seq2seq models
   - If it fails, something fundamental is wrong

5. COMPLEX TASKS
   - Real tasks involve nuanced patterns
   - Multiple features interact in complex ways
   - Transformers can learn these relationships
""")


if __name__ == "__main__":
    main()
