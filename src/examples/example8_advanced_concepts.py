#!/usr/bin/env python3
"""
Example 8: Advanced Transformer Concepts (Fleuret's Perspective)

This example covers advanced Transformer concepts inspired by 
François Fleuret's slides (https://fleuret.org/public/EN_20220809-Transformers/).

Key topics covered:
1. Information flow and capacity analysis
2. Attention as soft dictionary lookup
3. Complexity analysis and optimization
4. Inductive biases of Transformers
5. Polygon sequence modeling (advanced geometric task)

Run this example:
   python -m src.examples.example8_advanced_concepts
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import numpy as np
from typing import List, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transformers import MultiHeadAttention, Transformer


# ============================================================================
# PART 1: ATTENTION AS SOFT DICTIONARY LOOKUP
# ============================================================================

def demonstrate_soft_dictionary():
   """
   Show attention as a differentiable dictionary lookup.
   
   Traditional dictionary:
       dict[key] → value  (exact match)
   
   Attention dictionary:
       softmax(query · keys^T) · values  (soft match)
   
   This is key to understanding why Transformers work:
   - Queries "search" through keys
   - Values are returned based on similarity
   - Everything is differentiable → can be learned
   """
   print("=" * 70)
   print("Attention as Soft Dictionary Lookup")
   print("=" * 70)
   
   print("""
╔══════════════════════════════════════════════════════════════════════╗
║              TRADITIONAL vs ATTENTION DICTIONARY                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  TRADITIONAL DICTIONARY (Hard lookup):                                 ║
║  ──────────────────────────────────────────                               ║
║    colors = {"red": [1,0,0], "green": [0,1,0], "blue": [0,0,1]}      ║
║    colors["red"] → [1, 0, 0]  (exact match required)                 ║
║                                                                        ║
║  ATTENTION DICTIONARY (Soft lookup):                                  ║
║  ──────────────────────────────────────                               ║
║    query = "crimson"  (similar to red)                                ║
║    keys = ["red", "green", "blue"]                                    ║
║    values = [[1,0,0], [0,1,0], [0,0,1]]                              ║
║                                                                        ║
║    similarities = [0.8, 0.1, 0.1]  (cosine similarity)               ║
║    result = 0.8×[1,0,0] + 0.1×[0,1,0] + 0.1×[0,0,1]                 ║
║           = [0.8, 0.1, 0.1]  (mostly red!)                           ║
║                                                                        ║
║  KEY INSIGHT: Soft lookup enables:                                    ║
║  • Approximate matching (similar queries, similar results)            ║
║  • Gradient flow (differentiable → learnable)                        ║
║  • Interpolation between stored values                                ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════╝
""")
   
   # Demonstrate with code
   torch.manual_seed(42)
   
   # Create a simple "color" embedding space
   colors = {
       'red': torch.tensor([1.0, 0.0, 0.0]),
       'green': torch.tensor([0.0, 1.0, 0.0]),
       'blue': torch.tensor([0.0, 0.0, 1.0]),
       'yellow': torch.tensor([1.0, 1.0, 0.0]),
       'cyan': torch.tensor([0.0, 1.0, 1.0]),
       'magenta': torch.tensor([1.0, 0.0, 1.0]),
   }
   
   # Stack as keys and values
   keys = torch.stack(list(colors.values()))
   values = keys.clone()  # In this case, values = keys (RGB vectors)
   
   # Query: "orange" (between red and yellow)
   query_orange = torch.tensor([1.0, 0.5, 0.0])
   
   # Compute attention
   scores = torch.matmul(query_orange.unsqueeze(0), keys.T)
   weights = F.softmax(scores * 3, dim=-1)  # Temperature scaling
   result = torch.matmul(weights, values)
   
   print("Color dictionary lookup:")
   print(f"  Query: 'orange' = {query_orange.tolist()}")
   print(f"\n  Attention weights (similarity to each color):")
   for (name, _), w in zip(colors.items(), weights[0].tolist()):
       bar = "█" * int(w * 30)
       print(f"    {name:8s}: {w:.3f} {bar}")
   
   print(f"\n  Retrieved color: {result[0].numpy().round(3)}")
   print("  → Mostly red with some yellow (≈ orange!)")


# ============================================================================
# PART 2: COMPLEXITY ANALYSIS
# ============================================================================

def analyze_complexity():
   """
   Analyze computational complexity of different Transformer components.
   """
   print("\n" + "=" * 70)
   print("Computational Complexity Analysis")
   print("=" * 70)
   
   print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    COMPLEXITY BREAKDOWN                               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  Let: n = sequence length, d = model dimension                        ║
║                                                                        ║
║  1. EMBEDDING LOOKUP:           O(n)                                   ║
║     - Simple table lookup                                              ║
║                                                                        ║
║  2. POSITIONAL ENCODING:        O(n × d)                               ║
║     - Addition of pre-computed values                                  ║
║                                                                        ║
║  3. ATTENTION (per layer):                                             ║
║     ┌────────────────────────────────────────────────────────────┐    ║
║     │ Q, K, V projections:      O(n × d²)                        │    ║
║     │ QK^T computation:         O(n² × d)  ← BOTTLENECK          │    ║
║     │ Softmax:                  O(n²)                            │    ║
║     │ Attention × V:            O(n² × d)                        │    ║
║     │ Output projection:        O(n × d²)                        │    ║
║     └────────────────────────────────────────────────────────────┘    ║
║     Total attention:            O(n² × d + n × d²)                    ║
║                                                                        ║
║  4. FEED-FORWARD (per layer):   O(n × d × d_ff) = O(n × d²)           ║
║     - Usually d_ff = 4d                                                ║
║                                                                        ║
║  5. LAYER NORM:                 O(n × d)                               ║
║                                                                        ║
║  TOTAL per layer:               O(n² × d + n × d²)                    ║
║  TOTAL for L layers:            O(L × (n² × d + n × d²))              ║
║                                                                        ║
║  MEMORY:                                                               ║
║     - Storing attention matrix: O(n²) per head                        ║
║     - Total for h heads:        O(h × n²)                             ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════╝
""")
   
   # Visualize scaling
   fig, axes = plt.subplots(1, 2, figsize=(14, 5))
   
   # Sequence length scaling
   n_values = np.array([100, 500, 1000, 2000, 4000, 8000])
   d = 512
   
   attention_ops = n_values ** 2 * d
   ffn_ops = n_values * d * d
   
   axes[0].plot(n_values, attention_ops / 1e9, 'b-o', label='Attention O(n²d)', linewidth=2)
   axes[0].plot(n_values, ffn_ops / 1e9, 'r-s', label='FFN O(nd²)', linewidth=2)
   axes[0].set_xlabel('Sequence Length (n)')
   axes[0].set_ylabel('Operations (billions)')
   axes[0].set_title('Computational Cost vs Sequence Length\n(d=512)')
   axes[0].legend()
   axes[0].grid(True, alpha=0.3)
   axes[0].set_yscale('log')
   
   # Memory scaling
   memory_attn = n_values ** 2 * 4 / 1e6  # bytes (float32)
   axes[1].bar(range(len(n_values)), memory_attn)
   axes[1].set_xticks(range(len(n_values)))
   axes[1].set_xticklabels(n_values)
   axes[1].set_xlabel('Sequence Length (n)')
   axes[1].set_ylabel('Attention Memory (MB)')
   axes[1].set_title('Memory Required for Attention Matrix\n(Single head, float32)')
   axes[1].grid(True, alpha=0.3, axis='y')
   
   plt.tight_layout()
   plt.savefig('src/tmp/complexity_analysis.png', dpi=150)
   plt.close()
   
   print("\nVisualization saved to src/tmp/complexity_analysis.png")
   print("""
KEY INSIGHTS:
─────────────
1. Attention is O(n²) - QUADRATIC in sequence length
  - n=1000: 1 million operations
  - n=8000: 64 million operations
  
2. This limits context length in practice
  - GPT-2: 1024 tokens
  - GPT-3: 2048 tokens
  - Modern models: 8k-128k (with efficiency tricks)
  
3. Solutions exist:
  - Sparse attention (Longformer, BigBird)
  - Linear attention (Performer)
  - Flash Attention (hardware optimization)
""")


# ============================================================================
# PART 3: INDUCTIVE BIASES
# ============================================================================

def explain_inductive_biases():
   """
   Explain what inductive biases Transformers have (and don't have).
   """
   print("\n" + "=" * 70)
   print("Inductive Biases of Transformers")
   print("=" * 70)
   
   print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    INDUCTIVE BIASES                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  WHAT TRANSFORMERS ASSUME (Inductive biases they HAVE):               ║
║  ─────────────────────────────────────────────────────                ║
║  1. SET-LIKE PROCESSING                                               ║
║     • Without positional encoding, treats input as unordered set      ║
║     • Any permutation gives same attention pattern                    ║
║                                                                        ║
║  2. TRANSLATION EQUIVARIANCE (with positions)                         ║
║     • Same pattern shifted → same output shifted                      ║
║     • "cat sat" and "sat cat" use same attention mechanism            ║
║                                                                        ║
║  3. GLOBAL RECEPTIVE FIELD                                            ║
║     • Every position can attend to every other position               ║
║     • No explicit locality bias (unlike CNNs)                         ║
║                                                                        ║
║  4. PARAMETER SHARING ACROSS POSITIONS                                ║
║     • Same attention weights for all positions                        ║
║     • Same FFN applied to each position                               ║
║                                                                        ║
║  WHAT TRANSFORMERS DON'T ASSUME (biases they LACK):                  ║
║  ─────────────────────────────────────────────────                    ║
║  ✗ No inherent sequential ordering (must add position info)          ║
║  ✗ No locality bias (must learn from data)                           ║
║  ✗ No explicit hierarchy (flat processing)                           ║
║  ✗ No recursion (fixed depth)                                        ║
║                                                                        ║
║  COMPARISON:                                                           ║
║  ┌─────────────┬────────────┬─────────────────────────────┐           ║
║  │ Architecture│ Locality   │ Global View                 │           ║
║  ├─────────────┼────────────┼─────────────────────────────┤           ║
║  │ CNN         │ Strong     │ Needs many layers           │           ║
║  │ RNN         │ Sequential │ Through hidden state        │           ║
║  │ Transformer │ None       │ Immediate (single layer)    │           ║
║  └─────────────┴────────────┴─────────────────────────────┘           ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════╝
""")


# ============================================================================
# PART 4: POLYGON SEQUENCE MODELING (Advanced Geometric Example)
# ============================================================================

class PolygonGenerator:
   """
   Generate sequences of polygon vertices for Transformer modeling.
   
   This is a more nuanced geometric task:
   - Input: sequence of (x, y) coordinates
   - Output: polygon classification or next vertex prediction
   
   Includes: triangles, quadrilaterals, pentagons, hexagons, etc.
   """
   
   POLYGON_TYPES = [
       'equilateral_triangle', 'isosceles_triangle', 'scalene_triangle',
       'square', 'rectangle', 'parallelogram', 'rhombus', 'trapezoid',
       'regular_pentagon', 'regular_hexagon', 'regular_heptagon', 'regular_octagon',
       'irregular_quadrilateral', 'irregular_pentagon', 'star_5', 'star_6'
   ]
   
   @staticmethod
   def generate_regular_polygon(n_sides: int, center: Tuple[float, float] = (0, 0), 
                                 radius: float = 1.0, rotation: float = 0.0) -> np.ndarray:
       """Generate vertices of a regular polygon."""
       angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + rotation
       x = center[0] + radius * np.cos(angles)
       y = center[1] + radius * np.sin(angles)
       return np.stack([x, y], axis=1)
   
   @staticmethod
   def generate_star(n_points: int, outer_radius: float = 1.0, 
                     inner_radius: float = 0.4) -> np.ndarray:
       """Generate vertices of a star polygon."""
       angles = np.linspace(0, 2 * np.pi, 2 * n_points, endpoint=False)
       radii = np.array([outer_radius, inner_radius] * n_points)
       x = radii * np.cos(angles)
       y = radii * np.sin(angles)
       return np.stack([x, y], axis=1)
   
   @classmethod
   def generate_polygon(cls, polygon_type: str) -> Tuple[np.ndarray, str]:
       """Generate a polygon of the specified type."""
       rotation = np.random.uniform(0, 2 * np.pi)
       scale = np.random.uniform(0.5, 1.5)
       
       if polygon_type == 'equilateral_triangle':
           vertices = cls.generate_regular_polygon(3, rotation=rotation) * scale
       elif polygon_type == 'isosceles_triangle':
           vertices = np.array([[0, 1], [-0.7, -0.5], [0.7, -0.5]]) * scale
       elif polygon_type == 'scalene_triangle':
           vertices = np.array([[0, 1], [-0.5, -0.3], [0.8, -0.6]]) * scale
       elif polygon_type == 'square':
           vertices = cls.generate_regular_polygon(4, rotation=rotation) * scale
       elif polygon_type == 'rectangle':
           vertices = np.array([[-1, -0.5], [1, -0.5], [1, 0.5], [-1, 0.5]]) * scale
       elif polygon_type == 'parallelogram':
           vertices = np.array([[-0.8, -0.4], [0.8, -0.4], [1.2, 0.4], [-0.4, 0.4]]) * scale
       elif polygon_type == 'rhombus':
           vertices = np.array([[0, 1], [-0.6, 0], [0, -1], [0.6, 0]]) * scale
       elif polygon_type == 'trapezoid':
           vertices = np.array([[-1, -0.5], [1, -0.5], [0.6, 0.5], [-0.6, 0.5]]) * scale
       elif polygon_type == 'regular_pentagon':
           vertices = cls.generate_regular_polygon(5, rotation=rotation) * scale
       elif polygon_type == 'regular_hexagon':
           vertices = cls.generate_regular_polygon(6, rotation=rotation) * scale
       elif polygon_type == 'regular_heptagon':
           vertices = cls.generate_regular_polygon(7, rotation=rotation) * scale
       elif polygon_type == 'regular_octagon':
           vertices = cls.generate_regular_polygon(8, rotation=rotation) * scale
       elif polygon_type == 'irregular_quadrilateral':
           vertices = np.random.randn(4, 2) * scale
       elif polygon_type == 'irregular_pentagon':
           vertices = np.random.randn(5, 2) * scale
       elif polygon_type == 'star_5':
           vertices = cls.generate_star(5) * scale
       elif polygon_type == 'star_6':
           vertices = cls.generate_star(6) * scale
       else:
           vertices = cls.generate_regular_polygon(3) * scale
       
       # Apply rotation
       c, s = np.cos(rotation), np.sin(rotation)
       R = np.array([[c, -s], [s, c]])
       vertices = vertices @ R.T
       
       return vertices, polygon_type


def demonstrate_polygon_modeling():
   """
   Demonstrate polygon sequence modeling with Transformers.
   """
   print("\n" + "=" * 70)
   print("Polygon Sequence Modeling: Advanced Geometric Task")
   print("=" * 70)
   
   print("""
TASK: Model sequences of polygon vertices
──────────────────────────────────────────

This is more nuanced than simple shape classification:

1. INPUT: Sequence of (x, y) vertex coordinates
  Example: [(0,1), (-0.87,-0.5), (0.87,-0.5)]  → Triangle

2. CHALLENGES:
  • Variable number of vertices
  • Rotation invariance (same shape, different orientation)
  • Scale invariance (same shape, different size)
  • Order matters (vertices in sequence around polygon)

3. TASKS THE TRANSFORMER CAN LEARN:
  • Classify polygon type
  • Predict next vertex
  • Complete partial polygons
  • Detect if shape is convex
  • Compute area/perimeter

4. WHY TRANSFORMERS ARE GOOD FOR THIS:
  • Self-attention captures relationships between ALL vertices
  • Can learn that vertex 0 relates to vertex n-1 (closing the polygon)
  • Position encoding indicates vertex order
""")
   
   # Generate and visualize examples
   fig, axes = plt.subplots(2, 4, figsize=(16, 8))
   
   polygon_types = [
       'equilateral_triangle', 'square', 'regular_pentagon', 'regular_hexagon',
       'parallelogram', 'rhombus', 'star_5', 'irregular_quadrilateral'
   ]
   
   for idx, ptype in enumerate(polygon_types):
       ax = axes[idx // 4, idx % 4]
       
       vertices, _ = PolygonGenerator.generate_polygon(ptype)
       
       # Close the polygon for plotting
       vertices_closed = np.vstack([vertices, vertices[0]])
       
       ax.plot(vertices_closed[:, 0], vertices_closed[:, 1], 'b-', linewidth=2)
       ax.scatter(vertices[:, 0], vertices[:, 1], c='red', s=50, zorder=5)
       
       # Label vertices
       for i, (x, y) in enumerate(vertices):
           ax.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
       
       ax.set_title(ptype.replace('_', ' ').title(), fontsize=10)
       ax.set_aspect('equal')
       ax.grid(True, alpha=0.3)
       ax.set_xlim(-2, 2)
       ax.set_ylim(-2, 2)
   
   plt.suptitle('Polygon Types for Sequence Modeling\n(Vertices numbered in sequence order)', 
                fontsize=14, y=1.02)
   plt.tight_layout()
   plt.savefig('src/tmp/polygon_examples.png', dpi=150, bbox_inches='tight')
   plt.close()
   
   print("\nVisualization saved to src/tmp/polygon_examples.png")
   
   # Show sequence representation
   print("\nSequence representation example:")
   vertices, ptype = PolygonGenerator.generate_polygon('regular_pentagon')
   print(f"  Shape: {ptype}")
   print(f"  Vertices as sequence:")
   for i, (x, y) in enumerate(vertices):
       print(f"    Position {i}: ({x:.3f}, {y:.3f})")


# ============================================================================
# PART 5: ATTENTION PATTERN ANALYSIS
# ============================================================================

def analyze_attention_patterns():
   """
   Analyze what attention patterns might emerge for polygon modeling.
   """
   print("\n" + "=" * 70)
   print("Attention Patterns for Geometric Sequences")
   print("=" * 70)
   
   print("""
For polygon vertex sequences, attention might learn:

1. ADJACENT VERTEX ATTENTION
  • Vertex i attends strongly to vertices i-1 and i+1
  • Captures local edge information
  
2. OPPOSITE VERTEX ATTENTION  
  • In regular polygons, opposite vertices are related
  • Square: vertex 0 ↔ vertex 2
  
3. SYMMETRY DETECTION
  • Similar attention patterns for symmetric vertices
  • Helps identify regular polygons
  
4. CLOSURE ATTENTION
  • Last vertex attends to first vertex
  • Ensures polygon is closed
""")
   
   # Simulate attention patterns
   n_vertices = 6  # Hexagon
   
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   
   # Pattern 1: Adjacent attention
   adj_attn = np.zeros((n_vertices, n_vertices))
   for i in range(n_vertices):
       adj_attn[i, (i - 1) % n_vertices] = 0.4
       adj_attn[i, i] = 0.3
       adj_attn[i, (i + 1) % n_vertices] = 0.3
   
   im1 = axes[0, 0].imshow(adj_attn, cmap='Blues', vmin=0, vmax=1)
   axes[0, 0].set_title('Adjacent Vertex Attention\n(Local edge relationships)')
   axes[0, 0].set_xlabel('Key (Vertex)')
   axes[0, 0].set_ylabel('Query (Vertex)')
   plt.colorbar(im1, ax=axes[0, 0])
   
   # Pattern 2: Opposite attention (for hexagon)
   opp_attn = np.eye(n_vertices) * 0.5
   for i in range(n_vertices):
       opp_attn[i, (i + n_vertices // 2) % n_vertices] = 0.5
   
   im2 = axes[0, 1].imshow(opp_attn, cmap='Blues', vmin=0, vmax=1)
   axes[0, 1].set_title('Opposite Vertex Attention\n(Symmetry detection)')
   axes[0, 1].set_xlabel('Key (Vertex)')
   axes[0, 1].set_ylabel('Query (Vertex)')
   plt.colorbar(im2, ax=axes[0, 1])
   
   # Pattern 3: Global attention
   global_attn = np.ones((n_vertices, n_vertices)) / n_vertices
   
   im3 = axes[1, 0].imshow(global_attn, cmap='Blues', vmin=0, vmax=1)
   axes[1, 0].set_title('Global Attention\n(Overall shape understanding)')
   axes[1, 0].set_xlabel('Key (Vertex)')
   axes[1, 0].set_ylabel('Query (Vertex)')
   plt.colorbar(im3, ax=axes[1, 0])
   
   # Pattern 4: Closure attention
   closure_attn = np.eye(n_vertices) * 0.6
   closure_attn[-1, 0] = 0.4  # Last to first
   closure_attn[0, -1] = 0.4  # First to last
   
   im4 = axes[1, 1].imshow(closure_attn, cmap='Blues', vmin=0, vmax=1)
   axes[1, 1].set_title('Closure Attention\n(Polygon completion)')
   axes[1, 1].set_xlabel('Key (Vertex)')
   axes[1, 1].set_ylabel('Query (Vertex)')
   plt.colorbar(im4, ax=axes[1, 1])
   
   plt.suptitle('Hypothetical Attention Patterns for Hexagon Vertices', fontsize=14, y=1.02)
   plt.tight_layout()
   plt.savefig('src/tmp/polygon_attention.png', dpi=150, bbox_inches='tight')
   plt.close()
   
   print("\nVisualization saved to src/tmp/polygon_attention.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
   """Run all advanced concept demonstrations."""
   print("\n" + "#" * 70)
   print("# EXAMPLE 8: Advanced Transformer Concepts")
   print("#" * 70)
   print("\nInspired by: Fleuret's Transformer slides")
   print("https://fleuret.org/public/EN_20220809-Transformers/transformers-slides.pdf")
   
   # Run demonstrations
   demonstrate_soft_dictionary()
   analyze_complexity()
   explain_inductive_biases()
   demonstrate_polygon_modeling()
   analyze_attention_patterns()
   
   print("\n" + "=" * 70)
   print("Key Takeaways from Advanced Concepts:")
   print("=" * 70)
   print("""
1. ATTENTION AS SOFT DICTIONARY
  - Queries search through keys
  - Values returned based on similarity
  - Differentiable → learnable

2. COMPLEXITY MATTERS
  - O(n²) attention limits context length
  - Memory for attention matrix can be huge
  - Various efficiency techniques exist

3. INDUCTIVE BIASES
  - Transformers have few built-in assumptions
  - Must learn everything from data
  - Great flexibility, but needs more data

4. GEOMETRIC SEQUENCES
  - Polygons are a rich testbed for Transformers
  - Vertices as sequence of (x,y) coordinates
  - Attention can capture vertex relationships

5. ATTENTION PATTERNS
  - Different heads learn different patterns
  - Local (adjacent), global, symmetric patterns
  - Emerge naturally during training
""")


if __name__ == "__main__":
   main()
