#!/usr/bin/env python3
"""
Example 9: Toy Seq2Seq - Signal Copying/Smoothing with Transformers

Tâche SIMPLIFIÉE: Copier et lisser un signal
- Input: signal avec pics bruités
- Output: signal lissé (version propre du signal)

C'est plus adapté aux capacités du Transformer!

Run this example:
    python -m src.examples.example9_toy_seq2seq
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import random

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# DATASET SIMPLIFIÉ: Signal avec bruit → Signal propre
# ============================================================================

class SignalDataset(Dataset):
    """
    Dataset pour la tâche de transformation de signaux.
    
    REPRODUIT FIDÈLEMENT L'IMAGE "Toy seq2seq example":
    - Input: Signal avec pics NETS (pas gaussiens)
    - Output: Signal transformé
    - Pics bien séparés et distincts
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        seq_len: int = 100,
        num_peaks_per_type: int = 2,
        min_height: float = 5.0,
        max_height: float = 25.0,
        peak_width: int = 8,  # Largeur des pics (carrés/plats)
        noise_std: float = 0.1,  # Très peu de bruit
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_peaks_per_type = num_peaks_per_type
        self.min_height = min_height
        self.max_height = max_height
        self.peak_width = peak_width
        self.noise_std = noise_std
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.data = []
        
        for _ in range(num_samples):
            input_signal, output_signal, peak_info = self._generate_sample()
            self.data.append((input_signal, output_signal, peak_info))
    
    def _generate_sample(self):
        """Génère un échantillon avec des pics NETS comme dans l'image."""
        input_signal = np.zeros(self.seq_len)
        output_signal = np.zeros(self.seq_len)
        
        # Espacement minimum entre pics
        min_spacing = self.peak_width + 10
        all_positions = []
        
        # Pics de type A (triangles) - FORME CARRÉE/PLATE
        type_a_positions = []
        type_a_heights = []
        for _ in range(self.num_peaks_per_type):
            attempts = 0
            while attempts < 100:
                pos = random.randint(self.peak_width + 5, self.seq_len - self.peak_width - 5)
                if all(abs(pos - p) >= min_spacing for p in all_positions):
                    break
                attempts += 1
            
            height = random.uniform(self.min_height, self.max_height)
            type_a_positions.append(pos)
            type_a_heights.append(height)
            all_positions.append(pos)
            
            # PIC CARRÉ/PLAT (comme dans l'image)
            half_width = self.peak_width // 2
            for i in range(-half_width, half_width + 1):
                if 0 <= pos + i < self.seq_len:
                    input_signal[pos + i] = height
        
        # Pics de type B (carrés) - FORME CARRÉE/PLATE
        type_b_positions = []
        type_b_heights = []
        for _ in range(self.num_peaks_per_type):
            attempts = 0
            while attempts < 100:
                pos = random.randint(self.peak_width + 5, self.seq_len - self.peak_width - 5)
                if all(abs(pos - p) >= min_spacing for p in all_positions):
                    break
                attempts += 1
            
            height = random.uniform(self.min_height, self.max_height)
            type_b_positions.append(pos)
            type_b_heights.append(height)
            all_positions.append(pos)
            
            # PIC CARRÉ/PLAT
            half_width = self.peak_width // 2
            for i in range(-half_width, half_width + 1):
                if 0 <= pos + i < self.seq_len:
                    input_signal[pos + i] = height
        
        # Calculer les moyennes par type
        avg_a = np.mean(type_a_heights) if type_a_heights else 0
        avg_b = np.mean(type_b_heights) if type_b_heights else 0
        
        # Signal de sortie: MOYENNES aux positions des pics
        for pos in type_a_positions:
            half_width = self.peak_width // 2
            for i in range(-half_width, half_width + 1):
                if 0 <= pos + i < self.seq_len:
                    output_signal[pos + i] = avg_a
        
        for pos in type_b_positions:
            half_width = self.peak_width // 2
            for i in range(-half_width, half_width + 1):
                if 0 <= pos + i < self.seq_len:
                    output_signal[pos + i] = avg_b
        
        # Très peu de bruit (signal propre comme dans l'image)
        if self.noise_std > 0:
            input_signal += np.random.normal(0, self.noise_std, self.seq_len)
        
        peak_info = {
            'type_a_positions': type_a_positions,
            'type_a_heights': type_a_heights,
            'type_b_positions': type_b_positions,
            'type_b_heights': type_b_heights,
            'avg_a': avg_a,
            'avg_b': avg_b
        }
        
        return (
            torch.tensor(input_signal, dtype=torch.float32),
            torch.tensor(output_signal, dtype=torch.float32),
            peak_info
        )
    
    def __len__(self):
        """Retourne le nombre d'échantillons."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Retourne un échantillon."""
        return self.data[idx]


def collate_fn(batch):
    noisy = torch.stack([b[0] for b in batch])
    clean = torch.stack([b[1] for b in batch])
    peaks = [b[2] for b in batch]
    return noisy, clean, peaks


# ============================================================================
# MODÈLE SIMPLE: Encoder-only pour transformation signal → signal
# ============================================================================

class SimpleSignalTransformer(nn.Module):
    """
    Transformer SIMPLE pour transformation de signaux.
    
    Architecture encoder-only car input et output ont la même taille.
    """
    
    def __init__(
        self,
        seq_len: int = 64,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Projection simple: 1 → d_model
        self.input_proj = nn.Linear(1, d_model)
        
        # Positional encoding appris
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection sortie: d_model → 1
        self.output_proj = nn.Linear(d_model, 1)
    
    def forward(self, x):
        """
        x: [batch, seq_len]
        output: [batch, seq_len]
        """
        # [batch, seq_len] → [batch, seq_len, 1]
        x = x.unsqueeze(-1)
        
        # [batch, seq_len, 1] → [batch, seq_len, d_model]
        x = self.input_proj(x)
        
        # Ajouter position
        x = x + self.pos_embedding
        
        # Encoder
        x = self.encoder(x)
        
        # [batch, seq_len, d_model] → [batch, seq_len, 1] → [batch, seq_len]
        x = self.output_proj(x).squeeze(-1)
        
        return x


# ============================================================================
# ENTRAÎNEMENT ET VISUALISATION
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for noisy, clean, _ in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for noisy, clean, _ in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            loss = criterion(output, clean)
            total_loss += loss.item()
    return total_loss / len(loader)


def visualize_results(model, dataset, device, num_samples=4, save_path=None):
    """Visualise input bruité, output prédit, et target propre."""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    
    with torch.no_grad():
        for i in range(num_samples):
            noisy, clean, peaks = dataset[i]
            
            output = model(noisy.unsqueeze(0).to(device)).squeeze(0).cpu()
            
            ax = axes[i]
            x = np.arange(len(noisy))
            
            ax.plot(x, noisy.numpy(), 'b-', alpha=0.5, label='Input (bruité)', linewidth=1)
            ax.plot(x, clean.numpy(), 'g-', label='Target (propre)', linewidth=2)
            ax.plot(x, output.numpy(), 'r--', label='Prédit', linewidth=2)
            
            # Marquer les pics
            for pos in peaks:
                ax.axvline(x=pos, color='gray', linestyle=':', alpha=0.5)
            
            ax.set_title(f'Exemple {i+1}')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.5, 1.5)
    
    plt.suptitle('Débruitage de Signal avec Transformer\nBleu=Input bruité, Vert=Target, Rouge=Prédit', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualisation sauvegardée: {save_path}")
    plt.close()


def visualize_predictions(model, dataset, device, num_samples: int = 2, save_path: str = None):
    """Visualise les prédictions du modèle - STYLE DE L'IMAGE ORIGINALE."""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        input_signal, target_signal, peak_info = dataset[i]
        
        with torch.no_grad():
            src = input_signal.unsqueeze(0).to(device)
            output = model(src).squeeze(0).cpu()
            attn_weights = model.get_attention_weights(src).squeeze(0).cpu()
        
        x = np.arange(len(input_signal))
        seq_len = len(input_signal)
        
        # === Graphique 1: Signaux (style de l'image) ===
        ax1 = axes[i, 0]
        
        # Input en BLEU
        ax1.plot(x, input_signal.numpy(), 'b-', label='Input', linewidth=1.5)
        
        # Output en ORANGE/JAUNE
        ax1.plot(x, output.numpy(), color='orange', label='Output', linewidth=1.5)
        
        # Marqueurs sur l'axe X (comme dans l'image)
        y_marker = -1  # Position Y pour les marqueurs
        for pos in peak_info['type_a_positions']:
            ax1.plot(pos, y_marker, 'k^', markersize=12, markerfacecolor='black')
        for pos in peak_info['type_b_positions']:
            ax1.plot(pos, y_marker, 'ks', markersize=10, markerfacecolor='black')
        
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Valeur')
        ax1.set_title(f'Input (bleu) vs Output (orange)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, seq_len)
        ax1.set_ylim(-3, 30)
        
        # === Graphique 2: Matrice d'attention (style de l'image) ===
        ax2 = axes[i, 1]
        
        # Matrice d'attention en niveaux de gris INVERSÉ (blanc = forte attention)
        im = ax2.imshow(attn_weights.numpy(), cmap='gray_r', aspect='equal', 
                       vmin=0, vmax=attn_weights.max())
        
        # Marqueurs sur les axes (comme dans l'image)
        # Axe X (en bas)
        for pos in peak_info['type_a_positions']:
            ax2.plot(pos, seq_len + 3, 'k^', markersize=8, clip_on=False)
        for pos in peak_info['type_b_positions']:
            ax2.plot(pos, seq_len + 3, 'ks', markersize=7, clip_on=False)
        
        # Axe Y (à gauche)
        for pos in peak_info['type_a_positions']:
            ax2.plot(-3, pos, 'k^', markersize=8, clip_on=False)
        for pos in peak_info['type_b_positions']:
            ax2.plot(-3, pos, 'ks', markersize=7, clip_on=False)
        
        ax2.set_xlabel('Key Position')
        ax2.set_ylabel('Query Position')
        ax2.set_title('Attention Pattern')
        ax2.set_xlim(-0.5, seq_len - 0.5)
        ax2.set_ylim(seq_len - 0.5, -0.5)
    
    plt.suptitle('Toy seq2seq example', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualisation sauvegardée: {save_path}")
    
    plt.close()


def visualize_attention_analysis(model, dataset, device, save_path: str = None):
    """Analyse détaillée des patterns d'attention."""
    model.eval()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Prendre 4 exemples
    examples = [0, 1, 2, 3]
    
    for idx, i in enumerate(examples):
        ax = axes[idx // 2, idx % 2]
        
        input_signal, target_signal, peak_info = dataset[i]
        seq_len = len(input_signal)
        
        with torch.no_grad():
            src = input_signal.unsqueeze(0).to(device)
            attn_weights = model.get_attention_weights(src).squeeze(0).cpu().numpy()
        
        im = ax.imshow(attn_weights, cmap='gray_r', aspect='equal')
        
        # Marquer les positions des pics avec des lignes claires
        for pos in peak_info['type_a_positions']:
            ax.axvline(x=pos, color='blue', linestyle='-', alpha=0.8, linewidth=2, label='Type A' if pos == peak_info['type_a_positions'][0] else '')
            ax.axhline(y=pos, color='blue', linestyle='-', alpha=0.8, linewidth=2)
        
        for pos in peak_info['type_b_positions']:
            ax.axvline(x=pos, color='red', linestyle='-', alpha=0.8, linewidth=2, label='Type B' if pos == peak_info['type_b_positions'][0] else '')
            ax.axhline(y=pos, color='red', linestyle='-', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title(f'Exemple {i+1}\n'
                     f'Type A: {peak_info["type_a_positions"]}, '
                     f'Type B: {peak_info["type_b_positions"]}')
        ax.set_xlim(-0.5, seq_len - 0.5)
        ax.set_ylim(seq_len - 0.5, -0.5)
        
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Analyse des Patterns d\'Attention\n'
                 'Lignes bleues = pics Type A (triangles), Lignes rouges = pics Type B (carrés)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualisation sauvegardée: {save_path}")
    
    plt.close()


def main():
    """Démonstration complète du Toy Seq2Seq."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 9: Toy Seq2Seq - Signal Processing with Transformers")
    print("=" * 70)
    
    print("""
Tâche: Transformer des signaux avec pics (STYLE DE L'IMAGE)
──────────────────────────────────────────────────────────

Cette tâche reproduit l'image "Toy seq2seq example":
- Input: Signal bleu avec pics NETS (carrés/plats)
- Output: Signal orange (pics moyennés par type)
- Attention: Matrice en niveaux de gris
- Marqueurs: ▲ triangles et ■ carrés aux positions des pics
""")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    output_dir = Path(__file__).parent.parent / 'tmp'
    output_dir.mkdir(exist_ok=True)
    
    # Dataset avec pics NETS
    print("\n📚 Création du dataset (pics nets comme dans l'image)...")
    train_dataset = SignalDataset(
        num_samples=2000, 
        seq_len=100, 
        peak_width=8,      # Pics larges et nets
        noise_std=0.1,     # Très peu de bruit
        seed=42
    )
    val_dataset = SignalDataset(
        num_samples=200, 
        seq_len=100, 
        peak_width=8,
        noise_std=0.1,
        seed=123
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
    
    # Modèle
    print("\n🧠 Création du modèle...")
    model = SimpleSignalTransformer(
        seq_len=100,      # Doit correspondre au seq_len du dataset!
        d_model=64,
        num_heads=4,
        d_ff=256,
        num_layers=3,
        dropout=0.1
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Paramètres: {num_params:,}")
    
    # Entraînement
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    print("\n" + "=" * 70)
    print("ENTRAÎNEMENT")
    print("=" * 70)
    
    train_losses, val_losses = [], []
    
    for epoch in range(30):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/30: train={train_loss:.4f}, val={val_loss:.4f}")
    
    # Résumé
    print(f"\n📊 Résumé:")
    print(f"   Loss initiale: {train_losses[0]:.4f}")
    print(f"   Loss finale: {train_losses[-1]:.4f}")
    reduction = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
    print(f"   Réduction: {reduction:.1f}%")
    
    if reduction > 50:
        print("   ✅ Excellent! Le modèle a bien appris!")
    elif reduction > 20:
        print("   ✓ Bon progrès!")
    else:
        print("   ⚠️ La réduction est faible, essayez plus d'epochs")
    
    # Visualisations
    print("\n📊 Génération des visualisations...")
    
    # 1. Courbe de training
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, 'b-', label='Train')
    ax.plot(val_losses, 'orange', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'toy_seq2seq_training.png', dpi=150)
    plt.close()
    print(f"   ✅ {output_dir / 'toy_seq2seq_training.png'}")
    
    # 2. Résultats
    visualize_results(model, val_dataset, device, num_samples=4, 
                     save_path=output_dir / 'toy_seq2seq_results.png')
    
    print("\n" + "=" * 70)
    print("✅ Terminé!")
    print("=" * 70)


if __name__ == "__main__":
    main()
