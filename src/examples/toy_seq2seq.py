#!/usr/bin/env python3
"""
Toy Seq2Seq - Signal Copying/Smoothing with Transformers

Tâche SIMPLIFIÉE: Copier et lisser un signal
- Input: signal avec pics bruités
- Output: signal lissé (version propre du signal)

C'est plus adapté aux capacités du Transformer!

Run this example:
    python -m src.examples.toy_seq2seq
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
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
        noise_std: float = 0.3,  # Très peu de bruit
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
        """Génère un échantillon avec des pics"""
        input_signal = np.zeros(self.seq_len)
        output_signal = np.zeros(self.seq_len)
        
        # Espacement minimum entre pics
        min_spacing = self.peak_width + 20
        all_positions = []
        
        # Pics de type A (triangles)
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
            
            # PIC TRIANGLE (forme pointue) ▲
            half_width = self.peak_width // 2
            for i in range(-half_width, half_width + 1):
                if 0 <= pos + i < self.seq_len:
                    # Forme triangulaire: hauteur décroît linéairement depuis le centre
                    distance_from_center = abs(i)
                    triangle_height = height * (1 - distance_from_center / (half_width + 1))
                    input_signal[pos + i] = max(0, triangle_height)
        
        # Pics de type B (carrés) - FORME CARRÉE/PLATE ■
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
            
            # PIC CARRÉ (plateau plat) ■
            half_width = self.peak_width // 2
            for i in range(-half_width, half_width + 1):
                if 0 <= pos + i < self.seq_len:
                    input_signal[pos + i] = height
        
        # Calculer les moyennes par type
        avg_a = np.mean(type_a_heights) if type_a_heights else 0
        avg_b = np.mean(type_b_heights) if type_b_heights else 0
        
        # Signal de sortie: MOYENNES aux positions des pics
        # Les triangles restent des triangles, les carrés restent des carrés
        for pos in type_a_positions:
            half_width = self.peak_width // 2
            for i in range(-half_width, half_width + 1):
                if 0 <= pos + i < self.seq_len:
                    # Triangle avec hauteur moyenne
                    distance_from_center = abs(i)
                    triangle_height = avg_a * (1 - distance_from_center / (half_width + 1))
                    output_signal[pos + i] = max(0, triangle_height)
        
        for pos in type_b_positions:
            half_width = self.peak_width // 2
            for i in range(-half_width, half_width + 1):
                if 0 <= pos + i < self.seq_len:
                    # Carré avec hauteur moyenne
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
    Transformer avec PATCHING pour transformation de signaux.
    
    Idée clé:
    - Au lieu de traiter chaque point comme un token (naïf et inefficace)
    - On regroupe les points en "patches" (morceaux de signal)
    - Chaque patch capture le contexte local (forme du signal)
    - Réduit le nombre de tokens (moins de calcul d'attention)
    
    Exemple avec seq_len=100 et patch_size=5:
    - Ancien: 100 tokens de dimension 1 (naïf)
    - Nouveau: 20 tokens de dimension 5 (efficace)
    
    Architecture encoder-only car input et output ont la même taille.
    """
    
    def __init__(
        self,
        seq_len: int = 100,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        patch_size: int = 5  # NOUVEAU: taille des patches
    ):
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.patch_size = patch_size
        
        # Nombre de patches (tokens)
        assert seq_len % patch_size == 0, f"seq_len ({seq_len}) doit être divisible par patch_size ({patch_size})"
        self.num_patches = seq_len // patch_size
        
        # Projection: patch_size → d_model (au lieu de 1 → d_model)
        # Chaque patch contient patch_size points consécutifs
        self.input_proj = nn.Linear(patch_size, d_model)
        
        # Positional encoding appris (sur les patches, pas les points!)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection sortie: d_model → patch_size (reconstruit le patch)
        self.output_proj = nn.Linear(d_model, patch_size)
    
    def forward(self, x):
        """
        x: [batch, seq_len]
        output: [batch, seq_len]
        
        Processus:
        1. Découper le signal en patches
        2. Projeter chaque patch vers d_model dimensions
        3. Ajouter positional encoding
        4. Encoder (self-attention entre patches)
        5. Projeter vers patch_size
        6. Reconstruire le signal complet
        """
        batch_size = x.shape[0]
        
        # 1. Découper en patches: [batch, seq_len] → [batch, num_patches, patch_size]
        x = x.view(batch_size, self.num_patches, self.patch_size)
        
        # 2. Projection: [batch, num_patches, patch_size] → [batch, num_patches, d_model]
        x = self.input_proj(x)
        
        # 3. Ajouter position (chaque patch sait où il est dans le signal)
        x = x + self.pos_embedding
        
        # 4. Encoder: self-attention entre patches
        x = self.encoder(x)
        
        # 5. Projection sortie: [batch, num_patches, d_model] → [batch, num_patches, patch_size]
        x = self.output_proj(x)
        
        # 6. Reconstruire: [batch, num_patches, patch_size] → [batch, seq_len]
        x = x.view(batch_size, self.seq_len)
        
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
            for pos in peaks.get('type_a_positions', []) + peaks.get('type_b_positions', []):
                ax.axvline(x=pos, color='gray', linestyle=':', alpha=0.5)
            
            ax.set_title(f'Exemple {i+1}')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-2, 30)  # Adapté aux pics de hauteur 5-25
    
    plt.suptitle('Transformation de Signal avec Transformer\nBleu=Input, Vert=Target (moyenné), Rouge=Prédit', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualisation sauvegardée: {save_path}")
    plt.close()





def main():
    """Démonstration complète du Toy Seq2Seq."""
    
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
    print(f"\n Device: {device}")
    
    output_dir = Path(__file__).parent.parent / 'tmp'
    output_dir.mkdir(exist_ok=True)
    
    # Dataset avec pics NETS
    print("\n Création du dataset (pics nets comme dans l'image)...")
    train_dataset = SignalDataset(
        num_samples=2000, 
        seq_len=100, 
        peak_width=8,      # Pics larges et nets
        noise_std=0.6,     # bruit
        seed=42
    )
    val_dataset = SignalDataset(
        num_samples=200, 
        seq_len=100, 
        peak_width=8,
        noise_std=0.6,
        seed=123
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
    
    # Modèle
    print("\n Création du modèle...")
    model = SimpleSignalTransformer(
        seq_len=100,      # Doit correspondre au seq_len du dataset!
        d_model=64,
        num_heads=4,
        d_ff=256,
        num_layers=3,
        dropout=0.1,
        patch_size=5      # PATCHING: 100 points → 20 patches de 5 points
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
    print(f"\n Résumé:")
    print(f"   Loss initiale: {train_losses[0]:.4f}")
    print(f"   Loss finale: {train_losses[-1]:.4f}")
    reduction = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
    print(f"   Réduction: {reduction:.1f}%")
    
    if reduction > 50:
        print("   Excellent! Le modèle a bien appris!")
    elif reduction > 20:
        print("   Bon progrès!")
    else:
        print("    La réduction est faible, essayez plus d'epochs")
    
    # Visualisations
    print("\n" \
    " Génération des visualisations...")
    
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
    print(f"   {output_dir / 'toy_seq2seq_training.png'}")
    
    # 2. Résultats
    visualize_results(model, val_dataset, device, num_samples=4, 
                     save_path=output_dir / 'toy_seq2seq_results.png')
    
    print("\n" + "=" * 70)
    print(" Terminé!")
    print("=" * 70)


if __name__ == "__main__":
    main()
