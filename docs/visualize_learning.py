#!/usr/bin/env python3
"""
VISUALISATION DE L'APPRENTISSAGE - EXAMPLE 9 TOY SEQ2SEQ
========================================================

Ce script montre EN DÉTAIL comment le modèle apprend à transformer
des signaux avec pics en signaux avec pics moyennés.

Exécuter avec:
    python -m docs.visualize_learning
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.examples.toy_seq2seq import (
    SignalDataset,
    SimpleSignalTransformer,
    collate_fn,
    train_epoch,
    evaluate
)


def test_model_on_samples(model, dataset, device, num_samples=4):
    """Teste le modèle sur quelques échantillons et retourne les résultats."""
    model.eval()
    results = []
    
    with torch.no_grad():
        for i in range(num_samples):
            input_signal, target_signal, peak_info = dataset[i]
            
            output = model(input_signal.unsqueeze(0).to(device)).squeeze(0).cpu()
            
            mse = ((output - target_signal) ** 2).mean().item()
            
            results.append({
                'input': input_signal,
                'target': target_signal,
                'output': output,
                'peak_info': peak_info,
                'mse': mse
            })
    
    return results


def visualize_epoch_predictions(results, epoch, save_path=None):
    """Visualise les prédictions pour une epoch donnée."""
    num_samples = len(results)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i, result in enumerate(results):
        ax = axes[i]
        x = np.arange(len(result['input']))
        
        # Plot signals
        ax.plot(x, result['input'].numpy(), 'b-', alpha=0.6, label='Input', linewidth=1)
        ax.plot(x, result['target'].numpy(), 'g-', label='Target', linewidth=2)
        ax.plot(x, result['output'].numpy(), 'r--', label='Prédit', linewidth=2)
        
        # Marquer les pics
        for pos in result['peak_info']['type_a_positions']:
            ax.axvline(x=pos, color='blue', linestyle=':', alpha=0.3)
        for pos in result['peak_info']['type_b_positions']:
            ax.axvline(x=pos, color='red', linestyle=':', alpha=0.3)
        
        ax.set_title(f"Exemple {i+1} - MSE: {result['mse']:.4f}")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2, 30)
    
    plt.suptitle(f'Prédictions à l\'Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_learning_progress(epoch_data, save_path=None):
    """Visualise la progression de l'apprentissage."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = [d['epoch'] for d in epoch_data]
    train_losses = [d['train_loss'] for d in epoch_data]
    val_losses = [d['val_loss'] for d in epoch_data]
    avg_mses = [d['avg_mse'] for d in epoch_data]
    
    # 1. Courbes de loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, val_losses, 'orange', marker='o', label='Val Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Évolution de la Loss', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. MSE sur échantillons de test
    ax2 = axes[0, 1]
    ax2.plot(epochs, avg_mses, 'g-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MSE moyen (test)', fontsize=12)
    ax2.set_title('Qualité des Prédictions', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    # 3. Amélioration relative
    ax3 = axes[1, 0]
    if len(train_losses) > 1:
        improvements = [(train_losses[0] - l) / train_losses[0] * 100 for l in train_losses]
        ax3.bar(epochs, improvements, color='purple', alpha=0.7)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Réduction de Loss (%)', fontsize=12)
        ax3.set_title('Amélioration par rapport à Epoch 1', fontsize=13)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Résumé textuel
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary = f"""
RÉSUMÉ DE L'APPRENTISSAGE
═══════════════════════════════════════

Epochs: {len(epochs)}

Loss initiale (train): {train_losses[0]:.4f}
Loss finale (train): {train_losses[-1]:.4f}
Réduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%

Loss initiale (val): {val_losses[0]:.4f}
Loss finale (val): {val_losses[-1]:.4f}

MSE initial (test): {avg_mses[0]:.4f}
MSE final (test): {avg_mses[-1]:.4f}

INTERPRÉTATION:
"""
    
    if train_losses[-1] < train_losses[0] * 0.3:
        summary += "✅ EXCELLENT! Le modèle a très bien appris!"
    elif train_losses[-1] < train_losses[0] * 0.5:
        summary += "✓ BON! Le modèle a bien appris."
    elif train_losses[-1] < train_losses[0] * 0.8:
        summary += "→ CORRECT. Progrès visible."
    else:
        summary += "⚠️ FAIBLE. Plus d'epochs nécessaires."
    
    ax4.text(0.05, 0.95, summary, 
             transform=ax4.transAxes,
             fontsize=11,
             verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Visualisation de l\'Apprentissage - Toy Seq2Seq', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Sauvegardé: {save_path}")
    plt.close()


def visualize_before_after(model_before, model_after, dataset, device, save_path=None):
    """Compare les prédictions avant et après entraînement."""
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    
    for i in range(4):
        input_signal, target_signal, peak_info = dataset[i]
        x = np.arange(len(input_signal))
        
        # Avant entraînement
        ax_before = axes[i, 0]
        with torch.no_grad():
            output_before = model_before(input_signal.unsqueeze(0).to(device)).squeeze(0).cpu()
        
        ax_before.plot(x, input_signal.numpy(), 'b-', alpha=0.5, label='Input', linewidth=1)
        ax_before.plot(x, target_signal.numpy(), 'g-', label='Target', linewidth=2)
        ax_before.plot(x, output_before.numpy(), 'r--', label='Prédit', linewidth=2)
        
        mse_before = ((output_before - target_signal) ** 2).mean().item()
        ax_before.set_title(f'AVANT - Exemple {i+1} (MSE: {mse_before:.2f})')
        ax_before.legend(loc='upper right', fontsize=8)
        ax_before.grid(True, alpha=0.3)
        ax_before.set_ylim(-2, 30)
        
        # Après entraînement
        ax_after = axes[i, 1]
        with torch.no_grad():
            output_after = model_after(input_signal.unsqueeze(0).to(device)).squeeze(0).cpu()
        
        ax_after.plot(x, input_signal.numpy(), 'b-', alpha=0.5, label='Input', linewidth=1)
        ax_after.plot(x, target_signal.numpy(), 'g-', label='Target', linewidth=2)
        ax_after.plot(x, output_after.numpy(), 'r--', label='Prédit', linewidth=2)
        
        mse_after = ((output_after - target_signal) ** 2).mean().item()
        ax_after.set_title(f'APRÈS - Exemple {i+1} (MSE: {mse_after:.4f})')
        ax_after.legend(loc='upper right', fontsize=8)
        ax_after.grid(True, alpha=0.3)
        ax_after.set_ylim(-2, 30)
    
    plt.suptitle('Comparaison AVANT vs APRÈS entraînement', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Sauvegardé: {save_path}")
    plt.close()


def main():
    """Visualisation complète de l'apprentissage."""
    
    print("\n" + "="*70)
    print("VISUALISATION DE L'APPRENTISSAGE - TOY SEQ2SEQ")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Device: {device}")
    
    # Dossier de sortie
    output_dir = Path(__file__).parent.parent / 'src' / 'tmp'
    output_dir.mkdir(exist_ok=True)
    
    # Dataset
    print("\n Création des datasets...")
    train_dataset = SignalDataset(num_samples=1000, seq_len=100, seed=42)
    val_dataset = SignalDataset(num_samples=100, seq_len=100, seed=123)
    test_dataset = SignalDataset(num_samples=20, seq_len=100, seed=456)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
    
    print(f"   Train: {len(train_dataset)} échantillons")
    print(f"   Val: {len(val_dataset)} échantillons")
    print(f"   Test: {len(test_dataset)} échantillons")
    
    # Modèle
    print("\n Création du modèle...")
    model = SimpleSignalTransformer(
        seq_len=100, d_model=64, num_heads=4, d_ff=256, num_layers=3, patch_size=5
    ).to(device)
    
    # Sauvegarder l'état initial pour comparaison
    model_initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Paramètres: {num_params:,}")
    
    # Entraînement
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    print("\n" + "="*70)
    print("ENTRAÎNEMENT AVEC VISUALISATION")
    print("="*70)
    
    num_epochs = 30
    epoch_data = []
    
    for epoch in range(1, num_epochs + 1):
        # Entraînement
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        # Test sur échantillons
        results = test_model_on_samples(model, test_dataset, device, num_samples=4)
        avg_mse = np.mean([r['mse'] for r in results])
        
        # Sauvegarder les données
        epoch_data.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'avg_mse': avg_mse,
            'results': results
        })
        
        # Affichage
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{num_epochs}: "
                  f"train={train_loss:.4f}, val={val_loss:.4f}, test_mse={avg_mse:.4f}")
            
            # Sauvegarder les prédictions à cette epoch
            visualize_epoch_predictions(
                results, epoch, 
                save_path=output_dir / f'predictions_epoch_{epoch:02d}.png'
            )
    
    # Visualisations finales
    print("\n Génération des visualisations finales...")
    
    # 1. Progression de l'apprentissage
    visualize_learning_progress(
        epoch_data,
        save_path=output_dir / 'learning_progress.png'
    )
    
    # 2. Comparaison avant/après
    # Créer un modèle "avant" avec les poids initiaux
    model_before = SimpleSignalTransformer(
        seq_len=100, d_model=64, num_heads=4, d_ff=256, num_layers=3, patch_size=5
    ).to(device)
    model_before.load_state_dict(model_initial_state)
    
    visualize_before_after(
        model_before, model, test_dataset, device,
        save_path=output_dir / 'before_after_comparison.png'
    )
    
    # Résumé final
    print("\n" + "="*70)
    print("RÉSUMÉ")
    print("="*70)
    
    initial_loss = epoch_data[0]['train_loss']
    final_loss = epoch_data[-1]['train_loss']
    reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"""
   Loss initiale: {initial_loss:.4f}
   Loss finale: {final_loss:.4f}
   Réduction: {reduction:.1f}%
   
   Visualisations générées dans {output_dir}:
    learning_progress.png - Courbes de loss
    before_after_comparison.png - Comparaison avant/après
    predictions_epoch_XX.png - Prédictions par epoch
    """)
    
    if reduction > 70:
        print("    EXCELLENT! Le modèle a très bien appris!")
    elif reduction > 50:
        print("   ✓ BON! Le modèle a bien appris.")
    else:
        print("   → Progrès visible, mais plus d'epochs aideraient.")
    
    print("\n" + "="*70)
    print("   Visualisation terminée!")
    print("="*70)


if __name__ == "__main__":
    main()
