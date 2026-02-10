#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPLICATION DÉTAILLÉE DE L'ENTRAÎNEMENT - EXAMPLE 9 TOY SEQ2SEQ
===============================================================

Ce fichier explique LIGNE PAR LIGNE ce qui se passe pendant l'entraînement
du modèle Transformer sur la tâche de transformation de signaux.

Exécuter avec:
    python -m docs.training_explained
"""

import sys
import io
from pathlib import Path

# Forcer l'encodage UTF-8 sur Windows pour supporter les caractères spéciaux
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from src.examples.toy_seq2seq import (
    SignalDataset,
    SimpleSignalTransformer,
    collate_fn
)


# ============================================================================
# ÉTAPE 1 : PRÉPARATION DES DONNÉES
# ============================================================================

def etape1_preparation_donnees():
    """
    Prépare les données avant l'entraînement.
    
    OBJECTIF : Créer des signaux avec pics que le modèle va apprendre à transformer.
    """
    
    print("\n" + "="*70)
    print("ÉTAPE 1 : PRÉPARATION DES DONNÉES")
    print("="*70)
    
    # 1.1 - CRÉATION DU DATASET
    # ─────────────────────────
    print("\n1.1 - Création du Dataset")
    print("-" * 40)
    
    dataset = SignalDataset(
        num_samples=100,      # 100 exemples pour la démo
        seq_len=100,          # Signal de 100 points
        num_peaks_per_type=2, # 2 pics triangles, 2 pics carrés
        seed=42               # Reproductibilité
    )
    
    print(f"   Dataset créé: {len(dataset)} échantillons")
    print(f"   Longueur des signaux: 100 points")
    
    # 1.2 - STRUCTURE D'UN ÉCHANTILLON
    # ─────────────────────────────────
    print("\n1.2 - Structure d'un échantillon")
    print("-" * 40)
    
    input_signal, output_signal, peak_info = dataset[0]
    
    print(f"""
   Chaque échantillon contient 3 éléments:
   
   1. INPUT SIGNAL (ce que le modèle reçoit):
      Shape: {input_signal.shape}
      Type: {input_signal.dtype}
      Contenu: Signal avec pics de différentes hauteurs
      
   2. OUTPUT SIGNAL (ce que le modèle doit prédire):
      Shape: {output_signal.shape}
      Type: {output_signal.dtype}
      Contenu: Signal avec pics MOYENNÉS par type
      
   3. PEAK INFO (métadonnées):
      - Positions des triangles: {peak_info['type_a_positions']}
      - Hauteurs des triangles: {[f'{h:.1f}' for h in peak_info['type_a_heights']]}
      - Positions des carrés: {peak_info['type_b_positions']}
      - Hauteurs des carrés: {[f'{h:.1f}' for h in peak_info['type_b_heights']]}
      - Moyenne triangles: {peak_info['avg_a']:.2f}
      - Moyenne carrés: {peak_info['avg_b']:.2f}
    """)
    
    # 1.3 - LA TÂCHE À APPRENDRE
    # ──────────────────────────
    print("\n1.3 - La tâche à apprendre")
    print("-" * 40)
    
    print(f"""
   TÂCHE: Transformer les pics selon leur type
   
   INPUT:
   ┌────────────────────────────────────────────────────┐
   │    ▲ h={peak_info['type_a_heights'][0]:.1f}                          │
   │   /│\\     ▲ h={peak_info['type_a_heights'][1]:.1f}                   │
   │  / │ \\   /│\\      ■ h={peak_info['type_b_heights'][0]:.1f}          │
   │ /  │  \\ / │ \\    ┌─┐    ■ h={peak_info['type_b_heights'][1]:.1f}    │
   │/   │   \\  │  \\   │ │   ┌─┐                        │
   └────────────────────────────────────────────────────┘
   
   OUTPUT (moyennes par type):
   ┌────────────────────────────────────────────────────┐
   │    ▲ h={peak_info['avg_a']:.1f}                          │
   │   /│\\     ▲ h={peak_info['avg_a']:.1f}                   │
   │  / │ \\   /│\\      ■ h={peak_info['avg_b']:.1f}          │
   │ /  │  \\ / │ \\    ┌─┐    ■ h={peak_info['avg_b']:.1f}    │
   │/   │   \\  │  \\   │ │   ┌─┐                        │
   └────────────────────────────────────────────────────┘
   
   → Les triangles ont maintenant TOUS la même hauteur (moyenne)
   → Les carrés ont maintenant TOUS la même hauteur (moyenne)
    """)
    
    # 1.4 - CRÉATION DU DATALOADER
    # ────────────────────────────
    print("\n1.4 - Création du DataLoader")
    print("-" * 40)
    
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    print(f"""
   DataLoader configuré:
   - Batch size: 16 échantillons par batch
   - Shuffle: True (mélange les données)
   - Nombre de batches: {len(dataloader)}
   
   POURQUOI UN DATALOADER ?
   → Regroupe les données en "batches" pour l'entraînement
   → Le modèle traite 16 signaux à la fois (plus efficace)
   → Le shuffle évite que le modèle mémorise l'ordre
    """)
    
    return dataset, dataloader


# ============================================================================
# ÉTAPE 2 : CRÉATION DU MODÈLE
# ============================================================================

def etape2_creation_modele():
    """
    Crée le modèle Transformer pour signaux AVEC PATCHING.
    """
    
    print("\n" + "="*70)
    print("ÉTAPE 2 : CRÉATION DU MODÈLE (AVEC PATCHING)")
    print("="*70)
    
    model = SimpleSignalTransformer(
        seq_len=100,      # Longueur des signaux
        d_model=64,       # Dimension des embeddings
        num_heads=4,      # Nombre de têtes d'attention
        d_ff=256,         # Dimension feed-forward
        num_layers=3,     # Nombre de couches
        dropout=0.1,      # Régularisation
        patch_size=5      # PATCHING: 5 points par token
    )
    
    print(f"""
   ARCHITECTURE DU MODÈLE AVEC PATCHING:
   ─────────────────────────────────────
   
   INPUT: Signal 1D [batch, 100]
          │
          ▼
   ┌──────────────────────────┐
   │   DÉCOUPAGE EN PATCHES   │  100 points → 20 patches de 5
   └──────────────────────────┘
          │
          ▼
   ┌──────────────────────────┐
   │   INPUT PROJECTION       │  5 → 64 dimensions
   │   (Linear + Position)    │
   └──────────────────────────┘
          │
          ▼
   ┌──────────────────────────┐
   │   ENCODER (3 couches)    │
   │                          │
   │   Pour chaque couche:    │
   │   ├─ Self-Attention      │  Attention entre PATCHES
   │   ├─ Add & Norm          │
   │   ├─ Feed-Forward        │  MLP à 256 dims
   │   └─ Add & Norm          │
   └──────────────────────────┘
          │
          ▼
   ┌──────────────────────────┐
   │   OUTPUT PROJECTION      │  64 → 5 dimensions
   │   (Linear)               │
   └──────────────────────────┘
          │
          ▼
   ┌──────────────────────────┐
   │   RECONSTRUCTION         │  20 patches → 100 points
   └──────────────────────────┘
          │
          ▼
   OUTPUT: Signal 1D [batch, 100]
   
   
   AVANTAGE DU PATCHING:
   ─────────────────────
   - Ancien: 100 tokens → Attention O(100²) = 10,000 opérations
   - Nouveau: 20 tokens → Attention O(20²) = 400 opérations
   → 25x moins de calcul! Et meilleur contexte local.
   
   
   PARAMÈTRES DU MODÈLE:
   ─────────────────────
    """)
    
    # Compter les paramètres
    total_params = 0
    for name, param in model.named_parameters():
        num = param.numel()
        total_params += num
        print(f"   {name}: {num:,} paramètres")
    
    print(f"\n   TOTAL: {total_params:,} paramètres à apprendre!")
    
    print(f"""
   
   POURQUOI CES CHOIX ?
   ────────────────────
   - patch_size=5: Regroupe 5 points (contexte local)
   - d_model=64: Petit modèle suffisant pour cette tâche
   - num_heads=4: 4 "points de vue" différents
   - num_layers=3: Assez profond pour apprendre les patterns
   - Encoder-only: Pas besoin de decoder (input/output même taille)
    """)
    
    return model


# ============================================================================
# ÉTAPE 3 : UNE ITÉRATION D'ENTRAÎNEMENT
# ============================================================================

def etape3_une_iteration(model, dataloader, device):
    """
    Explique EN DÉTAIL ce qui se passe pour UN SEUL BATCH.
    """
    
    print("\n" + "="*70)
    print("ÉTAPE 3 : ANATOMIE D'UNE ITÉRATION")
    print("="*70)
    
    # Préparer
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Récupérer un batch
    inputs, targets, _ = next(iter(dataloader))
    inputs, targets = inputs.to(device), targets.to(device)
    
    print(f"""
   ┌────────────────────────────────────────────────────────────────────┐
   │  INSTRUCTION 1: Récupérer un batch                                 │
   └────────────────────────────────────────────────────────────────────┘
   
   inputs shape: {inputs.shape}  → 16 signaux de 100 points
   targets shape: {targets.shape} → 16 signaux cibles
   
   
   ┌────────────────────────────────────────────────────────────────────┐
   │  INSTRUCTION 2: Reset des gradients                                │
   └────────────────────────────────────────────────────────────────────┘
   
   optimizer.zero_grad()
   
   → Remet tous les gradients à zéro
   → OBLIGATOIRE car PyTorch accumule les gradients par défaut
    """)
    
    optimizer.zero_grad()
    
    print(f"""
   ┌────────────────────────────────────────────────────────────────────┐
   │  INSTRUCTION 3: FORWARD PASS (Prédiction)                          │
   └────────────────────────────────────────────────────────────────────┘
   
   output = model(inputs)
   
   Que se passe-t-il ? (AVEC PATCHING)
   """)
    
    # Forward détaillé AVEC PATCHING
    batch_size = inputs.shape[0]
    
    print("   1. Découpage en patches:")
    print(f"      inputs: {inputs.shape}")
    x_patches = inputs.view(batch_size, model.num_patches, model.patch_size)
    print(f"      → patches: {x_patches.shape} ({model.num_patches} patches de {model.patch_size} points)")
    
    print("\n   2. Input projection:")
    x_proj = model.input_proj(x_patches)  # [batch, num_patches, d_model]
    print(f"      après projection: {x_proj.shape}")
    
    x_pos = x_proj + model.pos_embedding
    print(f"      + positional encoding: {x_pos.shape}")
    
    print("\n   3. Encoder (Self-Attention entre PATCHES):")
    print("      Chaque patch peut 'regarder' tous les autres patches")
    print("""
      ╔═══════════════════════════════════════════════════════════════════╗
      ║  COMMENT L'ATTENTION FONCTIONNE ENTRE PATCHES ?                   ║
      ╠═══════════════════════════════════════════════════════════════════╣
      ║                                                                   ║
      ║  Signal: [___▲___][___▲___][___■___][_______][___■___]           ║
      ║           patch1    patch2   patch3   patch4   patch5             ║
      ║                                                                   ║
      ║  Self-Attention: Chaque patch calcule des SCORES avec tous:       ║
      ║                                                                   ║
      ║  patch1 (▲) regarde:                                              ║
      ║    - patch1: score=0.1  (lui-même)                                ║
      ║    - patch2: score=0.7  (▲ similaire! haute attention)            ║
      ║    - patch3: score=0.05 (■ différent, basse attention)            ║
      ║    - patch4: score=0.1  (plat, peu intéressant)                   ║
      ║    - patch5: score=0.05 (■ différent)                             ║
      ║                                                                   ║
      ║  Résultat: patch1 "écoute" surtout patch2 (même type ▲)           ║
      ║  → Il peut calculer la moyenne des triangles!                     ║
      ║                                                                   ║
      ║  MAGIE: Le modèle APPREND ces scores pendant l'entraînement       ║
      ║  Les poids Q, K, V s'ajustent pour que:                           ║
      ║    - Triangles aient des représentations similaires               ║
      ║    - Carrés aient des représentations similaires                  ║
      ║    - Triangles et carrés soient différents                        ║
      ╚═══════════════════════════════════════════════════════════════════╝
      """)
    
    encoded = model.encoder(x_pos)
    print(f"      après encoder: {encoded.shape}")
    
    print("\n   4. Output projection + reconstruction:")
    x_out = model.output_proj(encoded)  # [batch, num_patches, patch_size]
    print(f"      après projection: {x_out.shape}")
    output = x_out.view(batch_size, model.seq_len)
    print(f"      signal reconstitué: {output.shape}")
    
    print(f"""
   
   ┌────────────────────────────────────────────────────────────────────┐
   │  INSTRUCTION 4: Calcul de la LOSS                                  │
   └────────────────────────────────────────────────────────────────────┘
   
   loss = criterion(output, targets)
   
   Critère: MSE (Mean Squared Error)
   → Mesure la différence au carré entre prédiction et cible
   
   Formule: loss = moyenne((output - targets)²)
    """)
    
    loss = criterion(output, targets)
    print(f"   Loss calculée: {loss.item():.4f}")
    
    print(f"""
   
   ┌────────────────────────────────────────────────────────────────────┐
   │  INSTRUCTION 5: BACKWARD PASS (Calcul des gradients)               │
   └────────────────────────────────────────────────────────────────────┘
   
   loss.backward()
   
   → Calcule ∂loss/∂poids pour CHAQUE paramètre
   → Utilise la rétropropagation (chain rule)
   → Les gradients indiquent comment modifier les poids
    """)
    
    loss.backward()
    
    # Montrer quelques gradients
    print("\n   Exemples de gradients calculés:")
    for name, param in list(model.named_parameters())[:3]:
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"      {name}: gradient norm = {grad_norm:.6f}")
    
    print(f"""
   
   ┌────────────────────────────────────────────────────────────────────┐
   │  INSTRUCTION 6: UPDATE (Mise à jour des poids)                     │
   └────────────────────────────────────────────────────────────────────┘
   
   optimizer.step()
   
   Formule (Adam simplifié):
   poids_nouveau = poids_ancien - learning_rate × gradient
   
   Exemple concret:
    """)
    
    # Montrer un exemple de mise à jour
    # Trouver un paramètre avec au moins 2 dimensions et shape[0] > 1
    param_exemple = None
    param_nom = None
    for name, p in model.named_parameters():
        if p.dim() >= 2 and p.shape[0] > 1 and p.shape[1] > 1:
            param_exemple = p
            param_nom = name
            break
    
    if param_exemple is not None:
        poids_avant = param_exemple.data[0, 0].item()
        optimizer.step()
        poids_apres = param_exemple.data[0, 0].item()
        
        print(f"   Paramètre: {param_nom}")
        print(f"   Un poids avant: {poids_avant:.6f}")
        print(f"   Un poids après: {poids_apres:.6f}")
        print(f"   Changement: {poids_apres - poids_avant:.6f}")
    else:
        optimizer.step()
        print("   (Poids mis à jour)")
    
    print(f"""
   
   ┌────────────────────────────────────────────────────────────────────┐
   │  RÉSUMÉ D'UNE ITÉRATION                                            │
   └────────────────────────────────────────────────────────────────────┘
   
   1. Récupérer un batch (16 signaux)
   2. Reset gradients
   3. Forward: inputs → modèle → prédictions
   4. Loss: comparer prédictions vs cibles
   5. Backward: calculer les gradients
   6. Update: ajuster les poids
   
   → Répéter pour tous les batches = 1 EPOCH
   → Répéter pour plusieurs epochs = ENTRAÎNEMENT COMPLET
    """)
    
    return loss.item()


# ============================================================================
# ÉTAPE 4 : ENTRAÎNEMENT COMPLET
# ============================================================================

def etape4_entrainement_complet():
    """
    Montre l'entraînement complet sur quelques epochs.
    """
    
    print("\n" + "="*70)
    print("ÉTAPE 4 : ENTRAÎNEMENT COMPLET")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    train_dataset = SignalDataset(num_samples=500, seq_len=100, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    # Modèle
    model = SimpleSignalTransformer(seq_len=100, d_model=64, num_heads=4, num_layers=2, patch_size=5).to(device)
    
    # Optimizer et Loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"""
   Configuration:
   - Device: {device}
   - Échantillons: 500
   - Batch size: 32
   - Batches par epoch: {len(train_loader)}
   - Epochs: 5
    """)
    
    print("\n   DÉBUT DE L'ENTRAÎNEMENT:")
    print("   " + "-" * 50)
    
    for epoch in range(5):
        model.train()
        total_loss = 0
        
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Barre de progression visuelle
        bar_length = int((5 - avg_loss) * 10)
        bar = "█" * max(0, bar_length) + "░" * (50 - max(0, bar_length))
        
        print(f"   Epoch {epoch+1}: Loss = {avg_loss:.4f} [{bar}]")
    
    print(f"""
   
   INTERPRÉTATION:
   ───────────────
   - La loss diminue → Le modèle apprend!
   - Il comprend de mieux en mieux la tâche
   - Il apprend à identifier les pics et calculer les moyennes
    """)
    
    return model


# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================

def resume_final():
    """
    Résumé de tout le processus.
    """
    
    print("\n" + "#"*70)
    print("# RÉSUMÉ : QU'EST-CE QUE L'ENTRAÎNEMENT ?")
    print("#"*70)
    
    print(f"""
   
   L'ENTRAÎNEMENT EN UNE PHRASE:
   ════════════════════════════
   Ajuster progressivement les poids du modèle pour qu'il transforme
   correctement les signaux d'entrée en signaux de sortie attendus.
   
   
   LE PROCESSUS POUR EXAMPLE 9 (TOY SEQ2SEQ):
   ══════════════════════════════════════════
   
   DONNÉES:
   ┌─────────────────┐      ┌─────────────────┐
   │ Signal INPUT    │ ───► │ Signal OUTPUT   │
   │ (pics variés)   │      │ (pics moyennés) │
   └─────────────────┘      └─────────────────┘
   
   MODÈLE:
   ┌─────────────────────────────────────────────────────┐
   │                 TRANSFORMER                         │
   │                                                     │
   │   Input ──► Projection ──► Encoder ──► Output      │
   │             (1→64 dim)    (attention)  (64→1 dim)  │
   └─────────────────────────────────────────────────────┘
   
   BOUCLE D'ENTRAÎNEMENT:
   
   POUR chaque epoch (passage sur toutes les données):
   │
   ├── POUR chaque batch (groupe de 32 signaux):
   │   │
   │   ├── 1. Forward: signal → modèle → prédiction
   │   │
   │   ├── 2. Loss: comparer prédiction vs cible
   │   │       loss = MSE(prédiction, cible)
   │   │
   │   ├── 3. Backward: calculer gradients
   │   │       ∂loss/∂poids pour chaque paramètre
   │   │
   │   └── 4. Update: ajuster les poids
   │           poids = poids - lr × gradient
   │
   └── → Le modèle s'améliore à chaque batch!
   
   
   CE QUE LE MODÈLE APPREND:
   ═════════════════════════
   
   1. IDENTIFIER les pics dans le signal
      → L'attention "regarde" les positions avec des valeurs élevées
   
   2. CLASSER les pics par type (triangle vs carré)
      → Apprend les patterns de forme
   
   3. CALCULER la moyenne par type
      → Les poids apprennent cette opération mathématique
   
   4. GÉNÉRER le signal de sortie
      → Remplace chaque pic par la moyenne de son type
   
   
   VOCABULAIRE:
   ════════════
   
    Batch     : Groupe de signaux traités ensemble (32)
    Epoch     : 1 passage sur toutes les données
    Loss      : Mesure de l'erreur (MSE)
    Gradient  : Direction pour améliorer les poids
    Optimizer : Algorithme qui met à jour les poids (Adam)
   
   
   RÉSULTAT:
   ═════════
   Après entraînement, le modèle peut:
   - Recevoir un nouveau signal avec pics
   - Identifier automatiquement les types de pics
   - Produire le signal transformé (pics moyennés)
   
   Sans jamais avoir été explicitement programmé pour le faire!
   C'est la puissance de l'apprentissage automatique!
    """)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Exécute le guide complet."""
    
    print("\n" + "="*70)
    print("   GUIDE COMPLET : COMPRENDRE L'ENTRAÎNEMENT")
    print("   Example 9 - Toy Seq2Seq Signal Processing")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Étape 1
    dataset, dataloader = etape1_preparation_donnees()
    input("\n[Appuyez sur Entrée pour continuer...]")
    
    # Étape 2
    model = etape2_creation_modele()
    input("\n[Appuyez sur Entrée pour continuer...]")
    
    # Étape 3
    etape3_une_iteration(model, dataloader, device)
    input("\n[Appuyez sur Entrée pour continuer...]")
    
    # Étape 4
    etape4_entrainement_complet()
    input("\n[Appuyez sur Entrée pour continuer...]")
    
    # Résumé
    resume_final()
    
    print("\n" + "="*70)
    print("*** Vous comprenez maintenant l'entrainement en profondeur! ***")
    print("="*70)


if __name__ == "__main__":
    main()
