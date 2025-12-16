"""
EXPLICATION DÉTAILLÉE DE L'ENTRAÎNEMENT - EXEMPLE 4
===================================================

Ce fichier explique LIGNE PAR LIGNE ce qui se passe pendant l'entraînement.
"""

# IMPORTS NÉCESSAIRES
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Importer les classes de l'exemple 4
from src.examples.example4_translation import (
    NumberToWordVocab,
    NumberToWordDataset,
    collate_fn,
    Transformer
)

# ============================================================================
# ÉTAPE 1 : PRÉPARATION DES DONNÉES
# ============================================================================

def etape1_preparation_donnees():
    """
    Prépare les données avant l'entraînement.
    
    OBJECTIF : Transformer du texte en nombres que le modèle peut traiter.
    """
    
    # 1.1 - CRÉATION DU VOCABULAIRE
    # ─────────────────────────────
    vocab = NumberToWordVocab()
    
    # QUE FAIT LE VOCABULAIRE ?
    # Il crée une table de correspondance : token ↔ indice
    
    # Source (chiffres):
    # '<pad>' → 0
    # '<sos>' → 1
    # '<eos>' → 2
    # '0' → 3
    # '1' → 4
    # '2' → 5
    # ...
    
    # Target (mots):
    # '<pad>' → 0
    # '<sos>' → 1
    # '<eos>' → 2
    # 'zero' → 3
    # 'one' → 4
    # 'two' → 5
    # ...
    
    print("Vocabulaires créés!")
    print(f"Source vocab size: {vocab.src_vocab_size}")  # 13 tokens
    print(f"Target vocab size: {vocab.tgt_vocab_size}")  # 13 tokens
    
    
    # 1.2 - GÉNÉRATION DES DONNÉES D'ENTRAÎNEMENT
    # ───────────────────────────────────────────
    train_dataset = NumberToWordDataset(vocab, num_samples=2000, seed=42)
    
    # QUE FAIT LE DATASET ?
    # Il génère 2000 paires (source, target) :
    
    # Exemple 1:
    # Texte:    "42" → ['four', 'two']
    # Encodé:   [1, 7, 5, 2] → [1, 7, 5, 2]
    #           ↑  ↑  ↑  ↑     ↑  ↑  ↑  ↑
    #          sos 4  2 eos   sos four two eos
    
    # Exemple 2:
    # Texte:    "8" → ['eight']
    # Encodé:   [1, 11, 2] → [1, 11, 2]
    #           ↑  ↑   ↑     ↑  ↑    ↑
    #          sos 8  eos   sos eight eos
    
    print(f"\n{len(train_dataset)} exemples générés")
    
    # Regardons un exemple:
    src, tgt = train_dataset[0]
    print(f"Exemple encodé: {src} → {tgt}")
    print(f"Exemple décodé: '{vocab.decode_source(src)}' → {vocab.decode_target(tgt)}")
    
    
    # 1.3 - CRÉATION DU DATALOADER
    # ────────────────────────────
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,  # Traite 32 exemples à la fois
        shuffle=True,   # Mélange les données (important!)
        collate_fn=lambda b: collate_fn(b, vocab.pad_idx)
    )
    
    # QUE FAIT LE DATALOADER ?
    # Il groupe les données en "batches" de 32 exemples
    # et ajoute du padding pour que tout ait la même longueur
    
    # Sans padding (longueurs différentes):
    # Exemple 1: [1, 4, 2]        (longueur 3)
    # Exemple 2: [1, 4, 5, 6, 2]  (longueur 5)
    # ❌ Impossible de mettre dans un tenseur!
    
    # Avec padding (même longueur):
    # Exemple 1: [1, 4, 2, 0, 0]  (longueur 5, +2 pads)
    # Exemple 2: [1, 4, 5, 6, 2]  (longueur 5)
    # ✅ On peut créer un tenseur 2D!
    
    print(f"\nDataLoader créé: {len(train_loader)} batches de 32 exemples")


# ============================================================================
# ÉTAPE 2 : CRÉATION DU MODÈLE
# ============================================================================

def etape2_creation_modele():
    """
    Crée le modèle Transformer.
    
    OBJECTIF : Initialiser un réseau de neurones qui va apprendre.
    """
    
    model = Transformer(
        src_vocab_size=13,      # Taille vocabulaire source
        tgt_vocab_size=13,      # Taille vocabulaire cible
        d_model=64,             # Dimension des embeddings
        num_heads=4,            # Nombre de têtes d'attention
        d_ff=256,               # Dimension feed-forward
        num_encoder_layers=2,   # 2 couches encoder
        num_decoder_layers=2,   # 2 couches decoder
        dropout=0.1             # Dropout 10%
    )
    
    # STRUCTURE DU MODÈLE :
    # ────────────────────
    # 
    # INPUT (indices) → EMBEDDINGS (vecteurs)
    #                       ↓
    #                   ENCODER (2 couches)
    #                   - Self-attention
    #                   - Feed-forward
    #                       ↓
    #                   MEMORY (représentation source)
    #                       ↓
    #                   DECODER (2 couches)
    #                   - Self-attention
    #                   - Cross-attention (regarde la source)
    #                   - Feed-forward
    #                       ↓
    #                   OUTPUT PROJECTION
    #                       ↓
    #                   PROBABILITÉS pour chaque mot
    
    # NOMBRE DE PARAMÈTRES :
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Modèle créé avec {num_params:,} paramètres")
    
    # Ces paramètres sont les "poids" que le modèle va ajuster
    # pendant l'entraînement pour apprendre la tâche.
    
    # Au début : Poids ALÉATOIRES → Prédictions nulles
    # Après entraînement : Poids OPTIMISÉS → Bonnes prédictions


# ============================================================================
# ÉTAPE 3 : UNE ITÉRATION D'ENTRAÎNEMENT (1 BATCH)
# ============================================================================

def etape3_une_iteration_detaillee():
    """
    Explique EN DÉTAIL ce qui se passe pour UN SEUL BATCH.
    
    C'est LE CŒUR de l'entraînement !
    """
    
    print("\n" + "="*70)
    print("ANATOMIE D'UNE ITÉRATION D'ENTRAÎNEMENT")
    print("="*70)
    
    # ────────────────────────────────────────────────────────────────────
    # INSTRUCTION 1 : Récupérer un batch de données
    # ────────────────────────────────────────────────────────────────────
    
    # for src, tgt in dataloader:
    
    # QUE SE PASSE-T-IL ?
    # Le DataLoader donne 32 paires (source, target)
    
    # src shape: [32, max_src_len]  (32 séquences sources)
    # tgt shape: [32, max_tgt_len]  (32 séquences cibles)
    
    # EXEMPLE DE BATCH (simplifié à 3 exemples):
    src_batch_exemple = [
        [1, 7, 5, 2, 0],     # "42" + padding
        [1, 4, 5, 6, 2],     # "123"
        [1, 11, 2, 0, 0]     # "8" + padding
    ]
    
    tgt_batch_exemple = [
        [1, 7, 5, 2, 0],     # "four two" + padding
        [1, 4, 5, 6, 2],     # "one two three"
        [1, 11, 2, 0, 0]     # "eight" + padding
    ]
    
    print("\n1. Batch récupéré:")
    print(f"   Source shape: [batch=32, seq_len=variable]")
    print(f"   Target shape: [batch=32, seq_len=variable]")
    
    
    # ────────────────────────────────────────────────────────────────────
    # INSTRUCTION 2 : Transférer sur GPU/CPU
    # ────────────────────────────────────────────────────────────────────
    
    # src, tgt = src.to(device), tgt.to(device)
    
    # QUE SE PASSE-T-IL ?
    # Les tenseurs sont copiés sur le GPU (si disponible)
    # pour des calculs plus rapides
    
    print("\n2. Données transférées sur device (GPU ou CPU)")
    
    
    # ────────────────────────────────────────────────────────────────────
    # INSTRUCTION 3 : Préparer input et output du decoder
    # ────────────────────────────────────────────────────────────────────
    
    # tgt_input = tgt[:, :-1]   # Sans le dernier token
    # tgt_output = tgt[:, 1:]   # Sans le premier token
    
    # POURQUOI CETTE SÉPARATION ?
    # C'est le principe du TEACHER FORCING !
    
    # Target original:  [<sos>, one, two, three, <eos>]
    #                     ↓
    # Decoder INPUT:    [<sos>, one, two, three]       (enlève <eos>)
    # Decoder OUTPUT:   [one, two, three, <eos>]       (enlève <sos>)
    
    # Le modèle apprend à prédire le token SUIVANT :
    # - Reçoit <sos> → doit prédire "one"
    # - Reçoit <sos> one → doit prédire "two"
    # - Reçoit <sos> one two → doit prédire "three"
    # - Reçoit <sos> one two three → doit prédire <eos>
    
    print("\n3. Target séparé en input/output (Teacher Forcing):")
    print("   Target:       [<sos>, one, two, three, <eos>]")
    print("   Decoder IN:   [<sos>, one, two, three]")
    print("   Decoder OUT:  [one, two, three, <eos>]")
    
    
    # ────────────────────────────────────────────────────────────────────
    # INSTRUCTION 4 : Créer le masque de padding
    # ────────────────────────────────────────────────────────────────────
    
    # src_padding_mask = create_padding_mask(src, pad_idx)
    
    # QUE FAIT CE MASQUE ?
    # Il indique au modèle : "Ignore les tokens <pad>"
    
    # Source:  [1, 7, 5, 2, 0, 0]  (0 = pad)
    # Masque:  [F, F, F, F, T, T]  (T = True = ignore)
    
    # Pourquoi ? Le padding n'a aucun sens sémantique,
    # il ne sert qu'à aligner les longueurs.
    
    print("\n4. Masque de padding créé:")
    print("   Source: [<sos>, 4, 2, <eos>, <pad>, <pad>]")
    print("   Masque: [False, F, F, False, TRUE, TRUE]")
    print("   → Le modèle ignorera les <pad> dans l'attention")
    
    
    # ────────────────────────────────────────────────────────────────────
    # INSTRUCTION 5 : Reset des gradients
    # ────────────────────────────────────────────────────────────────────
    
    # optimizer.zero_grad()
    
    # QUE SE PASSE-T-IL ?
    # PyTorch ACCUMULE les gradients par défaut.
    # On doit les remettre à zéro avant chaque itération.
    
    # Gradient = Dérivée de la loss par rapport aux poids
    # → Indique dans quelle direction modifier les poids
    
    print("\n5. Gradients remis à zéro")
    
    
    # ────────────────────────────────────────────────────────────────────
    # INSTRUCTION 6 : FORWARD PASS (Prédiction)
    # ────────────────────────────────────────────────────────────────────
    
    # output = model(src, tgt_input, src_mask=src_padding_mask)
    
    # C'EST ICI QUE LA MAGIE OPÈRE !
    
    # QUE FAIT LE MODÈLE ?
    # 
    # 1. ENCODER traite la source "42" :
    #    [1, 7, 5, 2] → Embeddings → Self-Attention → Feed-Forward
    #    → Produit MEMORY (représentation de "42")
    # 
    # 2. DECODER génère la prédiction :
    #    Reçoit [<sos>, four, two] (decoder input)
    #    + MEMORY (de l'encoder)
    #    → Self-Attention (regarde ce qu'il a déjà généré)
    #    → Cross-Attention (regarde la source via MEMORY)
    #    → Feed-Forward
    #    → Output Projection
    #    → PROBABILITÉS pour chaque mot du vocabulaire
    # 
    # Output shape: [batch=32, seq_len, vocab_size=13]
    # 
    # Pour chaque position, on a 13 probabilités :
    # Position 0: [0.01, 0.05, 0.02, 0.01, 0.85, ...]  ← prédit "four" (85%)
    # Position 1: [0.02, 0.03, 0.01, 0.01, 0.02, 0.88, ...]  ← prédit "two" (88%)
    # Position 2: [0.01, 0.02, 0.92, ...]  ← prédit <eos> (92%)
    
    print("\n6. FORWARD PASS:")
    print("   Source → Encoder → MEMORY")
    print("   Target + MEMORY → Decoder → Prédictions")
    print("   Output shape: [batch=32, seq_len, vocab_size=13]")
    
    
    # ────────────────────────────────────────────────────────────────────
    # INSTRUCTION 7 : Reshape pour la loss
    # ────────────────────────────────────────────────────────────────────
    
    # output = output.contiguous().view(-1, output.size(-1))
    # tgt_output = tgt_output.contiguous().view(-1)
    
    # POURQUOI CE RESHAPE ?
    # CrossEntropyLoss attend des tenseurs 2D :
    # - Prédictions: [N, vocab_size]
    # - Cibles: [N]
    
    # AVANT reshape:
    # output: [batch=32, seq_len=4, vocab=13]
    # tgt_output: [batch=32, seq_len=4]
    
    # APRÈS reshape (flatten):
    # output: [32*4=128, 13]  → 128 prédictions, 13 classes chacune
    # tgt_output: [128]       → 128 vraies valeurs
    
    print("\n7. Reshape des tenseurs:")
    print("   Avant: [32, 4, 13] et [32, 4]")
    print("   Après: [128, 13] et [128]")
    print("   → Aplatit batch et séquence en une seule dimension")
    
    
    # ────────────────────────────────────────────────────────────────────
    # INSTRUCTION 8 : Calculer la LOSS (Erreur)
    # ────────────────────────────────────────────────────────────────────
    
    # loss = criterion(output, tgt_output)
    
    # QU'EST-CE QUE LA LOSS ?
    # C'est une MESURE D'ERREUR : distance entre prédiction et vérité
    
    # criterion = CrossEntropyLoss(ignore_index=pad_idx)
    
    # CrossEntropyLoss fait 2 choses :
    # 1. Softmax : Convertit scores en probabilités
    # 2. Negative Log-Likelihood : Pénalise les mauvaises prédictions
    
    # EXEMPLE :
    # Vérité: token 4 ("four")
    # Prédiction: [0.01, 0.05, 0.02, 0.01, 0.85, 0.02, ...]
    #                                      ↑
    #                                   position 4
    # Loss = -log(0.85) = 0.16  (Faible loss = Bonne prédiction!)
    
    # Si la prédiction était mauvaise :
    # Prédiction: [0.01, 0.05, 0.02, 0.80, 0.05, ...]
    #                               ↑ prédit position 3 au lieu de 4
    # Loss = -log(0.05) = 3.0  (Haute loss = Mauvaise prédiction!)
    
    # ignore_index=pad_idx : N'inclut PAS les tokens <pad> dans le calcul
    
    print("\n8. Calcul de la LOSS (CrossEntropyLoss):")
    print("   Compare prédictions vs vérités")
    print("   Ignore les tokens <pad>")
    print("   Loss basse = Bonnes prédictions")
    print("   Loss haute = Mauvaises prédictions")
    
    
    # ────────────────────────────────────────────────────────────────────
    # INSTRUCTION 9 : BACKWARD PASS (Calcul des gradients)
    # ────────────────────────────────────────────────────────────────────
    
    # loss.backward()
    
    # C'EST LA MAGIE DE L'APPRENTISSAGE !
    
    # QUE FAIT backward() ?
    # Il calcule les GRADIENTS par rétropropagation (backpropagation)
    
    # Gradient = ∂Loss/∂Poids = "Comment la loss change si je modifie ce poids"
    
    # Pour CHAQUE paramètre du modèle (tous les poids) :
    # - Si gradient > 0 : Augmenter ce poids AUGMENTE la loss → Il faut le DIMINUER
    # - Si gradient < 0 : Augmenter ce poids DIMINUE la loss → Il faut l'AUGMENTER
    
    # RÉTROPROPAGATION :
    # Loss → Output Layer → Hidden Layers → ... → Input Layer
    # On propage l'erreur en arrière pour calculer le gradient de chaque poids
    
    print("\n9. BACKWARD PASS (Rétropropagation):")
    print("   Calcule les gradients pour TOUS les paramètres")
    print("   Gradient = Direction pour diminuer la loss")
    
    
    # ────────────────────────────────────────────────────────────────────
    # INSTRUCTION 10 : Gradient Clipping
    # ────────────────────────────────────────────────────────────────────
    
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # POURQUOI CLIPPER LES GRADIENTS ?
    # Problème : Parfois les gradients deviennent ÉNORMES (exploding gradients)
    # → Mise à jour trop grande → Le modèle diverge
    
    # Solution : Limiter la norme des gradients à 1.0
    
    # Si ||gradient|| > 1.0 :
    #     gradient = gradient / ||gradient||  (normalise à 1.0)
    
    print("\n10. Gradient Clipping:")
    print("    Limite la norme des gradients à 1.0")
    print("    Évite les mises à jour trop grandes")
    
    
    # ────────────────────────────────────────────────────────────────────
    # INSTRUCTION 11 : UPDATE (Mise à jour des poids)
    # ────────────────────────────────────────────────────────────────────
    
    # optimizer.step()
    
    # C'EST ICI QU'ON APPREND VRAIMENT !
    
    # QUE FAIT step() ?
    # Il met à jour TOUS les poids du modèle selon les gradients
    
    # optimizer = Adam(model.parameters(), lr=0.001)
    
    # ALGORITHME ADAM (simplifi é) :
    # Pour chaque poids w :
    #     w_nouveau = w_ancien - learning_rate * gradient
    
    # learning_rate (lr) = Taille du pas
    # - lr trop grand : Le modèle diverge (saute partout)
    # - lr trop petit : Apprentissage très lent
    # - lr=0.001 : Bon compromis pour commencer
    
    # EXEMPLE :
    # Poids avant: w = 0.5
    # Gradient: ∂Loss/∂w = 2.0  (augmenter w augmente la loss)
    # Update: w = 0.5 - 0.001 * 2.0 = 0.498  (on DIMINUE w)
    
    # Après cette mise à jour :
    # Les poids sont LÉGÈREMENT meilleurs
    # Le modèle fait de LÉGÈREMENT meilleures prédictions
    
    print("\n11. OPTIMIZER STEP:")
    print("    Met à jour TOUS les poids du modèle")
    print("    Formule: poids_nouveau = poids_ancien - lr * gradient")
    print("    → Le modèle s'améliore un peu!")
    
    
    # ────────────────────────────────────────────────────────────────────
    # RÉSUMÉ D'UNE ITÉRATION
    # ────────────────────────────────────────────────────────────────────
    
    print("\n" + "="*70)
    print("RÉSUMÉ : Une itération = Un petit pas d'apprentissage")
    print("="*70)
    print("""
    1. Données récupérées (1 batch = 32 exemples)
    2. Forward pass → Prédictions
    3. Calcul de la loss → Mesure d'erreur
    4. Backward pass → Calcul des gradients
    5. Optimizer step → Mise à jour des poids
    
    → Le modèle est maintenant LÉGÈREMENT meilleur!
    
    Répéter 63 fois (63 batches) = 1 EPOCH
    Répéter 20 epochs = Le modèle devient expert!
    """)


# ============================================================================
# ÉTAPE 4 : UNE EPOCH COMPLÈTE
# ============================================================================

def etape4_une_epoch_complete():
    """
    Une epoch = Passage complet sur TOUTES les données.
    """
    
    print("\n" + "="*70)
    print("UNE EPOCH COMPLÈTE")
    print("="*70)
    
    # Une epoch, c'est simplement :
    # FOR EACH batch in dataloader:
    #     [faire une itération comme expliqué ci-dessus]
    
    # Avec nos données :
    # - 2000 exemples d'entraînement
    # - Batch size = 32
    # - Nombre de batches = 2000 / 32 = 63 batches (arrondi)
    
    # 1 EPOCH = 63 itérations
    
    print("""
    Dataset: 2000 exemples
    Batch size: 32
    Nombre de batches: 63
    
    1 EPOCH = Boucle sur 63 batches:
        Batch 1: exemples 0-31 → Itération 1
        Batch 2: exemples 32-63 → Itération 2
        Batch 3: exemples 64-95 → Itération 3
        ...
        Batch 63: exemples 1984-1999 → Itération 63
    
    Après 1 epoch:
    - Le modèle a vu TOUS les exemples UNE FOIS
    - Il a fait 63 mises à jour de poids
    - Il est meilleur qu'avant!
    
    Loss moyenne de l'epoch:
        total_loss / 63 = loss moyenne par batch
    """)


# ============================================================================
# ÉTAPE 5 : ENTRAÎNEMENT COMPLET (20 EPOCHS)
# ============================================================================

def etape5_entrainement_complet():
    """
    L'entraînement complet : 20 epochs.
    """
    
    print("\n" + "="*70)
    print("ENTRAÎNEMENT COMPLET - 20 EPOCHS")
    print("="*70)
    
    print("""
    num_epochs = 20
    
    FOR epoch in range(20):
        # ENTRAÎNEMENT
        FOR each batch (63 batches):
            - Forward pass
            - Calcul loss
            - Backward pass
            - Update poids
        
        → train_loss = loss moyenne de l'epoch
        
        # VALIDATION
        FOR each batch in val_loader:
            - Forward pass (SANS backward!)
            - Calcul loss
        
        → val_loss = loss moyenne sur validation
    
    ÉVOLUTION :
    ───────────
    Epoch 1:  train_loss = 1.5, val_loss = 0.7
              → Le modèle devine au hasard
    
    Epoch 5:  train_loss = 0.3, val_loss = 0.05
              → Le modèle commence à bien faire!
    
    Epoch 10: train_loss = 0.1, val_loss = 0.03
              → Le modèle est bon!
    
    Epoch 20: train_loss = 0.05, val_loss = 0.02
              → Le modèle est quasi parfait!
    
    NOMBRE TOTAL DE MISES À JOUR :
    ────────────────────────────────
    20 epochs × 63 batches = 1260 mises à jour de poids
    
    Le modèle a appris en faisant 1260 petits ajustements!
    """)


# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================

def resume_final():
    """
    Résumé de tout le processus d'entraînement.
    """
    
    print("\n" + "#"*70)
    print("# RÉSUMÉ COMPLET : QU'EST-CE QUE L'ENTRAÎNEMENT ?")
    print("#"*70)
    
    print("""
    L'ENTRAÎNEMENT EN UNE PHRASE :
    ══════════════════════════════
    Ajuster progressivement les poids du modèle pour qu'il fasse
    de meilleures prédictions sur les données.
    
    
    LE PROCESSUS COMPLET :
    ═════════════════════
    
    1. INITIALISATION
       ├─ Créer le modèle (poids aléatoires)
       ├─ Préparer les données (tokenization, batches)
       └─ Choisir optimizer (Adam) et loss (CrossEntropy)
    
    2. BOUCLE D'ENTRAÎNEMENT (20 epochs)
       │
       ├─ Pour chaque epoch:
       │   │
       │   ├─ Pour chaque batch (63 batches):
       │   │   │
       │   │   ├─ FORWARD PASS
       │   │   │   └─ Modèle fait des prédictions
       │   │   │
       │   │   ├─ CALCUL LOSS
       │   │   │   └─ Mesure l'erreur (prédictions vs vérités)
       │   │   │
       │   │   ├─ BACKWARD PASS
       │   │   │   └─ Calcule les gradients (rétropropagation)
       │   │   │
       │   │   └─ UPDATE
       │   │       └─ Ajuste les poids (optimizer.step())
       │   │
       │   └─ → Le modèle s'améliore à chaque batch!
       │
       └─ → Après 20 epochs, le modèle est performant!
    
    
    LES ACTEURS PRINCIPAUX :
    ═══════════════════════
    
    MODÈLE (Transformer)
       - Contient les POIDS (paramètres à apprendre)
       - Fait des PRÉDICTIONS
       - Initialement nul, devient expert après entraînement
    
    LOSS (CrossEntropyLoss)
       - MESURE l'erreur du modèle
       - Guide l'apprentissage : "Le modèle doit minimiser cette valeur"
    
    GRADIENTS
       - DIRECTION pour améliorer les poids
       - Calculés par backward()
       - Indiquent "comment modifier chaque poids"
    
    OPTIMIZER (Adam)
       - AJUSTE les poids selon les gradients
       - Formule : poids = poids - lr * gradient
       - Fait progresser le modèle
    
    
    ANALOGIE : Apprendre à jouer aux fléchettes
    ═══════════════════════════════════════════
    
    BUT : Atteindre le centre de la cible
    
    1. FORWARD : Tu lances une fléchette
       → Prédiction
    
    2. LOSS : Tu mesures la distance au centre
       → Erreur
    
    3. BACKWARD : Tu analyses "comment ajuster ton geste"
       → Gradients
    
    4. UPDATE : Tu modifies légèrement ton prochain lancer
       → Mise à jour des poids
    
    Après 1260 lancers (20 epochs × 63 batches):
    → Tu deviens un expert!
    
    
    POURQUOI ÇA MARCHE ?
    ═══════════════════
    
    ✅ GRADIENT DESCENT : On descend progressivement vers le minimum
       Comme descendre une montagne en suivant la pente
    
    ✅ BACKPROPAGATION : On propage l'erreur en arrière
       Permet de calculer le gradient pour CHAQUE poids
    
    ✅ ITÉRATIONS MULTIPLES : On s'améliore petit à petit
       1 itération = petit progrès
       1260 itérations = grande expertise!
    
    ✅ DONNÉES VARIÉES : On apprend sur plein d'exemples
       Le modèle généralise au lieu de mémoriser
    
    
    VOCABULAIRE RÉCAPITULATIF :
    ══════════════════════════
    
    Dataset       : Collection d'exemples (src, tgt)
    Batch         : Groupe de 32 exemples traités ensemble
    Itération     : Traitement d'1 batch (forward + backward + update)
    Epoch         : Passage complet sur toutes les données
    Modèle        : Réseau de neurones avec poids à apprendre
    Loss          : Mesure d'erreur à minimiser
    Gradient      : Direction pour améliorer les poids
    Optimizer     : Algorithme qui ajuste les poids
    Entraînement  : Processus complet d'apprentissage
    
    
    RÉSULTAT FINAL :
    ═══════════════
    
    Après entraînement, le modèle peut traduire:
    
    "42"   → "four two"        ✓
    "123"  → "one two three"   ✓
    "007"  → "zero zero seven" ✓
    "9876" → "nine eight seven six" ✓
    
    Le modèle a APPRIS cette tâche grâce à :
    - 20 epochs
    - 1260 mises à jour
    - Gradient descent + backpropagation
    - Des données d'entraînement variées
    """)


if __name__ == "__main__":
    print("\n🎓 GUIDE COMPLET : COMPRENDRE L'ENTRAÎNEMENT")
    print("=" * 70)
    
    etape1_preparation_donnees()
    input("\n[Appuyez sur Entrée pour continuer...]")
    
    etape2_creation_modele()
    input("\n[Appuyez sur Entrée pour continuer...]")
    
    etape3_une_iteration_detaillee()
    input("\n[Appuyez sur Entrée pour continuer...]")
    
    etape4_une_epoch_complete()
    input("\n[Appuyez sur Entrée pour continuer...]")
    
    etape5_entrainement_complet()
    input("\n[Appuyez sur Entrée pour continuer...]")
    
    resume_final()
    
    print("\n" + "="*70)
    print("🎉 Vous comprenez maintenant l'entraînement en profondeur!")
    print("="*70)
