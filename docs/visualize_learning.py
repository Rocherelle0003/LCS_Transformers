"""
VISUALISATION DE L'APPRENTISSAGE - 5 PREMIÈRES EPOCHS
====================================================

Ce script montre EN DÉTAIL comment le modèle apprend pendant les 5 premières epochs.
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

from src.examples.example4_translation import (
    NumberToWordVocab,
    NumberToWordDataset,
    collate_fn,
    Transformer,
    create_padding_mask,
    translate
)

DIGIT_WORDS = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
}


def train_one_batch_verbose(model, src, tgt, optimizer, criterion, pad_idx, device, batch_num):
    """Entraîne sur un batch et affiche les détails."""
    model.train()
    
    src, tgt = src.to(device), tgt.to(device)
    
    # Préparer input/output
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
    
    # Masque
    src_padding_mask = create_padding_mask(src, pad_idx)
    
    # Forward
    optimizer.zero_grad()
    output = model(src, tgt_input, src_mask=src_padding_mask)
    
    # Loss
    output_flat = output.contiguous().view(-1, output.size(-1))
    tgt_output_flat = tgt_output.contiguous().view(-1)
    loss = criterion(output_flat, tgt_output_flat)
    
    # Backward
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()


def test_model_detailed(model, vocab, device, test_numbers, epoch):
    """
    Teste le modèle en détail et montre EXACTEMENT ce qu'il prédit.
    Affiche les probabilités et le processus de génération.
    """
    model.eval()
    results = []
    
    print(f"\n   {'='*60}")
    print(f"   TEST DÉTAILLÉ DU MODÈLE - EPOCH {epoch}")
    print(f"   {'='*60}")
    
    for number in test_numbers:
        print(f"\n   Test: '{number}' → {' '.join([DIGIT_WORDS[d] for d in number])}")
        print(f"   {'-'*60}")
        
        src_tokens = vocab.encode_source(number)
        src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)
        
        # Afficher l'encodage source
        src_str = ' '.join([vocab.src_idx2token[i] for i in src_tokens])
        print(f"   Source encodée: {src_str}")
        
        # Encoder la source
        with torch.no_grad():
            memory = model.encode(src_tensor)
        
        print(f"   Encoder → Memory shape: {memory.shape}")
        
        # Génération token par token
        tgt_tokens = [vocab.sos_idx]
        predicted_words = []
        
        print(f"\n   Génération autoregressive:")
        max_len = 20
        
        for step in range(max_len):
            tgt = torch.tensor([tgt_tokens]).to(device)
            
            with torch.no_grad():
                # Decoder
                output = model.decode(tgt, memory)
                # Output projection
                logits = model.output_projection(output[:, -1, :])
                # Probabilités
                probs = torch.softmax(logits, dim=-1)
                
                # Top-3 prédictions
                top_probs, top_indices = torch.topk(probs[0], k=3)
                
                # Meilleure prédiction
                next_token = top_indices[0].item()
                next_prob = top_probs[0].item()
            
            # Afficher les top-3 prédictions
            print(f"   └─ Step {step+1}:")
            for i, (idx, prob) in enumerate(zip(top_indices[:3], top_probs[:3])):
                token = vocab.tgt_idx2token[idx.item()]
                marker = "✓" if i == 0 else " "
                print(f"      {marker} {token:10s} : {prob.item()*100:5.1f}%")
            
            # Si c'est <eos>, on arrête
            if next_token == vocab.eos_idx:
                print(f"      → Génération terminée (<eos> prédit)")
                break
            
            # Ajouter le token prédit
            tgt_tokens.append(next_token)
            word = vocab.tgt_idx2token[next_token]
            predicted_words.append(word)
            print(f"      → Token choisi: '{word}' (confiance: {next_prob*100:.1f}%)")
        
        # Résultat final
        expected = [DIGIT_WORDS[d] for d in number]
        correct = predicted_words == expected
        status = "✅ CORRECT" if correct else "❌ INCORRECT"
        
        print(f"\n   Résultat:")
        print(f"   • Prédit:  {' '.join(predicted_words) if predicted_words else '(vide)'}")
        print(f"   • Attendu: {' '.join(expected)}")
        print(f"   • Statut:  {status}")
        
        results.append({
            'number': number,
            'predicted': predicted_words,
            'expected': expected,
            'correct': correct
        })
    
    accuracy = sum(1 for r in results if r['correct']) / len(results)
    print(f"\n   {'='*60}")
    print(f"   PRÉCISION GLOBALE: {accuracy*100:.1f}% ({sum(1 for r in results if r['correct'])}/{len(results)})")
    print(f"   {'='*60}")
    
    return results, accuracy


def test_predictions_verbose(model, vocab, device, test_numbers):
    """Teste les prédictions et affiche les résultats détaillés."""
    model.eval()
    results = []
    
    print("\n   Prédictions sur exemples de test:")
    print("   " + "-" * 50)
    
    for number in test_numbers:
        src_tokens = vocab.encode_source(number)
        src_tensor = torch.tensor(src_tokens)
        
        predicted = translate(model, src_tensor, vocab, device)
        expected = [DIGIT_WORDS[d] for d in number]
        
        correct = predicted == expected
        status = "✓" if correct else "✗"
        
        pred_str = ' '.join(predicted) if predicted else '(vide)'
        exp_str = ' '.join(expected)
        
        print(f"   {status} '{number}' → {pred_str:20} (attendu: {exp_str})")
        
        results.append({
            'number': number,
            'predicted': predicted,
            'expected': expected,
            'correct': correct
        })
    
    accuracy = sum(1 for r in results if r['correct']) / len(results)
    print(f"   Précision: {accuracy*100:.1f}%")
    
    return results, accuracy


def visualize_learning_progress(epoch_data):
    """Crée des visualisations de l'apprentissage."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Visualisation de l\'Apprentissage - 5 Premières Epochs', 
                 fontsize=16, fontweight='bold')
    
    # 1. Loss par epoch
    ax1 = axes[0, 0]
    epochs = [d['epoch'] for d in epoch_data]
    train_losses = [d['train_loss'] for d in epoch_data]
    
    ax1.plot(epochs, train_losses, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss Moyenne', fontsize=12)
    ax1.set_title('Évolution de la Loss pendant l\'entraînement', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # Annoter chaque point
    for i, (e, loss) in enumerate(zip(epochs, train_losses)):
        ax1.annotate(f'{loss:.3f}', 
                    xy=(e, loss), 
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9)
    
    # 2. Précision par epoch
    ax2 = axes[0, 1]
    accuracies = [d['accuracy'] * 100 for d in epoch_data]
    
    ax2.plot(epochs, accuracies, 'g-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Précision (%)', fontsize=12)
    ax2.set_title('Évolution de la Précision', fontsize=13)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    
    # Annoter chaque point
    for i, (e, acc) in enumerate(zip(epochs, accuracies)):
        ax2.annotate(f'{acc:.1f}%', 
                    xy=(e, acc), 
                    xytext=(5, -15),
                    textcoords='offset points',
                    fontsize=9)
    
    # 3. Amélioration de la loss
    ax3 = axes[1, 0]
    if len(train_losses) > 1:
        improvements = [train_losses[0] - loss for loss in train_losses]
        ax3.bar(epochs, improvements, color='purple', alpha=0.7)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Réduction de Loss', fontsize=12)
        ax3.set_title('Amélioration par rapport à Epoch 1', fontsize=13)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Annoter les barres
        for i, (e, imp) in enumerate(zip(epochs, improvements)):
            ax3.text(e, imp + 0.02, f'{imp:.3f}', 
                    ha='center', va='bottom', fontsize=9)
    
    # 4. Détails par epoch
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Créer un tableau de résumé
    summary_text = "RÉSUMÉ DE L'APPRENTISSAGE\n" + "="*35 + "\n\n"
    
    for d in epoch_data:
        summary_text += f"Epoch {d['epoch']}:\n"
        summary_text += f"  Loss: {d['train_loss']:.4f}\n"
        summary_text += f"  Précision: {d['accuracy']*100:.1f}%\n"
        summary_text += f"  Batches: {d['num_batches']}\n"
        
        if d['epoch'] == 1:
            summary_text += f"  État: Apprentissage initial\n"
        elif d['accuracy'] < 0.5:
            summary_text += f"  État: En apprentissage...\n"
        elif d['accuracy'] < 0.9:
            summary_text += f"  État: Progrès significatif!\n"
        else:
            summary_text += f"  État: Quasi-expert!\n"
        summary_text += "\n"
    
    # Statistiques globales
    total_improvement = train_losses[0] - train_losses[-1]
    acc_improvement = (accuracies[-1] - accuracies[0])
    
    summary_text += f"\nSTATISTIQUES GLOBALES:\n"
    summary_text += f"  Réduction totale loss: {total_improvement:.4f}\n"
    summary_text += f"  Amélioration précision: +{acc_improvement:.1f}%\n"
    
    ax4.text(0.05, 0.95, summary_text, 
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = Path(__file__).parent.parent / 'src' / 'tmp' / 'learning_progress_5epochs.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualisation sauvegardée: {output_path}")
    
    plt.show()


def main():
    """Entraîne le modèle sur 5 epochs avec visualisation détaillée."""
    
    print("\n" + "="*70)
    print("VISUALISATION DE L'APPRENTISSAGE - 5 PREMIÈRES EPOCHS")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    # Vocabulaire et données
    vocab = NumberToWordVocab()
    train_dataset = NumberToWordDataset(vocab, num_samples=2000, seed=42)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, vocab.pad_idx)
    )
    
    print(f"📚 Dataset: {len(train_dataset)} exemples")
    print(f"📦 Batches par epoch: {len(train_loader)}")
    
    # Modèle
    model = Transformer(
        src_vocab_size=vocab.src_vocab_size,
        tgt_vocab_size=vocab.tgt_vocab_size,
        d_model=64,
        num_heads=4,
        d_ff=256,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres du modèle: {num_params:,}")
    
    # Optimizer et loss
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Exemples de test
    test_numbers = ['42', '123', '7', '89']
    
    print("\n" + "="*70)
    print("DÉBUT DE L'ENTRAÎNEMENT")
    print("="*70)
    
    epoch_data = []
    
    # Entraîner sur 5 epochs
    for epoch in range(1, 6):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/5")
        print(f"{'='*70}")
        
        model.train()
        batch_losses = []
        
        # Afficher progression détaillée pour les premiers batches
        for batch_idx, (src, tgt) in enumerate(train_loader):
            loss = train_one_batch_verbose(
                model, src, tgt, optimizer, criterion, 
                vocab.pad_idx, device, batch_idx
            )
            batch_losses.append(loss)
            
            # Afficher progression tous les 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = np.mean(batch_losses[-10:])
                progress = (batch_idx + 1) / len(train_loader) * 100
                print(f"   Batch {batch_idx+1:2d}/{len(train_loader)} "
                      f"({progress:5.1f}%) - Loss: {avg_loss:.4f}")
        
        # Loss moyenne de l'epoch
        epoch_loss = np.mean(batch_losses)
        print(f"\n   ✓ Loss moyenne de l'epoch: {epoch_loss:.4f}")
        
        # Tester les prédictions EN DÉTAIL
        results, accuracy = test_model_detailed(model, vocab, device, test_numbers, epoch)
        
        # Sauvegarder les données
        epoch_data.append({
            'epoch': epoch,
            'train_loss': epoch_loss,
            'accuracy': accuracy,
            'num_batches': len(train_loader),
            'results': results
        })
        
        # Analyse de progression
        print(f"\n   ANALYSE:")
        if epoch == 1:
            print(f"      • Le modèle commence à apprendre les patterns de base")
            print(f"      • Loss initiale élevée = prédictions quasi-aléatoires")
        else:
            prev_loss = epoch_data[epoch-2]['train_loss']
            loss_reduction = prev_loss - epoch_loss
            loss_reduction_pct = (loss_reduction / prev_loss) * 100
            
            print(f"      • Réduction de loss: {loss_reduction:.4f} ({loss_reduction_pct:.1f}%)")
            
            if loss_reduction > 0.1:
                print(f"      • Excellent progrès! Le modèle apprend rapidement")
            elif loss_reduction > 0.05:
                print(f"      • Bon progrès, apprentissage stable")
            elif loss_reduction > 0:
                print(f"      • Petit progrès, le modèle affine ses prédictions")
            else:
                print(f"      • Attention: pas d'amélioration")
            
            if accuracy > 0.75:
                print(f"      • Très bonne précision! Le modèle maîtrise la tâche")
            elif accuracy > 0.5:
                print(f"      • Précision correcte, encore des erreurs")
            else:
                print(f"      • Précision faible, apprentissage en cours")
    
    
    # Résumé final
    print("\n" + "="*70)
    print("RÉSUMÉ DE L'APPRENTISSAGE")
    print("="*70)
    
    print("\n📊 ÉVOLUTION DE LA LOSS:")
    for i, d in enumerate(epoch_data):
        marker = "→" if i == 0 else "↓"
        print(f"   Epoch {d['epoch']}: {d['train_loss']:.4f} {marker}")
    
    total_reduction = epoch_data[0]['train_loss'] - epoch_data[-1]['train_loss']
    reduction_pct = (total_reduction / epoch_data[0]['train_loss']) * 100
    print(f"\n   Réduction totale: {total_reduction:.4f} ({reduction_pct:.1f}%)")
    
    print("\n🎯 ÉVOLUTION DE LA PRÉCISION:")
    for d in epoch_data:
        bar_length = int(d['accuracy'] * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"   Epoch {d['epoch']}: [{bar}] {d['accuracy']*100:.1f}%")
    
    print("\n💡 OBSERVATIONS:")
    if epoch_data[-1]['accuracy'] > 0.9:
        print("   ✓ Le modèle a très bien appris en seulement 5 epochs!")
        print("   ✓ Il maîtrise la tâche de traduction chiffres→mots")
    elif epoch_data[-1]['accuracy'] > 0.7:
        print("   ✓ Le modèle a bien progressé en 5 epochs")
        print("   • Quelques epochs supplémentaires amélioreront les résultats")
    else:
        print("   • Le modèle est encore en phase d'apprentissage")
        print("   • Continuer l'entraînement sur plus d'epochs")
    
    # Créer les visualisations
    print("\n📊 Génération des visualisations...")
    visualize_learning_progress(epoch_data)
    
    print("\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ!")
    print("="*70)


if __name__ == "__main__":
    main()
