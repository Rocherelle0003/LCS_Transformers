#!/usr/bin/env python3
"""
Example 4: Sequence-to-Sequence Translation

This example demonstrates a complete sequence-to-sequence translation
task using Transformers. We'll implement a simple number-to-word
translation system (e.g., "123" → "one two three").

This example covers:
1. Data preparation and tokenization
2. Training loop implementation
3. Inference with autoregressive decoding
4. Attention visualization for translation

Based on "An Introduction to Transformers" by Richard E. Turner

Run this example:
    python -m src.examples.example4_translation
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import random

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transformers import Transformer


# Special tokens
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'  # Start of sequence
EOS_TOKEN = '<eos>'  # End of sequence

# Vocabulary for number-to-word translation
DIGIT_WORDS = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
}


class NumberToWordVocab:
    """Simple vocabulary for number-to-word translation."""
    
    def __init__(self):
        # Source vocabulary (digits + special tokens)
        self.src_special = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
        self.src_tokens = list('0123456789')
        self.src_vocab = self.src_special + self.src_tokens
        self.src_token2idx = {t: i for i, t in enumerate(self.src_vocab)}
        self.src_idx2token = {i: t for t, i in self.src_token2idx.items()}
        
        # Target vocabulary (words + special tokens)
        self.tgt_special = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
        self.tgt_tokens = list(DIGIT_WORDS.values())
        self.tgt_vocab = self.tgt_special + self.tgt_tokens
        self.tgt_token2idx = {t: i for i, t in enumerate(self.tgt_vocab)}
        self.tgt_idx2token = {i: t for t, i in self.tgt_token2idx.items()}
    
    @property
    def src_vocab_size(self) -> int:
        return len(self.src_vocab)
    
    @property
    def tgt_vocab_size(self) -> int:
        return len(self.tgt_vocab)
    
    @property
    def pad_idx(self) -> int:
        return self.src_token2idx[PAD_TOKEN]
    
    @property
    def sos_idx(self) -> int:
        return self.src_token2idx[SOS_TOKEN]
    
    @property
    def eos_idx(self) -> int:
        return self.src_token2idx[EOS_TOKEN]
    
    def encode_source(self, number: str) -> List[int]:
        """Convert number string to token indices."""
        tokens = [self.src_token2idx[SOS_TOKEN]]
        tokens.extend([self.src_token2idx[d] for d in number])
        tokens.append(self.src_token2idx[EOS_TOKEN])
        return tokens
    
    def encode_target(self, words: List[str]) -> List[int]:
        """Convert word list to token indices."""
        tokens = [self.tgt_token2idx[SOS_TOKEN]]
        tokens.extend([self.tgt_token2idx[w] for w in words])
        tokens.append(self.tgt_token2idx[EOS_TOKEN])
        return tokens
    
    def decode_source(self, indices: List[int]) -> str:
        """Convert indices back to number string."""
        return ''.join([
            self.src_idx2token[i] for i in indices 
            if self.src_idx2token[i] not in self.src_special
        ])
    
    def decode_target(self, indices: List[int]) -> List[str]:
        """Convert indices back to word list."""
        return [
            self.tgt_idx2token[i] for i in indices 
            if self.tgt_idx2token[i] not in self.tgt_special
        ]


class NumberToWordDataset(Dataset):
    """Dataset for number-to-word translation."""
    
    def __init__(
        self, 
        vocab: NumberToWordVocab, 
        num_samples: int = 1000,
        min_digits: int = 1,
        max_digits: int = 5,
        seed: int = 42
    ):
        self.vocab = vocab
        self.data = []
        
        random.seed(seed)
        
        for _ in range(num_samples):
            # Generate random number
            num_digits = random.randint(min_digits, max_digits)
            number = ''.join([random.choice('0123456789') for _ in range(num_digits)])
            
            # Convert to words
            words = [DIGIT_WORDS[d] for d in number]
            
            # Encode
            src = self.vocab.encode_source(number)
            tgt = self.vocab.encode_target(words)
            
            self.data.append((src, tgt))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.data[idx]


def collate_fn(batch: List[Tuple[List[int], List[int]]], pad_idx: int):
    """Collate batch with padding."""
    src_batch, tgt_batch = zip(*batch)
    
    # Find max lengths
    src_max_len = max(len(s) for s in src_batch)
    tgt_max_len = max(len(t) for t in tgt_batch)
    
    # Pad sequences
    src_padded = []
    tgt_padded = []
    
    for src, tgt in zip(src_batch, tgt_batch):
        src_padded.append(src + [pad_idx] * (src_max_len - len(src)))
        tgt_padded.append(tgt + [pad_idx] * (tgt_max_len - len(tgt)))
    
    return torch.tensor(src_padded), torch.tensor(tgt_padded)


def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Create mask for padding tokens.
    
    Creates a mask of shape (batch, 1, seq_len) that can be broadcasted
    for attention calculations.
    """
    # (batch, seq_len) -> True where padding
    mask = (seq == pad_idx)
    # (batch, 1, seq_len) for broadcasting
    return mask.unsqueeze(1)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    pad_idx: int,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        
        # Target input (without last token) and output (without first token)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Create padding mask for source
        src_padding_mask = create_padding_mask(src, pad_idx)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask=src_padding_mask)
        
        # Compute loss (flatten for cross entropy)
        output = output.contiguous().view(-1, output.size(-1))
        tgt_output = tgt_output.contiguous().view(-1)
        
        loss = criterion(output, tgt_output)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    pad_idx: int,
    device: torch.device
) -> float:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_padding_mask = create_padding_mask(src, pad_idx)
            output = model(src, tgt_input, src_mask=src_padding_mask)
            
            output = output.contiguous().view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)
            
            loss = criterion(output, tgt_output)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def translate(
    model: nn.Module,
    src: torch.Tensor,
    vocab: NumberToWordVocab,
    device: torch.device,
    max_len: int = 20
) -> List[str]:
    """Translate source sequence using greedy decoding."""
    model.eval()
    
    with torch.no_grad():
        src = src.unsqueeze(0).to(device)
        
        # Encode source
        memory = model.encode(src)
        
        # Initialize decoder input with SOS
        tgt_tokens = [vocab.sos_idx]
        
        for _ in range(max_len):
            tgt = torch.tensor([tgt_tokens]).to(device)
            
            # Decode
            output = model.decode(tgt, memory)
            
            # Get next token
            next_token = model.output_projection(output[:, -1, :]).argmax(dim=-1).item()
            
            if next_token == vocab.eos_idx:
                break
            
            tgt_tokens.append(next_token)
        
        # Decode tokens to words
        return vocab.decode_target(tgt_tokens[1:])  # Skip SOS


def demo_training():
    """Demonstrate training a small Transformer for translation."""
    print("=" * 60)
    print("Training Transformer for Number-to-Word Translation")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create vocabulary
    vocab = NumberToWordVocab()
    print(f"\nVocabulary sizes:")
    print(f"  Source: {vocab.src_vocab_size} (digits + special)")
    print(f"  Target: {vocab.tgt_vocab_size} (words + special)")
    
    # Create datasets
    train_dataset = NumberToWordDataset(vocab, num_samples=2000, seed=42)
    val_dataset = NumberToWordDataset(vocab, num_samples=200, seed=123)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    
    # Sample data
    print("\nSample data:")
    for i in range(3):
        src, tgt = train_dataset[i]
        print(f"  '{vocab.decode_source(src)}' → {vocab.decode_target(tgt)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, vocab.pad_idx)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32,
        collate_fn=lambda b: collate_fn(b, vocab.pad_idx)
    )
    
    # Create model (small for demonstration)
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
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\nTraining...")
    train_losses = []
    val_losses = []
    
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, 
                                  vocab.pad_idx, device)
        val_loss = evaluate(model, val_loader, criterion, vocab.pad_idx, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('src/tmp/training_progress.png', dpi=150)
    plt.close()
    print("\nTraining progress saved to src/tmp/training_progress.png")
    
    # Test translation
    print("\nTest Translations:")
    test_numbers = ['42', '123', '007', '9876']
    
    for number in test_numbers:
        src_tokens = vocab.encode_source(number)
        src_tensor = torch.tensor(src_tokens)
        
        predicted = translate(model, src_tensor, vocab, device)
        expected = [DIGIT_WORDS[d] for d in number]
        
        correct = predicted == expected
        status = "✓" if correct else "✗"
        print(f"  {status} '{number}' → {' '.join(predicted)} (expected: {' '.join(expected)})")
    
    return model, vocab


def demonstrate_attention_in_translation():
    """Show attention patterns during translation."""
    print("\n" + "=" * 60)
    print("Attention Visualization in Translation")
    print("=" * 60)
    
    print("""
During translation, we can visualize attention to understand
which source positions the model focuses on when generating
each target token.

For our number-to-word task:
- Each target word should attend strongly to its corresponding digit
- For '123' → 'one two three':
    - 'one' should attend to '1'
    - 'two' should attend to '2'
    - 'three' should attend to '3'

This interpretability is a key advantage of attention mechanisms!
""")


def explain_seq2seq_concepts():
    """Explain key sequence-to-sequence concepts."""
    print("\n" + "=" * 60)
    print("Sequence-to-Sequence Key Concepts")
    print("=" * 60)
    
    print("""
1. TEACHER FORCING (Training)
   ─────────────────────────
   During training, we use the GROUND TRUTH target tokens as decoder input,
   not the model's predictions. This provides stable gradients but creates
   exposure bias (model never sees its own mistakes during training).
   
   Input:  <sos> one two three
   Output: one two three <eos>
   
2. AUTOREGRESSIVE DECODING (Inference)
   ─────────────────────────────────
   During inference, we generate one token at a time:
   1. Input <sos> → predict "one"
   2. Input <sos> one → predict "two"
   3. Input <sos> one two → predict "three"
   4. Input <sos> one two three → predict <eos> → stop
   
3. DECODING STRATEGIES
   ───────────────────
   - Greedy: Take argmax at each step (fast but suboptimal)
   - Beam Search: Keep top-k hypotheses (better but slower)
   - Sampling: Sample from distribution (for diversity)
   - Top-k/Top-p: Sample from filtered distribution
   
4. HANDLING VARIABLE LENGTHS
   ────────────────────────
   - Padding: Pad shorter sequences to match longest in batch
   - Padding mask: Ignore padding tokens in attention
   - <eos> token: Know when to stop generating
""")


def main():
    """Run the translation demonstration."""
    print("\n" + "#" * 60)
    print("# EXAMPLE 4: Sequence-to-Sequence Translation")
    print("#" * 60)
    
    # Explain concepts first
    explain_seq2seq_concepts()
    
    # Train a small model
    model, vocab = demo_training()
    
    # Explain attention visualization
    demonstrate_attention_in_translation()
    
    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
1. COMPLETE PIPELINE
   - Tokenization: Convert text to indices
   - Embedding: Convert indices to vectors
   - Encoder: Build source representation
   - Decoder: Generate target autoregressively
   - Output projection: Convert to vocabulary probabilities

2. TRAINING CONSIDERATIONS
   - Teacher forcing for stable training
   - Cross-entropy loss with padding ignored
   - Learning rate scheduling often helps

3. INFERENCE CONSIDERATIONS
   - Autoregressive: Sequential generation
   - Decoding strategy affects quality vs speed
   - Beam search for best results

4. PRACTICAL TIPS
   - Start with small model for debugging
   - Monitor both train and val loss
   - Use gradient clipping for stability
   - Label smoothing can help generalization
""")


if __name__ == "__main__":
    main()
