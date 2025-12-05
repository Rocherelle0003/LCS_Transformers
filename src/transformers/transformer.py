"""
Complete Transformer Model

This module implements the complete Transformer architecture as described in 
"An Introduction to Transformers" by Richard E. Turner.

The Transformer model consists of:
1. Source and target embeddings
2. Positional encodings
3. Encoder stack
4. Decoder stack
5. Output projection layer

This implementation supports sequence-to-sequence tasks like
machine translation.

References:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - "An Introduction to Transformers" (Turner, 2024)
"""

import math
import torch
import torch.nn as nn

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """
    Complete Transformer Model for Sequence-to-Sequence tasks.
    
    This implements the full encoder-decoder Transformer architecture
    suitable for tasks like machine translation, summarization, etc.
    
    Architecture:
    1. Source tokens → Source Embedding + Positional Encoding → Encoder
    2. Target tokens → Target Embedding + Positional Encoding → Decoder
    3. Decoder output → Linear → Softmax → Output probabilities
    
    Args:
        src_vocab_size (int): Source vocabulary size
        tgt_vocab_size (int): Target vocabulary size
        d_model (int): Model dimension. Default: 512
        num_heads (int): Number of attention heads. Default: 8
        d_ff (int): Feed-forward hidden dimension. Default: 2048
        num_encoder_layers (int): Number of encoder layers. Default: 6
        num_decoder_layers (int): Number of decoder layers. Default: 6
        max_seq_len (int): Maximum sequence length. Default: 5000
        dropout (float): Dropout probability. Default: 0.1
        share_embeddings (bool): Share source and target embeddings. Default: False
    
    Example:
        >>> model = Transformer(
        ...     src_vocab_size=10000,
        ...     tgt_vocab_size=10000,
        ...     d_model=512,
        ...     num_heads=8,
        ...     d_ff=2048,
        ...     num_encoder_layers=6,
        ...     num_decoder_layers=6
        ... )
        >>> src = torch.randint(0, 10000, (2, 20))  # (batch, src_len)
        >>> tgt = torch.randint(0, 10000, (2, 15))  # (batch, tgt_len)
        >>> output = model(src, tgt)
        >>> print(output.shape)  # torch.Size([2, 15, 10000])
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
        share_embeddings: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        
        if share_embeddings:
            assert src_vocab_size == tgt_vocab_size, \
                "Shared embeddings require same vocab size"
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout
        )
        
        # Encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_decoder_layers,
            dropout=dropout
        )
        
        # Output projection (to vocabulary)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(
        self, 
        src: torch.Tensor, 
        src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Encode the source sequence.
        
        Args:
            src: Source token indices of shape (batch, src_len)
            src_mask: Optional source mask
        
        Returns:
            Encoded representation of shape (batch, src_len, d_model)
        """
        # Embed and add positional encoding
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.positional_encoding(src_embedded)
        
        # Encode
        return self.encoder(src_embedded, src_mask)
    
    def decode(
        self, 
        tgt: torch.Tensor, 
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Decode the target sequence given encoder output.
        
        Args:
            tgt: Target token indices of shape (batch, tgt_len)
            memory: Encoder output of shape (batch, src_len, d_model)
            tgt_mask: Optional target mask
            memory_mask: Optional cross-attention mask
        
        Returns:
            Decoded representation of shape (batch, tgt_len, d_model)
        """
        # Embed and add positional encoding
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        
        # Decode
        return self.decoder(tgt_embedded, memory, tgt_mask, memory_mask)
    
    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer.
        
        Args:
            src: Source token indices of shape (batch, src_len)
            tgt: Target token indices of shape (batch, tgt_len)
            src_mask: Optional source mask for encoder
            tgt_mask: Optional target mask for decoder
            memory_mask: Optional cross-attention mask
        
        Returns:
            Output logits of shape (batch, tgt_len, tgt_vocab_size)
        """
        # Encode source sequence
        memory = self.encode(src, src_mask)
        
        # Decode target sequence
        output = self.decode(tgt, memory, tgt_mask, memory_mask)
        
        # Project to vocabulary
        return self.output_projection(output)
    
    def generate(
        self, 
        src: torch.Tensor, 
        max_len: int,
        start_token: int,
        end_token: int = None,
        src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generate target sequence autoregressively.
        
        This implements greedy decoding for inference.
        
        Args:
            src: Source token indices of shape (batch, src_len)
            max_len: Maximum length of generated sequence
            start_token: Start-of-sequence token index
            end_token: End-of-sequence token index (optional)
            src_mask: Optional source mask
        
        Returns:
            Generated token indices of shape (batch, generated_len)
        """
        batch_size = src.size(0)
        device = src.device
        
        # Encode source
        memory = self.encode(src, src_mask)
        
        # Initialize with start token
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        # Autoregressive generation
        for _ in range(max_len - 1):
            # Decode
            output = self.decode(generated, memory)
            
            # Get next token (greedy)
            next_token_logits = self.output_projection(output[:, -1, :])
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have generated end token
            if end_token is not None and (next_token == end_token).all():
                break
        
        return generated
