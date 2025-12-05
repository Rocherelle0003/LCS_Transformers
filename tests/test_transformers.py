"""
Unit tests for Transformer components.
"""

import pytest
import torch
import torch.nn as nn
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transformers.attention import ScaledDotProductAttention, MultiHeadAttention
from src.transformers.positional_encoding import PositionalEncoding, LearnedPositionalEncoding
from src.transformers.feed_forward import FeedForwardNetwork
from src.transformers.encoder import TransformerEncoderBlock, TransformerEncoder
from src.transformers.decoder import TransformerDecoderBlock, TransformerDecoder
from src.transformers.transformer import Transformer


class TestScaledDotProductAttention:
    """Tests for ScaledDotProductAttention."""
    
    def test_output_shape(self):
        """Test that output shape matches expected dimensions."""
        attention = ScaledDotProductAttention()
        
        batch, heads, seq_len, d_k = 2, 8, 10, 64
        Q = torch.randn(batch, heads, seq_len, d_k)
        K = torch.randn(batch, heads, seq_len, d_k)
        V = torch.randn(batch, heads, seq_len, d_k)
        
        output, weights = attention(Q, K, V)
        
        assert output.shape == (batch, heads, seq_len, d_k)
        assert weights.shape == (batch, heads, seq_len, seq_len)
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 along key dimension."""
        attention = ScaledDotProductAttention()
        
        Q = torch.randn(2, 4, 8, 32)
        K = torch.randn(2, 4, 8, 32)
        V = torch.randn(2, 4, 8, 32)
        
        _, weights = attention(Q, K, V)
        
        # Each row should sum to 1
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    
    def test_masking(self):
        """Test that masked positions have zero attention weight."""
        attention = ScaledDotProductAttention()
        
        seq_len = 4
        Q = torch.randn(1, 1, seq_len, 8)
        K = torch.randn(1, 1, seq_len, 8)
        V = torch.randn(1, 1, seq_len, 8)
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        _, weights = attention(Q, K, V, mask=mask)
        
        # Upper triangle should be zero (masked out)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert weights[0, 0, i, j] < 1e-6


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""
    
    def test_output_shape(self):
        """Test that output maintains input dimensions."""
        d_model, num_heads = 512, 8
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        
        batch, seq_len = 2, 10
        x = torch.randn(batch, seq_len, d_model)
        
        output, weights = mha(x, x, x)
        
        assert output.shape == (batch, seq_len, d_model)
        assert weights.shape == (batch, num_heads, seq_len, seq_len)
    
    def test_different_query_key_lengths(self):
        """Test cross-attention with different Q and K lengths."""
        d_model, num_heads = 256, 4
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        
        batch = 2
        q_len, k_len = 8, 12
        
        Q = torch.randn(batch, q_len, d_model)
        K = torch.randn(batch, k_len, d_model)
        V = torch.randn(batch, k_len, d_model)
        
        output, weights = mha(Q, K, V)
        
        assert output.shape == (batch, q_len, d_model)
        assert weights.shape == (batch, num_heads, q_len, k_len)


class TestPositionalEncoding:
    """Tests for PositionalEncoding."""
    
    def test_output_shape(self):
        """Test that positional encoding maintains shape."""
        d_model, max_len = 64, 100
        pe = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
        
        batch, seq_len = 2, 50
        x = torch.randn(batch, seq_len, d_model)
        
        output = pe(x)
        
        assert output.shape == (batch, seq_len, d_model)
    
    def test_encoding_bounded(self):
        """Test that positional encoding values are bounded."""
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        
        # Encoding should be between -1 and 1 (sin/cos range)
        assert pe.pe.min() >= -1
        assert pe.pe.max() <= 1
    
    def test_different_positions_different_encodings(self):
        """Test that different positions have different encodings."""
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        
        pos_0 = pe.pe[0, 0, :]
        pos_1 = pe.pe[0, 1, :]
        
        assert not torch.allclose(pos_0, pos_1)


class TestFeedForwardNetwork:
    """Tests for FeedForwardNetwork."""
    
    def test_output_shape(self):
        """Test that FFN maintains input/output dimensions."""
        d_model, d_ff = 256, 1024
        ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff)
        
        batch, seq_len = 2, 10
        x = torch.randn(batch, seq_len, d_model)
        
        output = ffn(x)
        
        assert output.shape == (batch, seq_len, d_model)
    
    def test_activations(self):
        """Test different activation functions."""
        d_model, d_ff = 64, 256
        
        ffn_relu = FeedForwardNetwork(d_model, d_ff, activation='relu')
        ffn_gelu = FeedForwardNetwork(d_model, d_ff, activation='gelu')
        
        x = torch.randn(1, 5, d_model)
        
        out_relu = ffn_relu(x)
        out_gelu = ffn_gelu(x)
        
        assert out_relu.shape == x.shape
        assert out_gelu.shape == x.shape


class TestTransformerEncoder:
    """Tests for TransformerEncoder."""
    
    def test_single_block(self):
        """Test single encoder block."""
        block = TransformerEncoderBlock(d_model=128, num_heads=4, d_ff=512)
        
        x = torch.randn(2, 10, 128)
        output = block(x)
        
        assert output.shape == x.shape
    
    def test_full_encoder(self):
        """Test full encoder stack."""
        encoder = TransformerEncoder(
            d_model=128, num_heads=4, d_ff=512, num_layers=3
        )
        
        x = torch.randn(2, 10, 128)
        output = encoder(x)
        
        assert output.shape == x.shape
    
    def test_with_mask(self):
        """Test encoder with padding mask."""
        encoder = TransformerEncoder(
            d_model=128, num_heads=4, d_ff=512, num_layers=2
        )
        
        x = torch.randn(2, 10, 128)
        mask = torch.zeros(2, 10, 10, dtype=torch.bool)
        mask[0, :, 8:] = True  # Mask last 2 positions for first sample
        
        output = encoder(x, src_mask=mask)
        
        assert output.shape == x.shape


class TestTransformerDecoder:
    """Tests for TransformerDecoder."""
    
    def test_single_block(self):
        """Test single decoder block."""
        block = TransformerDecoderBlock(d_model=128, num_heads=4, d_ff=512)
        
        x = torch.randn(2, 8, 128)
        memory = torch.randn(2, 10, 128)
        output = block(x, memory)
        
        assert output.shape == x.shape
    
    def test_full_decoder(self):
        """Test full decoder stack."""
        decoder = TransformerDecoder(
            d_model=128, num_heads=4, d_ff=512, num_layers=3
        )
        
        x = torch.randn(2, 8, 128)
        memory = torch.randn(2, 10, 128)
        output = decoder(x, memory)
        
        assert output.shape == x.shape
    
    def test_causal_mask_generation(self):
        """Test causal mask generation."""
        block = TransformerDecoderBlock(d_model=64, num_heads=4, d_ff=256)
        
        mask = block.generate_causal_mask(5, torch.device('cpu'))
        
        expected = torch.tensor([
            [False, True, True, True, True],
            [False, False, True, True, True],
            [False, False, False, True, True],
            [False, False, False, False, True],
            [False, False, False, False, False]
        ])
        
        assert torch.equal(mask, expected)


class TestTransformer:
    """Tests for complete Transformer model."""
    
    def test_forward_pass(self):
        """Test forward pass through complete model."""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            d_ff=512,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        src = torch.randint(0, 1000, (2, 20))
        tgt = torch.randint(0, 1000, (2, 15))
        
        output = model(src, tgt)
        
        assert output.shape == (2, 15, 1000)
    
    def test_encode(self):
        """Test encoding only."""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            d_ff=512,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        src = torch.randint(0, 1000, (2, 20))
        memory = model.encode(src)
        
        assert memory.shape == (2, 20, 128)
    
    def test_generation(self):
        """Test autoregressive generation."""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            num_heads=2,
            d_ff=256,
            num_encoder_layers=1,
            num_decoder_layers=1
        )
        model.eval()
        
        src = torch.randint(0, 100, (1, 10))
        
        with torch.no_grad():
            generated = model.generate(
                src, 
                max_len=15, 
                start_token=1, 
                end_token=2
            )
        
        assert generated.shape[0] == 1
        assert generated.shape[1] <= 15
    
    def test_shared_embeddings(self):
        """Test model with shared embeddings."""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            d_ff=512,
            num_encoder_layers=2,
            num_decoder_layers=2,
            share_embeddings=True
        )
        
        # Check that embeddings are shared
        assert model.src_embedding is model.tgt_embedding
        
        src = torch.randint(0, 1000, (2, 20))
        tgt = torch.randint(0, 1000, (2, 15))
        
        output = model(src, tgt)
        assert output.shape == (2, 15, 1000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
