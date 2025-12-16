#!/usr/bin/env python3
"""
Tests pour example9_toy_seq2seq.py

Lance les tests avec:
    pytest tests/test_toy_seq2seq.py -v
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.examples.example9_toy_seq2seq import (
    SignalDataset,
    SimpleSignalTransformer,
    collate_fn,
    train_epoch,
    evaluate
)


class TestSignalDataset:
    """Tests pour le dataset de signaux."""
    
    def test_dataset_creation(self):
        """Test que le dataset se crée correctement."""
        dataset = SignalDataset(num_samples=10, seq_len=50, seed=42)
        assert len(dataset) == 10
    
    def test_dataset_shapes(self):
        """Test les formes des tenseurs."""
        dataset = SignalDataset(num_samples=5, seq_len=64, seed=42)
        input_signal, output_signal, peak_info = dataset[0]
        
        assert input_signal.shape == (64,)
        assert output_signal.shape == (64,)
        assert isinstance(peak_info, dict)
    
    def test_dataset_peak_info(self):
        """Test que peak_info contient les bonnes clés."""
        dataset = SignalDataset(num_samples=5, seq_len=100, seed=42)
        _, _, peak_info = dataset[0]
        
        required_keys = ['type_a_positions', 'type_a_heights', 
                        'type_b_positions', 'type_b_heights',
                        'avg_a', 'avg_b']
        
        for key in required_keys:
            assert key in peak_info, f"Clé manquante: {key}"
    
    def test_dataset_peaks_count(self):
        """Test le nombre de pics par type."""
        dataset = SignalDataset(
            num_samples=5, 
            seq_len=100, 
            num_peaks_per_type=2,
            seed=42
        )
        _, _, peak_info = dataset[0]
        
        assert len(peak_info['type_a_positions']) == 2
        assert len(peak_info['type_b_positions']) == 2
    
    def test_dataset_reproducibility(self):
        """Test que le seed produit des résultats reproductibles."""
        dataset1 = SignalDataset(num_samples=5, seq_len=50, seed=123)
        dataset2 = SignalDataset(num_samples=5, seq_len=50, seed=123)
        
        input1, _, _ = dataset1[0]
        input2, _, _ = dataset2[0]
        
        assert torch.allclose(input1, input2)
    
    def test_peaks_within_bounds(self):
        """Test que les pics sont dans les limites du signal."""
        dataset = SignalDataset(num_samples=10, seq_len=100, seed=42)
        
        for i in range(len(dataset)):
            _, _, peak_info = dataset[i]
            
            for pos in peak_info['type_a_positions']:
                assert 0 <= pos < 100, f"Position hors limites: {pos}"
            
            for pos in peak_info['type_b_positions']:
                assert 0 <= pos < 100, f"Position hors limites: {pos}"
    
    def test_output_is_averaged(self):
        """Test que l'output contient les moyennes correctes."""
        dataset = SignalDataset(num_samples=5, seq_len=100, seed=42)
        _, output_signal, peak_info = dataset[0]
        
        avg_a = peak_info['avg_a']
        avg_b = peak_info['avg_b']
        
        # Vérifie que la moyenne type A est calculée correctement
        expected_avg_a = np.mean(peak_info['type_a_heights'])
        assert abs(avg_a - expected_avg_a) < 0.01


class TestSimpleSignalTransformer:
    """Tests pour le modèle Transformer."""
    
    def test_model_creation(self):
        """Test que le modèle se crée correctement."""
        model = SimpleSignalTransformer(seq_len=64)
        assert model is not None
    
    def test_model_forward_shape(self):
        """Test la forme de sortie du modèle."""
        model = SimpleSignalTransformer(seq_len=64, d_model=32, num_heads=4)
        
        batch_size = 4
        x = torch.randn(batch_size, 64)
        
        output = model(x)
        
        assert output.shape == (batch_size, 64)
    
    def test_model_forward_single_sample(self):
        """Test avec un seul échantillon."""
        model = SimpleSignalTransformer(seq_len=50, d_model=32)
        
        x = torch.randn(1, 50)
        output = model(x)
        
        assert output.shape == (1, 50)
    
    def test_model_parameters_count(self):
        """Test que le modèle a des paramètres."""
        model = SimpleSignalTransformer(seq_len=64, d_model=64)
        
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0
    
    def test_model_trainable(self):
        """Test que le modèle peut être entraîné (gradients)."""
        model = SimpleSignalTransformer(seq_len=32, d_model=32, num_heads=2)
        
        x = torch.randn(2, 32)
        target = torch.randn(2, 32)
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        
        # Vérifie que les gradients sont calculés
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_eval_mode(self):
        """Test le mode évaluation."""
        model = SimpleSignalTransformer(seq_len=32)
        
        model.eval()
        assert not model.training
        
        model.train()
        assert model.training


class TestCollateFn:
    """Tests pour la fonction de collation."""
    
    def test_collate_basic(self):
        """Test la collation basique."""
        dataset = SignalDataset(num_samples=4, seq_len=50, seed=42)
        
        batch = [dataset[i] for i in range(4)]
        inputs, outputs, peak_infos = collate_fn(batch)
        
        assert inputs.shape == (4, 50)
        assert outputs.shape == (4, 50)
        assert len(peak_infos) == 4


class TestTraining:
    """Tests pour les fonctions d'entraînement."""
    
    def test_train_epoch_runs(self):
        """Test qu'une epoch d'entraînement s'exécute."""
        dataset = SignalDataset(num_samples=32, seq_len=32, seed=42)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=8, collate_fn=collate_fn
        )
        
        model = SimpleSignalTransformer(seq_len=32, d_model=32, num_heads=2, num_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        device = torch.device('cpu')
        
        loss = train_epoch(model, loader, optimizer, criterion, device)
        
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_evaluate_runs(self):
        """Test que l'évaluation s'exécute."""
        dataset = SignalDataset(num_samples=16, seq_len=32, seed=42)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=8, collate_fn=collate_fn
        )
        
        model = SimpleSignalTransformer(seq_len=32, d_model=32, num_heads=2, num_layers=1)
        criterion = torch.nn.MSELoss()
        device = torch.device('cpu')
        
        loss = evaluate(model, loader, criterion, device)
        
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_training_reduces_loss(self):
        """Test que l'entraînement réduit la loss (sur plusieurs epochs)."""
        dataset = SignalDataset(num_samples=64, seq_len=32, seed=42)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=16, collate_fn=collate_fn
        )
        
        model = SimpleSignalTransformer(seq_len=32, d_model=64, num_heads=4, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        device = torch.device('cpu')
        
        # Première epoch
        initial_loss = train_epoch(model, loader, optimizer, criterion, device)
        
        # Quelques epochs supplémentaires
        for _ in range(5):
            train_epoch(model, loader, optimizer, criterion, device)
        
        final_loss = train_epoch(model, loader, optimizer, criterion, device)
        
        # La loss devrait avoir diminué
        assert final_loss < initial_loss, \
            f"Loss n'a pas diminué: {initial_loss:.4f} → {final_loss:.4f}"


class TestIntegration:
    """Tests d'intégration end-to-end."""
    
    def test_full_pipeline(self):
        """Test le pipeline complet: dataset → model → training → prediction."""
        # Dataset
        train_dataset = SignalDataset(num_samples=50, seq_len=32, seed=42)
        val_dataset = SignalDataset(num_samples=10, seq_len=32, seed=123)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=10, shuffle=True, collate_fn=collate_fn
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=10, collate_fn=collate_fn
        )
        
        # Model
        model = SimpleSignalTransformer(
            seq_len=32, d_model=32, num_heads=2, num_layers=2
        )
        
        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        device = torch.device('cpu')
        
        # Train 3 epochs
        for _ in range(3):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, criterion, device)
        
        # Prediction
        model.eval()
        with torch.no_grad():
            input_signal, _, _ = val_dataset[0]
            output = model(input_signal.unsqueeze(0))
        
        assert output.shape == (1, 32)
        assert val_loss >= 0
    
    def test_model_improves_on_simple_task(self):
        """Test que le modèle apprend sur une tâche simple."""
        # Créer un dataset très simple (copier l'input)
        torch.manual_seed(42)
        
        # Dataset où output = input (tâche d'identité)
        class IdentityDataset(torch.utils.data.Dataset):
            def __init__(self, n_samples, seq_len):
                self.data = [torch.randn(seq_len) for _ in range(n_samples)]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                x = self.data[idx]
                return x, x, {}  # input = output
        
        dataset = IdentityDataset(100, 32)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=20, 
            collate_fn=lambda b: (
                torch.stack([x[0] for x in b]),
                torch.stack([x[1] for x in b]),
                [x[2] for x in b]
            )
        )
        
        model = SimpleSignalTransformer(seq_len=32, d_model=64, num_heads=4, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        device = torch.device('cpu')
        
        # Train
        losses = []
        for _ in range(10):
            loss = train_epoch(model, loader, optimizer, criterion, device)
            losses.append(loss)
        
        # La loss devrait diminuer significativement
        assert losses[-1] < losses[0] * 0.8, \
            f"Le modèle n'apprend pas assez vite: {losses[0]:.4f} → {losses[-1]:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
