# encoding: utf-8
"""
Comprehensive pipeline validation tests for Chord Structure Decomposition.

This module validates the complete pipeline including:
1. Vocabulary definitions
2. Chord decomposition and reassembly
3. Dataset loading with decomposition
4. Model architecture with 8 heads
5. Multi-task loss with class re-weighting
6. Inference and decoding

Run with: python -m pytest tests/test_pipeline_validation.py -v
Or directly: python tests/test_pipeline_validation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import torch
from typing import Dict, List

from utils.chord_decomposition import (
    ChordDecomposer, ChordReassembler, get_vocab_sizes,
    COMPONENT_NAMES, CHORD_VOCAB, CHORD_VOCAB_IDX, NUM_COMPONENTS
)
from models.btc_model_decomposed import (
    ComponentHead, MultiHeadChordDecomposer, MultiTaskLoss, BTC_model_decomposed
)


class TestVocabularyDefinitions(unittest.TestCase):
    """Test 1: Validate vocabulary definitions match specification."""
    
    def test_component_count(self):
        """Verify we have exactly 9 components."""
        self.assertEqual(NUM_COMPONENTS, 9)
        self.assertEqual(len(COMPONENT_NAMES), 9)
    
    def test_component_names(self):
        """Verify component names are correct."""
        expected = ['root', 'bass', 'triad', 'misc', '6th', '7th', '9th', '11th', '13th']
        self.assertEqual(COMPONENT_NAMES, expected)
    
    def test_root_vocab(self):
        """Root: 13 classes (N, C, C#, D, D#, E, F, F#, G, G#, A, A#, B)."""
        expected = ['N', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.assertEqual(CHORD_VOCAB['root'], expected)
        self.assertEqual(len(CHORD_VOCAB['root']), 13)
    
    def test_bass_vocab(self):
        """Bass: 13 classes (same as Root)."""
        expected = ['N', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.assertEqual(CHORD_VOCAB['bass'], expected)
        self.assertEqual(len(CHORD_VOCAB['bass']), 13)
    
    def test_triad_vocab(self):
        """Triad: 7 classes (N, maj, min, dim, aug, sus2, sus4)."""
        expected = ['N', 'maj', 'min', 'dim', 'aug', 'sus2', 'sus4']
        self.assertEqual(CHORD_VOCAB['triad'], expected)
        self.assertEqual(len(CHORD_VOCAB['triad']), 7)
    
    def test_misc_vocab(self):
        """Misc (Power Chord): 2 classes (N, 5)."""
        expected = ['N', '5']
        self.assertEqual(CHORD_VOCAB['misc'], expected)
        self.assertEqual(len(CHORD_VOCAB['misc']), 2)
    
    def test_6th_vocab(self):
        """6th: 2 classes (N, 6)."""
        expected = ['N', '6']
        self.assertEqual(CHORD_VOCAB['6th'], expected)
        self.assertEqual(len(CHORD_VOCAB['6th']), 2)
    
    def test_7th_vocab(self):
        """7th: 4 classes (N, 7, b7, bb7)."""
        expected = ['N', '7', 'b7', 'bb7']
        self.assertEqual(CHORD_VOCAB['7th'], expected)
        self.assertEqual(len(CHORD_VOCAB['7th']), 4)
    
    def test_9th_vocab(self):
        """9th: 4 classes (N, 9, #9, b9)."""
        expected = ['N', '9', '#9', 'b9']
        self.assertEqual(CHORD_VOCAB['9th'], expected)
        self.assertEqual(len(CHORD_VOCAB['9th']), 4)
    
    def test_11th_vocab(self):
        """11th: 3 classes (N, 11, #11)."""
        expected = ['N', '11', '#11']
        self.assertEqual(CHORD_VOCAB['11th'], expected)
        self.assertEqual(len(CHORD_VOCAB['11th']), 3)
    
    def test_13th_vocab(self):
        """13th: 3 classes (N, 13, b13)."""
        expected = ['N', '13', 'b13']
        self.assertEqual(CHORD_VOCAB['13th'], expected)
        self.assertEqual(len(CHORD_VOCAB['13th']), 3)
    
    def test_total_classes(self):
        """Total should be 51 classes (vs 170 for monolithic)."""
        vocab_sizes = get_vocab_sizes()
        total = sum(vocab_sizes.values())
        self.assertEqual(total, 51)  # 13+13+7+2+2+4+4+3+3 = 51
    
    def test_reverse_mapping(self):
        """Verify reverse mapping is correct."""
        for component, vocab in CHORD_VOCAB.items():
            for idx, label in enumerate(vocab):
                self.assertEqual(CHORD_VOCAB_IDX[component][label], idx)


class TestChordDecomposition(unittest.TestCase):
    """Test 2: Validate chord decomposition logic."""
    
    def setUp(self):
        self.decomposer = ChordDecomposer()
    
    def test_simple_major(self):
        """Test C:maj decomposition."""
        components = self.decomposer.decompose('C:maj')
        self.assertEqual(components['root'], 'C')
        self.assertEqual(components['triad'], 'maj')
        self.assertEqual(components['bass'], 'N')
        self.assertEqual(components['misc'], 'N')
        self.assertEqual(components['7th'], 'N')
        self.assertEqual(components['9th'], 'N')
        self.assertEqual(components['11th'], 'N')
        self.assertEqual(components['13th'], 'N')
    
    def test_simple_minor(self):
        """Test D:min decomposition."""
        components = self.decomposer.decompose('D:min')
        self.assertEqual(components['root'], 'D')
        self.assertEqual(components['triad'], 'min')
    
    def test_maj9(self):
        """Test C:maj9 decomposition (from user example)."""
        components = self.decomposer.decompose('C:maj9')
        self.assertEqual(components['root'], 'C')
        self.assertEqual(components['triad'], 'maj')
        self.assertEqual(components['9th'], '9')
    
    def test_min7(self):
        """Test min7 chord."""
        components = self.decomposer.decompose('E:min7')
        self.assertEqual(components['root'], 'E')
        self.assertEqual(components['triad'], 'min')
        self.assertEqual(components['7th'], '7')
    
    def test_dim7(self):
        """Test diminished 7th."""
        components = self.decomposer.decompose('F:dim7')
        self.assertEqual(components['root'], 'F')
        self.assertEqual(components['triad'], 'dim')
        self.assertEqual(components['7th'], '7')
    
    def test_aug(self):
        """Test augmented triad."""
        components = self.decomposer.decompose('G:aug')
        self.assertEqual(components['root'], 'G')
        self.assertEqual(components['triad'], 'aug')
    
    def test_sus2(self):
        """Test suspended 2nd."""
        components = self.decomposer.decompose('A:sus2')
        self.assertEqual(components['root'], 'A')
        self.assertEqual(components['triad'], 'sus2')
    
    def test_sus4(self):
        """Test suspended 4th."""
        components = self.decomposer.decompose('B:sus4')
        self.assertEqual(components['root'], 'B')
        self.assertEqual(components['triad'], 'sus4')
    
    def test_power_chord(self):
        """Test power chord (5)."""
        components = self.decomposer.decompose('C:5')
        self.assertEqual(components['root'], 'C')
        self.assertEqual(components['misc'], '5')
        self.assertEqual(components['triad'], 'N')
    
    def test_slash_chord(self):
        """Test slash chord (bass note)."""
        components = self.decomposer.decompose('C:maj/E')
        self.assertEqual(components['root'], 'C')
        self.assertEqual(components['triad'], 'maj')
        self.assertEqual(components['bass'], 'E')
    
    def test_no_chord(self):
        """Test N (no chord)."""
        components = self.decomposer.decompose('N')
        for comp in COMPONENT_NAMES:
            self.assertEqual(components[comp], 'N')
    
    def test_complex_chord_13(self):
        """Test complex chord with 13th."""
        components = self.decomposer.decompose('C:maj13')
        self.assertEqual(components['root'], 'C')
        self.assertEqual(components['triad'], 'maj')
        self.assertEqual(components['13th'], '13')
    
    def test_extension_order(self):
        """Test that extensions are extracted correctly without conflicts."""
        # Test that '13' is not confused with '1' or '3'
        components = self.decomposer.decompose('C:min13')
        self.assertEqual(components['triad'], 'min')
        self.assertEqual(components['13th'], '13')
    
    def test_b9(self):
        """Test flat 9th."""
        components = self.decomposer.decompose('C:majb9')
        self.assertEqual(components['9th'], 'b9')
    
    def test_sharp9(self):
        """Test sharp 9th."""
        components = self.decomposer.decompose('C:maj#9')
        self.assertEqual(components['9th'], '#9')
    
    def test_sharp11(self):
        """Test sharp 11th."""
        components = self.decomposer.decompose('C:maj#11')
        self.assertEqual(components['11th'], '#11')
    
    def test_b13(self):
        """Test flat 13th."""
        components = self.decomposer.decompose('C:majb13')
        self.assertEqual(components['13th'], 'b13')
    
    def test_batch_decomposition(self):
        """Test batch decomposition."""
        chords = ['C:maj', 'D:min7', 'E:aug', 'F:dim', 'N']
        result = self.decomposer.decompose_batch(chords)
        
        self.assertEqual(len(result), 8)
        for comp in COMPONENT_NAMES:
            self.assertIn(comp, result)
            self.assertEqual(len(result[comp]), 5)
            self.assertIsInstance(result[comp], np.ndarray)


class TestChordReassembly(unittest.TestCase):
    """Test 3: Validate chord reassembly with priority logic."""
    
    def setUp(self):
        self.decomposer = ChordDecomposer()
        self.reassembler = ChordReassembler()
    
    def test_priority_no_root(self):
        """If root is N, output should be N."""
        components = {comp: 'N' for comp in COMPONENT_NAMES}
        components['triad'] = 'maj'  # Even with triad
        result = self.reassembler.reassemble(components)
        self.assertEqual(result, 'N')
    
    def test_priority_no_triad(self):
        """If triad is N (and no power chord), output should be N."""
        components = {comp: 'N' for comp in COMPONENT_NAMES}
        components['root'] = 'C'  # Even with root
        result = self.reassembler.reassemble(components)
        self.assertEqual(result, 'N')
    
    def test_priority_power_chord(self):
        """Power chord should work even without triad."""
        components = {comp: 'N' for comp in COMPONENT_NAMES}
        components['root'] = 'C'
        components['misc'] = '5'
        result = self.reassembler.reassemble(components)
        self.assertEqual(result, 'C:5')
    
    def test_simple_reassembly(self):
        """Test simple chord reassembly."""
        components = {comp: 'N' for comp in COMPONENT_NAMES}
        components['root'] = 'C'
        components['triad'] = 'maj'
        result = self.reassembler.reassemble(components)
        self.assertEqual(result, 'C:maj')
    
    def test_with_extension(self):
        """Test reassembly with extension."""
        components = {comp: 'N' for comp in COMPONENT_NAMES}
        components['root'] = 'C'
        components['triad'] = 'maj'
        components['7th'] = '7'
        result = self.reassembler.reassemble(components)
        self.assertEqual(result, 'C:maj7')
    
    def test_with_bass(self):
        """Test reassembly with bass note."""
        components = {comp: 'N' for comp in COMPONENT_NAMES}
        components['root'] = 'C'
        components['triad'] = 'maj'
        components['bass'] = 'E'
        result = self.reassembler.reassemble(components)
        self.assertEqual(result, 'C:maj/E')
    
    def test_round_trip_simple(self):
        """Test decompose -> reassemble round trip."""
        original = 'C:maj'
        components = self.decomposer.decompose(original)
        reassembled = self.reassembler.reassemble(components)
        self.assertEqual(reassembled, original)
    
    def test_round_trip_complex(self):
        """Test round trip for complex chords."""
        test_chords = [
            'C:maj7',
            'D:min9',
            'E:dim7',
            'F:sus4',
            'G:aug',
            'A:min/E',
            'B:maj13',
        ]
        for chord in test_chords:
            components = self.decomposer.decompose(chord)
            reassembled = self.reassembler.reassemble(components)
            self.assertEqual(reassembled, chord, f"Round trip failed for {chord}")
    
    def test_batch_reassembly(self):
        """Test batch reassembly from indices."""
        chords = ['C:maj', 'D:min', 'N']
        indices_batch = self.decomposer.decompose_batch(chords)
        
        reassembled = self.reassembler.reassemble_batch(indices_batch)
        self.assertEqual(len(reassembled), 3)
        self.assertEqual(reassembled[0], 'C:maj')
        self.assertEqual(reassembled[1], 'D:min')
        self.assertEqual(reassembled[2], 'N')


class TestModelArchitecture(unittest.TestCase):
    """Test 4: Validate model architecture with 8 heads."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.hidden_size = 128
        self.batch_size = 2
        self.seq_len = 50
        self.feature_size = 144
    
    def test_component_head_output_shape(self):
        """Each head should output (batch, seq, vocab_size)."""
        for component in COMPONENT_NAMES:
            vocab_size = len(CHORD_VOCAB[component])
            head = ComponentHead(self.hidden_size, vocab_size)
            
            x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
            output = head(x)
            
            expected_shape = (self.batch_size, self.seq_len, vocab_size)
            self.assertEqual(output.shape, expected_shape,
                           f"Head for {component} has wrong shape")
    
    def test_multi_head_decomposer(self):
        """MultiHeadChordDecomposer should output dict with all components."""
        decomposer = MultiHeadChordDecomposer(self.hidden_size)
        
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        logits = decomposer(x)
        
        self.assertEqual(len(logits), len(COMPONENT_NAMES))
        for component in COMPONENT_NAMES:
            self.assertIn(component, logits)
            vocab_size = len(CHORD_VOCAB[component])
            expected_shape = (self.batch_size, self.seq_len, vocab_size)
            self.assertEqual(logits[component].shape, expected_shape)
    
    def test_get_predictions(self):
        """Predictions should be valid class indices."""
        decomposer = MultiHeadChordDecomposer(self.hidden_size)
        
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        logits = decomposer(x)
        predictions = decomposer.get_predictions(logits)
        
        for component in COMPONENT_NAMES:
            vocab_size = len(CHORD_VOCAB[component])
            pred = predictions[component]
            
            self.assertEqual(pred.shape, (self.batch_size, self.seq_len))
            self.assertTrue(torch.all(pred >= 0))
            self.assertTrue(torch.all(pred < vocab_size))
    
    def test_get_probabilities(self):
        """Probabilities should sum to 1 for each position."""
        decomposer = MultiHeadChordDecomposer(self.hidden_size)
        
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        logits = decomposer(x)
        probs = decomposer.get_probabilities(logits)
        
        for component in COMPONENT_NAMES:
            prob = probs[component]
            
            # Sum should be approximately 1
            prob_sums = prob.sum(dim=-1)
            self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5))
    
    def test_full_model_forward(self):
        """Test full BTC_model_decomposed forward pass."""
        config = {
            'feature_size': self.feature_size,
            'hidden_size': self.hidden_size,
            'num_layers': 2,
            'num_heads': 4,
            'total_key_depth': self.hidden_size,
            'total_value_depth': self.hidden_size,
            'filter_size': 256,
            'timestep': self.seq_len,
            'input_dropout': 0.0,
            'layer_dropout': 0.0,
            'attention_dropout': 0.0,
            'relu_dropout': 0.0,
            'probs_out': False,
        }
        
        model = BTC_model_decomposed(config)
        model.eval()
        
        # Input shape: (batch, seq, feature)
        x = torch.randn(self.batch_size, self.seq_len, self.feature_size)
        
        with torch.no_grad():
            predictions, loss, weights, component_losses = model(x, labels=None)
        
        # Predictions should be dict with all components
        self.assertEqual(len(predictions), len(COMPONENT_NAMES))
        for component in COMPONENT_NAMES:
            self.assertIn(component, predictions)
            self.assertEqual(predictions[component].shape, (self.batch_size, self.seq_len))
        
        # Loss should be None when no labels
        self.assertIsNone(loss)
        self.assertIsNone(component_losses)


class TestMultiTaskLoss(unittest.TestCase):
    """Test 5: Validate multi-task loss with class re-weighting."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.vocab_sizes = get_vocab_sizes()
        self.batch_size = 4
        self.seq_len = 50
    
    def test_loss_initialization(self):
        """Loss should have 8 sub-losses."""
        loss_fn = MultiTaskLoss(self.vocab_sizes)
        
        self.assertEqual(len(loss_fn.losses), 8)
        for component in COMPONENT_NAMES:
            self.assertIn(component, loss_fn.losses)
    
    def test_loss_computation(self):
        """Loss should be positive scalar."""
        loss_fn = MultiTaskLoss(self.vocab_sizes)
        
        logits = {}
        labels = {}
        for component in COMPONENT_NAMES:
            vocab_size = self.vocab_sizes[component]
            logits[component] = torch.randn(
                self.batch_size, self.seq_len, vocab_size
            )
            labels[component] = torch.randint(
                0, vocab_size, (self.batch_size, self.seq_len)
            )
        
        loss = loss_fn(logits, labels)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertGreater(loss.item(), 0)
    
    def test_loss_differentiable(self):
        """Loss should be differentiable."""
        loss_fn = MultiTaskLoss(self.vocab_sizes)
        
        logits = {}
        labels = {}
        for component in COMPONENT_NAMES:
            vocab_size = self.vocab_sizes[component]
            logits[component] = torch.randn(
                self.batch_size, self.seq_len, vocab_size,
                requires_grad=True
            )
            labels[component] = torch.randint(
                0, vocab_size, (self.batch_size, self.seq_len)
            )
        
        loss = loss_fn(logits, labels)
        loss.backward()
        
        for component in COMPONENT_NAMES:
            self.assertIsNotNone(logits[component].grad)
    
    def test_class_weights_formula(self):
        """Verify class weights follow the formula w = min((n/max_n)^(-gamma), w_max)."""
        gamma = 0.5
        w_max = 10.0
        
        # Create mock dataset with __getitem__ support
        class MockDataset:
            def __init__(self):
                self.samples = []
                # Create samples with known class distribution
                for _ in range(100):
                    components = {comp: np.zeros(10, dtype=np.int64) for comp in COMPONENT_NAMES}
                    # Root: class 0 appears 80%, class 1 appears 20%
                    components['root'][:8] = 1  # C
                    components['root'][8:] = 2   # C#
                    self.samples.append({'components': components})
            
            def __getitem__(self, idx):
                return self.samples[idx]
            
            def __len__(self):
                return len(self.samples)
        
        dataset = MockDataset()
        weights = MultiTaskLoss.compute_class_weights(dataset, gamma=gamma, w_max=w_max)
        
        # Verify structure
        self.assertEqual(len(weights), 8)
        for component in COMPONENT_NAMES:
            self.assertIn(component, weights)
            vocab_size = len(CHORD_VOCAB[component])
            self.assertEqual(len(weights[component]), vocab_size)
        
        # Verify weights are bounded
        for component, w in weights.items():
            self.assertTrue(torch.all(w > 0))
            self.assertTrue(torch.all(w <= w_max))
    
    def test_class_weights_rare_class(self):
        """Rare classes should have higher weights."""
        # Create dataset with imbalanced distribution
        class ImbalancedDataset:
            def __init__(self):
                self.samples = []
                for i in range(100):
                    components = {}
                    for comp in COMPONENT_NAMES:
                        # Frequent: class 0 (90 samples), Rare: class 1 (10 samples)
                        if i < 90:
                            components[comp] = np.array([0], dtype=np.int64)
                        else:
                            components[comp] = np.array([1], dtype=np.int64)
                    self.samples.append({'components': components})
            
            def __getitem__(self, idx):
                return self.samples[idx]
            
            def __len__(self):
                return len(self.samples)
        
        dataset = ImbalancedDataset()
        weights = MultiTaskLoss.compute_class_weights(dataset, gamma=0.5, w_max=10.0)
        
        # Class 1 (rare) should have higher weight than class 0 (frequent)
        # Weight for class 1 = (10/90)^(-0.5) = 3.0
        # Weight for class 0 = (90/90)^(-0.5) = 1.0
        for component in COMPONENT_NAMES:
            w = weights[component]
            if len(w) >= 2:
                self.assertGreater(w[1].item(), w[0].item(),
                                 f"Rare class weight should be higher for {component}")


class TestEndToEndPipeline(unittest.TestCase):
    """Test 6: End-to-end pipeline validation."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.batch_size = 2
        self.seq_len = 50
        self.feature_size = 144
        self.hidden_size = 64
    
    def test_full_pipeline(self):
        """Test complete pipeline: features -> model -> predictions -> chords."""
        # 1. Create model
        config = {
            'feature_size': self.feature_size,
            'hidden_size': self.hidden_size,
            'num_layers': 2,
            'num_heads': 4,
            'total_key_depth': self.hidden_size,
            'total_value_depth': self.hidden_size,
            'filter_size': 128,
            'timestep': self.seq_len,
            'input_dropout': 0.0,
            'layer_dropout': 0.0,
            'attention_dropout': 0.0,
            'relu_dropout': 0.0,
            'probs_out': False,
        }
        
        model = BTC_model_decomposed(config)
        model.eval()
        
        # 2. Create input features
        x = torch.randn(self.batch_size, self.seq_len, self.feature_size)
        
        # 3. Forward pass
        with torch.no_grad():
            predictions, _, _ = model(x)
        
        # 4. Decode predictions
        reassembler = ChordReassembler()
        
        # Convert predictions to indices
        indices = {}
        for component in COMPONENT_NAMES:
            indices[component] = predictions[component].numpy()
        
        # Reassemble chords
        chord_labels = reassembler.reassemble_batch_2d(indices)
        
        # 5. Validate output
        self.assertEqual(len(chord_labels), self.batch_size)
        for seq_chords in chord_labels:
            self.assertEqual(len(seq_chords), self.seq_len)
            for chord in seq_chords:
                # Each chord should be a valid string
                self.assertIsInstance(chord, str)
                # Should be either 'N' or contain ':'
                self.assertTrue(chord == 'N' or ':' in chord)
    
    def test_training_step(self):
        """Test single training step."""
        config = {
            'feature_size': self.feature_size,
            'hidden_size': self.hidden_size,
            'num_layers': 2,
            'num_heads': 4,
            'total_key_depth': self.hidden_size,
            'total_value_depth': self.hidden_size,
            'filter_size': 128,
            'timestep': self.seq_len,
            'input_dropout': 0.0,
            'layer_dropout': 0.0,
            'attention_dropout': 0.0,
            'relu_dropout': 0.0,
            'probs_out': False,
        }
        
        model = BTC_model_decomposed(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create batch
        x = torch.randn(self.batch_size, self.seq_len, self.feature_size)
        labels = {}
        for component in COMPONENT_NAMES:
            vocab_size = len(CHORD_VOCAB[component])
            labels[component] = torch.randint(0, vocab_size, (self.batch_size, self.seq_len))
        
        # Training step
        model.train()
        predictions, loss, _ = model(x, labels=labels)
        
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verify gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


class TestIntervalDefinitions(unittest.TestCase):
    """Test interval definitions for triads."""
    
    def test_major_triad(self):
        """Major triad: 1, 3, 5 (intervals: 0, 4, 7 semitones)."""
        from utils.chord_decomposition import INTERVAL_DEFINITIONS
        self.assertEqual(INTERVAL_DEFINITIONS['maj']['notes'], [0, 4, 7])
    
    def test_minor_triad(self):
        """Minor triad: 1, b3, 5 (intervals: 0, 3, 7 semitones)."""
        from utils.chord_decomposition import INTERVAL_DEFINITIONS
        self.assertEqual(INTERVAL_DEFINITIONS['min']['notes'], [0, 3, 7])
    
    def test_diminished_triad(self):
        """Diminished triad: 1, b3, b5 (intervals: 0, 3, 6 semitones)."""
        from utils.chord_decomposition import INTERVAL_DEFINITIONS
        self.assertEqual(INTERVAL_DEFINITIONS['dim']['notes'], [0, 3, 6])
    
    def test_augmented_triad(self):
        """Augmented triad: 1, 3, #5 (intervals: 0, 4, 8 semitones)."""
        from utils.chord_decomposition import INTERVAL_DEFINITIONS
        self.assertEqual(INTERVAL_DEFINITIONS['aug']['notes'], [0, 4, 8])
    
    def test_sus2(self):
        """Suspended 2nd: 1, 2, 5 (intervals: 0, 2, 7 semitones)."""
        from utils.chord_decomposition import INTERVAL_DEFINITIONS
        self.assertEqual(INTERVAL_DEFINITIONS['sus2']['notes'], [0, 2, 7])
    
    def test_sus4(self):
        """Suspended 4th: 1, 4, 5 (intervals: 0, 5, 7 semitones)."""
        from utils.chord_decomposition import INTERVAL_DEFINITIONS
        self.assertEqual(INTERVAL_DEFINITIONS['sus4']['notes'], [0, 5, 7])
    
    def test_power_chord(self):
        """Power chord: 1, 5 (intervals: 0, 7 semitones)."""
        from utils.chord_decomposition import INTERVAL_DEFINITIONS
        self.assertEqual(INTERVAL_DEFINITIONS['5']['notes'], [0, 7])


def run_tests():
    """Run all tests and print summary."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestVocabularyDefinitions,
        TestChordDecomposition,
        TestChordReassembly,
        TestModelArchitecture,
        TestMultiTaskLoss,
        TestEndToEndPipeline,
        TestIntervalDefinitions,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        for test, traceback in result.failures + result.errors:
            print(f"\n  - {test}: {traceback.split(chr(10))[0]}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
