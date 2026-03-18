#!/usr/bin/env python
# encoding: utf-8
"""
Unit tests for chord structure decomposition implementation.

Tests cover:
- Chord decomposition and reassembly
- Data loading with decomposition
- Model architecture
- Loss computation
- Inference pipeline
"""

import unittest
import numpy as np
import torch
from pathlib import Path

from utils.chord_decomposition import (
    ChordDecomposer, ChordReassembler, get_vocab_sizes, COMPONENT_NAMES,
    transpose_chord,
)
from models.btc_model_decomposed import (
    ComponentHead, MultiHeadChordDecomposer, MultiTaskLoss, BTC_model_decomposed
)


class TestChordDecomposition(unittest.TestCase):
    """Test chord decomposition and reassembly."""
    
    def setUp(self):
        self.decomposer = ChordDecomposer()
        self.reassembler = ChordReassembler()
    
    def test_basic_decomposition(self):
        """Test basic chord decomposition."""
        chord = 'C:maj'
        components = self.decomposer.decompose(chord)
        
        self.assertEqual(components['root'], 'C')
        self.assertEqual(components['triad'], 'maj')
        self.assertEqual(components['bass'], 'N')
        self.assertEqual(components['7th'], 'N')
    
    def test_complex_decomposition(self):
        """Test decomposition of complex chord — shorthand implies major 7th."""
        chord = 'D:maj9'
        components = self.decomposer.decompose(chord)
        
        self.assertEqual(components['root'], 'D')
        self.assertEqual(components['triad'], 'maj')
        self.assertEqual(components['7th'], '7')
        self.assertEqual(components['9th'], '9')
    
    def test_slash_chord_decomposition(self):
        """Test decomposition of slash chords."""
        chord = 'C:maj/E'
        components = self.decomposer.decompose(chord)
        
        self.assertEqual(components['root'], 'C')
        self.assertEqual(components['bass'], 'E')
        self.assertEqual(components['triad'], 'maj')
    
    def test_no_chord(self):
        """Test decomposition of no-chord."""
        chord = 'N'
        components = self.decomposer.decompose(chord)
        
        for component in COMPONENT_NAMES:
            self.assertEqual(components[component], 'N')
    
    def test_decompose_batch(self):
        """Test batch decomposition."""
        chords = ['C:maj', 'D:min7', 'E:aug', 'N']
        components_batch = self.decomposer.decompose_batch(chords)
        
        # Check structure
        self.assertEqual(len(components_batch), len(COMPONENT_NAMES))
        for component, indices in components_batch.items():
            self.assertEqual(len(indices), len(chords))
            self.assertIsInstance(indices, np.ndarray)
    
    def test_reassembly(self):
        """Test chord reassembly from components."""
        chord = 'C:maj7'
        components = self.decomposer.decompose(chord)
        reassembled = self.reassembler.reassemble(components)
        
        # Reassemble should recover original chord
        self.assertEqual(reassembled, chord)
    
    def test_reassembly_with_priority(self):
        """Test priority logic in reassembly."""
        # If triad is N, chord should be N
        components = {comp: 'N' for comp in COMPONENT_NAMES}
        components['root'] = 'C'  # Even with root set
        reassembled = self.reassembler.reassemble(components)
        
        self.assertEqual(reassembled, 'N')
    
    def test_round_trip(self):
        """Test decompose -> reassemble round trip.
        
        Shorthand notations expand to canonical form with implied tones.
        """
        test_cases = [
            ('C:maj', 'C:maj'),
            ('D:min9', 'D:min7(9)'),
            ('E:dim7', 'E:dim7'),
            ('F:sus4/C', 'F:sus4/C'),
            ('G:maj13', 'G:maj7(9)(11)(13)'),
            ('C:5', 'C:5'),
        ]
        for chord, expected in test_cases:
            components = self.decomposer.decompose(chord)
            reassembled = self.reassembler.reassemble(components)
            self.assertEqual(reassembled, expected, f"Round trip failed for {chord}")


class TestBassScaleDegree(unittest.TestCase):
    """Bug 1: bass scale-degree notation (/5, /b3) in .lab files."""

    def setUp(self):
        self.decomposer = ChordDecomposer()

    def test_bass_degree_5(self):
        c = self.decomposer.decompose('C:maj/5')
        self.assertEqual(c['root'], 'C')
        self.assertEqual(c['bass'], 'G')
        self.assertEqual(c['triad'], 'maj')

    def test_bass_degree_3(self):
        c = self.decomposer.decompose('D:maj/3')
        self.assertEqual(c['bass'], 'F#')

    def test_bass_degree_b3(self):
        c = self.decomposer.decompose('A:min/b3')
        self.assertEqual(c['bass'], 'C')

    def test_bass_degree_b7(self):
        c = self.decomposer.decompose('C:7/b7')
        self.assertEqual(c['bass'], 'A#')

    def test_bass_degree_7(self):
        c = self.decomposer.decompose('C:maj7/7')
        self.assertEqual(c['bass'], 'B')

    def test_bass_degree_b5(self):
        c = self.decomposer.decompose('G:dim/b5')
        self.assertEqual(c['bass'], 'C#')

    def test_bass_note_name_still_works(self):
        """Absolute note names should still be parsed correctly."""
        c = self.decomposer.decompose('C:maj/E')
        self.assertEqual(c['bass'], 'E')

    def test_bass_degree_with_extensions(self):
        c = self.decomposer.decompose('D:maj6(9)/5')
        self.assertEqual(c['root'], 'D')
        self.assertEqual(c['bass'], 'A')
        self.assertEqual(c['6th'], '6')
        self.assertEqual(c['9th'], '9')


class TestParenVsShorthand(unittest.TestCase):
    """Bug 2: parenthetical (7) = major 7th vs shorthand 7 = dominant."""

    def setUp(self):
        self.decomposer = ChordDecomposer()

    def test_min_paren_7_is_major_7th(self):
        c = self.decomposer.decompose('C:min(7)')
        self.assertEqual(c['triad'], 'min')
        self.assertEqual(c['7th'], '7')

    def test_min7_shorthand_is_dominant_7th(self):
        c = self.decomposer.decompose('C:min7')
        self.assertEqual(c['triad'], 'min')
        self.assertEqual(c['7th'], 'b7')

    def test_aug_paren_7(self):
        c = self.decomposer.decompose('C:aug(7)')
        self.assertEqual(c['triad'], 'aug')
        self.assertEqual(c['7th'], '7')

    def test_aug7_shorthand(self):
        c = self.decomposer.decompose('C:aug7')
        self.assertEqual(c['triad'], 'aug')
        self.assertEqual(c['7th'], 'b7')

    def test_sus4_paren_7(self):
        c = self.decomposer.decompose('C:sus4(7)')
        self.assertEqual(c['triad'], 'sus4')
        self.assertEqual(c['7th'], '7')

    def test_sus4_paren_b7(self):
        c = self.decomposer.decompose('C:sus4(b7)')
        self.assertEqual(c['triad'], 'sus4')
        self.assertEqual(c['7th'], 'b7')

    def test_sus2_paren_7(self):
        c = self.decomposer.decompose('C:sus2(7)')
        self.assertEqual(c['triad'], 'sus2')
        self.assertEqual(c['7th'], '7')

    def test_maj6_paren_7(self):
        c = self.decomposer.decompose('C:maj6(7)')
        self.assertEqual(c['triad'], 'maj')
        self.assertEqual(c['6th'], '6')
        self.assertEqual(c['7th'], '7')

    def test_min6_paren_7(self):
        c = self.decomposer.decompose('C:min6(7)')
        self.assertEqual(c['triad'], 'min')
        self.assertEqual(c['6th'], '6')
        self.assertEqual(c['7th'], '7')

    def test_min_paren_9_is_add9(self):
        """min(9) = add 9 (no implied 7th), vs min9 shorthand."""
        c = self.decomposer.decompose('C:min(9)')
        self.assertEqual(c['triad'], 'min')
        self.assertEqual(c['9th'], '9')
        self.assertEqual(c['7th'], 'N')

    def test_min9_shorthand_implies_b7(self):
        c = self.decomposer.decompose('C:min9')
        self.assertEqual(c['triad'], 'min')
        self.assertEqual(c['9th'], '9')
        self.assertEqual(c['7th'], 'b7')


class TestPowerChordExtensions(unittest.TestCase):
    """Bug 3: 5(...) power chord with extensions."""

    def setUp(self):
        self.decomposer = ChordDecomposer()
        self.reassembler = ChordReassembler()

    def test_plain_5(self):
        c = self.decomposer.decompose('C:5')
        self.assertEqual(c['misc'], '5')
        self.assertEqual(c['triad'], 'N')

    def test_5_b7(self):
        c = self.decomposer.decompose('C:5(b7)')
        self.assertEqual(c['misc'], '5')
        self.assertEqual(c['7th'], 'b7')
        self.assertEqual(c['triad'], 'N')

    def test_5_9(self):
        c = self.decomposer.decompose('C:5(9)')
        self.assertEqual(c['misc'], '5')
        self.assertEqual(c['9th'], '9')

    def test_5_b13(self):
        c = self.decomposer.decompose('C:5(b13)')
        self.assertEqual(c['misc'], '5')
        self.assertEqual(c['13th'], 'b13')

    def test_5_b7_b9(self):
        c = self.decomposer.decompose('C:5(b7,b9)')
        self.assertEqual(c['misc'], '5')
        self.assertEqual(c['7th'], 'b7')
        self.assertEqual(c['9th'], 'b9')

    def test_pedal(self):
        c = self.decomposer.decompose('C:pedal')
        self.assertEqual(c['misc'], '5')

    def test_quality_1(self):
        c = self.decomposer.decompose('C:1')
        self.assertEqual(c['misc'], '5')

    def test_reassemble_power_with_ext(self):
        c = self.decomposer.decompose('C:5(b7)')
        reassembled = self.reassembler.reassemble(c)
        self.assertEqual(reassembled, 'C:5(b7)')

    def test_reassemble_power_with_multi_ext(self):
        c = self.decomposer.decompose('C:5(b7,b9)')
        reassembled = self.reassembler.reassemble(c)
        self.assertEqual(reassembled, 'C:5(b7)(b9)')


class TestMin7b5AsHdim(unittest.TestCase):
    """Bug 4: min7(b5) should decompose the same as hdim7."""

    def setUp(self):
        self.decomposer = ChordDecomposer()

    def test_min7_b5_equals_hdim7(self):
        hdim = self.decomposer.decompose('C:hdim7')
        minb5 = self.decomposer.decompose('C:min7(b5)')
        self.assertEqual(hdim['triad'], 'dim')
        self.assertEqual(hdim['7th'], 'b7')
        self.assertEqual(minb5['triad'], hdim['triad'])
        self.assertEqual(minb5['7th'], hdim['7th'])

    def test_min_b5_no_7th(self):
        """min(b5) without 7th → just dim triad."""
        c = self.decomposer.decompose('C:min(b5)')
        self.assertEqual(c['triad'], 'dim')
        self.assertEqual(c['7th'], 'N')

    def test_maj_sharp5_becomes_aug(self):
        c = self.decomposer.decompose('C:maj(#5)')
        self.assertEqual(c['triad'], 'aug')


class TestOmitNotation(unittest.TestCase):
    """Bug 5: * (omit) notation."""

    def setUp(self):
        self.decomposer = ChordDecomposer()

    def test_maj_omit_3(self):
        c = self.decomposer.decompose('C:maj(*3)')
        self.assertEqual(c['triad'], 'N')
        self.assertEqual(c['misc'], '5')

    def test_min_omit_b3(self):
        c = self.decomposer.decompose('A:min(*b3)')
        self.assertEqual(c['triad'], 'N')
        self.assertEqual(c['misc'], '5')

    def test_7_omit_3(self):
        c = self.decomposer.decompose('C:7(*3)')
        self.assertEqual(c['triad'], 'N')
        self.assertEqual(c['misc'], '5')
        self.assertEqual(c['7th'], 'b7')

    def test_7_omit_5_with_9(self):
        """*5 without *3 preserves triad identity."""
        c = self.decomposer.decompose('C:7(*5,9)')
        self.assertEqual(c['triad'], 'maj')
        self.assertEqual(c['7th'], 'b7')
        self.assertEqual(c['9th'], '9')

    def test_maj_omit_3_and_5(self):
        c = self.decomposer.decompose('C:maj(*3,*5)')
        self.assertEqual(c['triad'], 'N')
        self.assertEqual(c['misc'], '5')

    def test_min7_omit_b3_and_5(self):
        c = self.decomposer.decompose('C:min7(*b3,*5)')
        self.assertEqual(c['triad'], 'N')
        self.assertEqual(c['misc'], '5')
        self.assertEqual(c['7th'], 'b7')

    def test_maj7_omit_3(self):
        c = self.decomposer.decompose('C:maj7(*3)')
        self.assertEqual(c['triad'], 'N')
        self.assertEqual(c['misc'], '5')
        self.assertEqual(c['7th'], '7')


class TestComprehensiveShorthands(unittest.TestCase):
    """Verify all standard shorthands still decompose correctly."""

    def setUp(self):
        self.d = ChordDecomposer()

    def test_maj(self):
        c = self.d.decompose('C:maj')
        self.assertEqual(c['triad'], 'maj')

    def test_min(self):
        c = self.d.decompose('C:min')
        self.assertEqual(c['triad'], 'min')

    def test_dim(self):
        c = self.d.decompose('C:dim')
        self.assertEqual(c['triad'], 'dim')

    def test_aug(self):
        c = self.d.decompose('C:aug')
        self.assertEqual(c['triad'], 'aug')

    def test_sus2(self):
        c = self.d.decompose('C:sus2')
        self.assertEqual(c['triad'], 'sus2')

    def test_sus4(self):
        c = self.d.decompose('C:sus4')
        self.assertEqual(c['triad'], 'sus4')

    def test_maj7(self):
        c = self.d.decompose('C:maj7')
        self.assertEqual(c['triad'], 'maj')
        self.assertEqual(c['7th'], '7')

    def test_min7(self):
        c = self.d.decompose('C:min7')
        self.assertEqual(c['triad'], 'min')
        self.assertEqual(c['7th'], 'b7')

    def test_dom7(self):
        c = self.d.decompose('C:7')
        self.assertEqual(c['triad'], 'maj')
        self.assertEqual(c['7th'], 'b7')

    def test_dim7(self):
        c = self.d.decompose('C:dim7')
        self.assertEqual(c['triad'], 'dim')
        self.assertEqual(c['7th'], 'bb7')

    def test_hdim7(self):
        c = self.d.decompose('C:hdim7')
        self.assertEqual(c['triad'], 'dim')
        self.assertEqual(c['7th'], 'b7')

    def test_minmaj7(self):
        c = self.d.decompose('C:minmaj7')
        self.assertEqual(c['triad'], 'min')
        self.assertEqual(c['7th'], '7')

    def test_maj6(self):
        c = self.d.decompose('C:maj6')
        self.assertEqual(c['triad'], 'maj')
        self.assertEqual(c['6th'], '6')

    def test_min6(self):
        c = self.d.decompose('C:min6')
        self.assertEqual(c['triad'], 'min')
        self.assertEqual(c['6th'], '6')

    def test_dom9(self):
        c = self.d.decompose('C:9')
        self.assertEqual(c['triad'], 'maj')
        self.assertEqual(c['7th'], 'b7')
        self.assertEqual(c['9th'], '9')

    def test_maj9(self):
        c = self.d.decompose('C:maj9')
        self.assertEqual(c['triad'], 'maj')
        self.assertEqual(c['7th'], '7')
        self.assertEqual(c['9th'], '9')

    def test_min9(self):
        c = self.d.decompose('C:min9')
        self.assertEqual(c['triad'], 'min')
        self.assertEqual(c['7th'], 'b7')
        self.assertEqual(c['9th'], '9')

    def test_7sus4(self):
        c = self.d.decompose('C:7sus4')
        self.assertEqual(c['triad'], 'sus4')
        self.assertEqual(c['7th'], 'b7')

    def test_aug7(self):
        c = self.d.decompose('C:aug7')
        self.assertEqual(c['triad'], 'aug')
        self.assertEqual(c['7th'], 'b7')


class TestCombinedParenExtensions(unittest.TestCase):
    """Mixed shorthand + parenthetical extensions from real .lab files."""

    def setUp(self):
        self.d = ChordDecomposer()

    def test_maj7_9(self):
        c = self.d.decompose('C:maj7(9)')
        self.assertEqual(c['triad'], 'maj')
        self.assertEqual(c['7th'], '7')
        self.assertEqual(c['9th'], '9')

    def test_7_b9(self):
        c = self.d.decompose('C:7(b9)')
        self.assertEqual(c['triad'], 'maj')
        self.assertEqual(c['7th'], 'b7')
        self.assertEqual(c['9th'], 'b9')

    def test_7_sharp9_b13(self):
        c = self.d.decompose('C:7(#9,b13)')
        self.assertEqual(c['triad'], 'maj')
        self.assertEqual(c['7th'], 'b7')
        self.assertEqual(c['9th'], '#9')
        self.assertEqual(c['13th'], 'b13')

    def test_min7_9_11(self):
        c = self.d.decompose('C:min7(9,11)')
        self.assertEqual(c['triad'], 'min')
        self.assertEqual(c['7th'], 'b7')
        self.assertEqual(c['9th'], '9')
        self.assertEqual(c['11th'], '11')

    def test_7sus4_13(self):
        c = self.d.decompose('C:7sus4(13)')
        self.assertEqual(c['triad'], 'sus4')
        self.assertEqual(c['7th'], 'b7')
        self.assertEqual(c['13th'], '13')

    def test_dim7_b13(self):
        c = self.d.decompose('C:dim7(b13)')
        self.assertEqual(c['triad'], 'dim')
        self.assertEqual(c['7th'], 'bb7')
        self.assertEqual(c['13th'], 'b13')

    def test_hdim7_b9(self):
        c = self.d.decompose('C:hdim7(b9)')
        self.assertEqual(c['triad'], 'dim')
        self.assertEqual(c['7th'], 'b7')
        self.assertEqual(c['9th'], 'b9')

    def test_minmaj7_9(self):
        c = self.d.decompose('C:minmaj7(9)')
        self.assertEqual(c['triad'], 'min')
        self.assertEqual(c['7th'], '7')
        self.assertEqual(c['9th'], '9')

    def test_aug7_b9(self):
        c = self.d.decompose('C:aug7(b9)')
        self.assertEqual(c['triad'], 'aug')
        self.assertEqual(c['7th'], 'b7')
        self.assertEqual(c['9th'], 'b9')

    def test_maj6_9(self):
        c = self.d.decompose('C:maj6(9)')
        self.assertEqual(c['triad'], 'maj')
        self.assertEqual(c['6th'], '6')
        self.assertEqual(c['9th'], '9')

    def test_min6_b7(self):
        c = self.d.decompose('C:min6(b7)')
        self.assertEqual(c['triad'], 'min')
        self.assertEqual(c['6th'], '6')
        self.assertEqual(c['7th'], 'b7')

    def test_sus4_b7_9(self):
        c = self.d.decompose('C:sus4(b7,9)')
        self.assertEqual(c['triad'], 'sus4')
        self.assertEqual(c['7th'], 'b7')
        self.assertEqual(c['9th'], '9')

    def test_maj7_9_sharp11(self):
        c = self.d.decompose('C:maj7(9,#11)')
        self.assertEqual(c['triad'], 'maj')
        self.assertEqual(c['7th'], '7')
        self.assertEqual(c['9th'], '9')
        self.assertEqual(c['11th'], '#11')


class TestTransposeChord(unittest.TestCase):
    """Test chord transposition utility."""

    def test_no_shift(self):
        self.assertEqual(transpose_chord('C:maj', 0), 'C:maj')

    def test_special_labels(self):
        self.assertEqual(transpose_chord('N', 5), 'N')
        self.assertEqual(transpose_chord('X', 3), 'X')
        self.assertEqual(transpose_chord('', 2), '')

    def test_shift_up(self):
        self.assertEqual(transpose_chord('A:min', 3), 'C:min')

    def test_shift_down(self):
        self.assertEqual(transpose_chord('D:maj7', -2), 'C:maj7')

    def test_wrap_around(self):
        self.assertEqual(transpose_chord('B:min', 1), 'C:min')
        self.assertEqual(transpose_chord('C:maj', -1), 'B:maj')

    def test_with_bass(self):
        self.assertEqual(transpose_chord('C:maj7(9)/E', 2), 'D:maj7(9)/F#')

    def test_flat_root(self):
        self.assertEqual(transpose_chord('Bb:7', 2), 'C:7')
        self.assertEqual(transpose_chord('Eb:min', 1), 'E:min')

    def test_flat_bass(self):
        self.assertEqual(transpose_chord('C:maj/Bb', 2), 'D:maj/C')

    def test_extensions_preserved(self):
        self.assertEqual(transpose_chord('A:min7(9)(13)', 3), 'C:min7(9)(13)')

    def test_plain_root(self):
        self.assertEqual(transpose_chord('G', 2), 'A')

    def test_sharp_root(self):
        self.assertEqual(transpose_chord('F#:dim7', 1), 'G:dim7')

    def test_all_12_shifts(self):
        """Shifting by 12 semitones should return to the same chord."""
        chord = 'C#:maj7(9)/G#'
        self.assertEqual(transpose_chord(chord, 12), chord)

    def test_bass_numeric(self):
        """Bass note like /5 (scale degree) should pass through unchanged."""
        self.assertEqual(transpose_chord('D:maj6(9)/5', 3), 'F:maj6(9)/5')


class TestVocabulary(unittest.TestCase):
    """Test vocabulary definitions."""
    
    def test_vocab_sizes(self):
        """Test vocabulary sizes."""
        vocab_sizes = get_vocab_sizes()
        
        self.assertEqual(vocab_sizes['root'], 13)
        self.assertEqual(vocab_sizes['bass'], 13)
        self.assertEqual(vocab_sizes['triad'], 7)
        self.assertEqual(vocab_sizes['misc'], 2)
        self.assertEqual(vocab_sizes['7th'], 4)
        self.assertEqual(vocab_sizes['9th'], 4)
        self.assertEqual(vocab_sizes['11th'], 3)
        self.assertEqual(vocab_sizes['13th'], 3)
    
    def test_total_classes(self):
        """Test total number of classes."""
        vocab_sizes = get_vocab_sizes()
        total = sum(vocab_sizes.values())
        
        # 51 total classes vs. ~170 for monolithic
        self.assertEqual(total, 51)


class TestModelArchitecture(unittest.TestCase):
    """Test model architecture."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.hidden_size = 256
        self.batch_size = 4
        self.seq_len = 100
    
    def test_component_head(self):
        """Test single component head."""
        vocab_size = 13
        head = ComponentHead(self.hidden_size, vocab_size)
        head = head.to(self.device)
        
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        logits = head(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, vocab_size))
    
    def test_multi_head_decomposer(self):
        """Test multi-head chord decomposer."""
        decomposer = MultiHeadChordDecomposer(self.hidden_size)
        decomposer = decomposer.to(self.device)
        
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        logits = decomposer(x)
        
        # Check logits structure
        self.assertEqual(len(logits), len(COMPONENT_NAMES))
        for component in COMPONENT_NAMES:
            self.assertIn(component, logits)
            logits_c = logits[component]
            self.assertEqual(logits_c.shape[0], self.batch_size)
            self.assertEqual(logits_c.shape[1], self.seq_len)
    
    def test_get_predictions(self):
        """Test getting predictions from logits."""
        decomposer = MultiHeadChordDecomposer(self.hidden_size)
        
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        logits = decomposer(x)
        predictions = decomposer.get_predictions(logits)
        
        for component in COMPONENT_NAMES:
            pred = predictions[component]
            # Check predictions are class indices
            self.assertTrue(torch.all(pred >= 0))
            self.assertTrue(torch.all(pred < decomposer.vocab_sizes[component]))


class TestMultiTaskLoss(unittest.TestCase):
    """Test multi-task loss computation."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.vocab_sizes = get_vocab_sizes()
        self.batch_size = 4
        self.seq_len = 100
    
    def test_loss_initialization(self):
        """Test loss function initialization."""
        loss_fn = MultiTaskLoss(self.vocab_sizes)
        
        self.assertEqual(len(loss_fn.losses), len(COMPONENT_NAMES))
        for component in COMPONENT_NAMES:
            self.assertIn(component, loss_fn.losses)
    
    def test_loss_computation(self):
        """Test loss computation."""
        loss_fn = MultiTaskLoss(self.vocab_sizes)
        loss_fn = loss_fn.to(self.device)
        
        # Create dummy logits and labels
        logits = {}
        labels = {}
        for component in COMPONENT_NAMES:
            vocab_size = self.vocab_sizes[component]
            logits[component] = torch.randn(
                self.batch_size, self.seq_len, vocab_size,
                device=self.device
            )
            labels[component] = torch.randint(
                0, vocab_size, (self.batch_size, self.seq_len),
                device=self.device
            )
        
        # Compute loss
        loss = loss_fn(logits, labels)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())  # Scalar
        self.assertGreater(loss.item(), 0)
    
    def test_class_weights(self):
        """Test class weight computation."""
        # Create dummy dataset with components
        class DummyDataset:
            def __init__(self, n_samples=100):
                self.n_samples = n_samples
            
            def __len__(self):
                return self.n_samples
            
            def __getitem__(self, idx):
                components = {}
                for component in COMPONENT_NAMES:
                    vocab_size = get_vocab_sizes()[component]
                    components[component] = np.random.randint(0, vocab_size, 100)
                return {'components': components}
        
        dataset = DummyDataset()
        class_weights = MultiTaskLoss.compute_class_weights(
            dataset, gamma=0.5, w_max=10.0, device=self.device
        )
        
        # Check structure
        self.assertEqual(len(class_weights), len(COMPONENT_NAMES))
        for component, weights in class_weights.items():
            self.assertIsInstance(weights, torch.Tensor)
            self.assertEqual(len(weights), self.vocab_sizes[component])
            self.assertTrue(torch.all(weights > 0))
            self.assertTrue(torch.all(weights <= 10.0))


class TestInferenceUtils(unittest.TestCase):
    """Test inference utilities."""
    
    def setUp(self):
        self.decomposer = ChordDecomposer()
    
    def test_decode_predictions(self):
        """Test decoding predictions to chord labels."""
        # This would require more complex setup with actual model
        # For now, just test the decomposer works
        chords = ['C:maj', 'D:min', 'E:aug']
        components_batch = self.decomposer.decompose_batch(chords)
        
        self.assertIsNotNone(components_batch)
        self.assertEqual(len(components_batch), len(COMPONENT_NAMES))


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def setUp(self):
        self.device = torch.device('cpu')
    
    def test_full_pipeline(self):
        """Test full decomposition pipeline."""
        # 1. Decompose chords
        decomposer = ChordDecomposer()
        test_chords = ['C:maj7', 'D:min9/F#', 'E:aug']
        components_batch = decomposer.decompose_batch(test_chords)
        
        # 2. Create model with appropriate vocab sizes
        config = {
            'feature_size': 192,
            'hidden_size': 256,
            'num_layers': 4,
            'num_heads': 8,
            'total_key_depth': 256,
            'total_value_depth': 256,
            'filter_size': 1024,
            'timestep': 100,
            'input_dropout': 0.1,
            'layer_dropout': 0.1,
            'attention_dropout': 0.1,
            'relu_dropout': 0.1,
            'probs_out': False,
            'use_decomposition': True,
            'class_weight_gamma': 0.5,
            'class_weight_max': 10.0,
        }
        
        # Create loss function
        vocab_sizes = get_vocab_sizes()
        loss_fn = MultiTaskLoss(vocab_sizes)
        
        # Create dummy logits and labels
        batch_size = 2
        seq_len = 100
        
        logits = {}
        labels = {}
        for component in COMPONENT_NAMES:
            vocab_size = vocab_sizes[component]
            logits[component] = torch.randn(batch_size, seq_len, vocab_size)
            labels[component] = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Compute loss
        loss = loss_fn(logits, labels)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)


if __name__ == '__main__':
    unittest.main()
