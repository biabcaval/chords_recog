# encoding: utf-8
"""
Exhaustive round-trip tests for chord decomposition and reassembly.

Validates that decompose -> reassemble produces musically correct results
for a comprehensive set of chord labels, including all qualities, extensions,
bass notes, and shorthand notations found in standard datasets.

Run with: python tests/test_roundtrip_exhaustive.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from utils.chord_decomposition import (
    ChordDecomposer, ChordReassembler, COMPONENT_NAMES, CHORD_VOCAB
)


class TestRoundTripExhaustive(unittest.TestCase):
    """Exhaustive round-trip: decompose -> reassemble for all common chord types."""

    def setUp(self):
        self.decomposer = ChordDecomposer()
        self.reassembler = ChordReassembler()

    def _roundtrip(self, chord_in, expected_out=None):
        """Decompose and reassemble, asserting the expected canonical output."""
        if expected_out is None:
            expected_out = chord_in
        components = self.decomposer.decompose(chord_in)
        result = self.reassembler.reassemble(components)
        self.assertEqual(result, expected_out,
                         f"Round-trip: {chord_in!r} -> components={components} -> {result!r}, expected {expected_out!r}")

    # --- Basic triads ---

    def test_no_chord(self):
        self._roundtrip('N')

    def test_unknown_chord(self):
        self._roundtrip('X', 'N')

    def test_all_roots_major(self):
        for root in CHORD_VOCAB['root'][1:]:
            self._roundtrip(f'{root}:maj')

    def test_all_roots_minor(self):
        for root in CHORD_VOCAB['root'][1:]:
            self._roundtrip(f'{root}:min')

    def test_triads(self):
        triads = ['maj', 'min', 'dim', 'aug', 'sus2', 'sus4']
        for triad in triads:
            self._roundtrip(f'C:{triad}')

    # --- Power chords ---

    def test_power_chord(self):
        self._roundtrip('C:5')
        self._roundtrip('E:5')

    def test_power_chord_with_bass(self):
        self._roundtrip('C:5/G')

    # --- 6th chords ---

    def test_maj6(self):
        self._roundtrip('C:maj6')

    def test_min6(self):
        self._roundtrip('A:min6')

    # --- 7th chords ---

    def test_dom7(self):
        self._roundtrip('C:7')

    def test_maj7(self):
        self._roundtrip('C:maj7')

    def test_min7(self):
        self._roundtrip('E:min7')

    def test_dim7(self):
        self._roundtrip('F:dim7')

    def test_hdim7(self):
        self._roundtrip('B:hdim7')

    def test_minmaj7(self):
        self._roundtrip('C:minmaj7')

    def test_aug7(self):
        """aug + b7 -> aug7"""
        self._roundtrip('C:aug7')

    # --- 9th chords (shorthand with implied tones) ---

    def test_dom9_shorthand(self):
        self._roundtrip('C:9', 'C:7(9)')

    def test_maj9_shorthand(self):
        self._roundtrip('C:maj9', 'C:maj7(9)')

    def test_min9_shorthand(self):
        self._roundtrip('D:min9', 'D:min7(9)')

    def test_9_add_parenthetical(self):
        """Parenthetical (9) means 'add' — no implied 7th."""
        self._roundtrip('C:maj(9)', 'C:maj(9)')

    def test_min_add9_parenthetical(self):
        self._roundtrip('C:min(9)', 'C:min(9)')

    # --- Altered 9ths ---

    def test_7_b9_parenthetical(self):
        self._roundtrip('C:7(b9)', 'C:7(b9)')

    def test_7_sharp9_parenthetical(self):
        self._roundtrip('C:7(#9)', 'C:7(#9)')

    def test_maj7_b9_parenthetical(self):
        self._roundtrip('C:maj7(b9)', 'C:maj7(b9)')

    # --- 11th chords ---

    def test_dom11_shorthand(self):
        self._roundtrip('C:11', 'C:7(9)(11)')

    def test_min11_shorthand(self):
        self._roundtrip('C:min11', 'C:min7(9)(11)')

    def test_sharp11_parenthetical(self):
        self._roundtrip('C:maj7(#11)', 'C:maj7(#11)')

    # --- 13th chords ---

    def test_dom13_shorthand(self):
        self._roundtrip('C:13', 'C:7(9)(11)(13)')

    def test_maj13_shorthand(self):
        self._roundtrip('G:maj13', 'G:maj7(9)(11)(13)')

    def test_min13_shorthand(self):
        self._roundtrip('A:min13', 'A:min7(9)(11)(13)')

    def test_b13_parenthetical(self):
        self._roundtrip('C:7(b13)', 'C:7(b13)')

    # --- Slash chords (inversions) ---

    def test_slash_chord_maj(self):
        self._roundtrip('C:maj/E')

    def test_slash_chord_min7(self):
        self._roundtrip('A:min7/E')

    def test_slash_chord_sus4(self):
        self._roundtrip('F:sus4/C')

    def test_slash_chord_dom7(self):
        self._roundtrip('G:7/B')

    # --- Flat notes (enharmonic normalization) ---

    def test_flat_root_normalized(self):
        """Bb -> A# normalization."""
        components = self.decomposer.decompose('Bb:maj')
        self.assertEqual(components['root'], 'A#')
        result = self.reassembler.reassemble(components)
        self.assertEqual(result, 'A#:maj')

    def test_flat_bass_normalized(self):
        components = self.decomposer.decompose('C:maj/Bb')
        self.assertEqual(components['bass'], 'A#')
        result = self.reassembler.reassemble(components)
        self.assertEqual(result, 'C:maj/A#')

    # --- Combined extensions from .lab format ---

    def test_maj7_9_parenthetical(self):
        self._roundtrip('C:maj7(9)', 'C:maj7(9)')

    def test_7_b9_sharp11(self):
        self._roundtrip('C:7(b9)(#11)', 'C:7(b9)(#11)')

    def test_min7_b13(self):
        self._roundtrip('C:min7(b13)', 'C:min7(b13)')

    def test_sus4_b7_parenthetical(self):
        self._roundtrip('C:sus4(b7)', 'C:sus4(b7)')

    # --- Edge cases: reassembler priority rules ---

    def test_root_N_forces_N(self):
        components = {comp: 'N' for comp in COMPONENT_NAMES}
        components['triad'] = 'maj'
        components['7th'] = 'b7'
        result = self.reassembler.reassemble(components)
        self.assertEqual(result, 'N')

    def test_triad_N_forces_N(self):
        components = {comp: 'N' for comp in COMPONENT_NAMES}
        components['root'] = 'C'
        components['7th'] = 'b7'
        result = self.reassembler.reassemble(components)
        self.assertEqual(result, 'N')

    def test_power_chord_ignores_triad(self):
        components = {comp: 'N' for comp in COMPONENT_NAMES}
        components['root'] = 'C'
        components['misc'] = '5'
        components['triad'] = 'maj'
        result = self.reassembler.reassemble(components)
        self.assertEqual(result, 'C:5')

    def test_6th_with_7th_not_dropped(self):
        """6th should appear as parenthetical when combined with 7th."""
        components = {comp: 'N' for comp in COMPONENT_NAMES}
        components['root'] = 'A'
        components['triad'] = 'min'
        components['6th'] = '6'
        components['7th'] = 'b7'
        result = self.reassembler.reassemble(components)
        self.assertIn('(6)', result)

    def test_bb7_with_non_dim_triad(self):
        """bb7 with non-dim triad should use (bb7) notation."""
        components = {comp: 'N' for comp in COMPONENT_NAMES}
        components['root'] = 'C'
        components['triad'] = 'maj'
        components['7th'] = 'bb7'
        result = self.reassembler.reassemble(components)
        self.assertIn('(bb7)', result)
        self.assertNotEqual(result, 'C:maj7')

    # --- Decomposer: implied tones detail ---

    def test_min9_implies_b7(self):
        components = self.decomposer.decompose('D:min9')
        self.assertEqual(components['triad'], 'min')
        self.assertEqual(components['7th'], 'b7')
        self.assertEqual(components['9th'], '9')

    def test_dom9_implies_b7(self):
        components = self.decomposer.decompose('C:9')
        self.assertEqual(components['triad'], 'maj')
        self.assertEqual(components['7th'], 'b7')
        self.assertEqual(components['9th'], '9')

    def test_maj9_implies_major7(self):
        components = self.decomposer.decompose('C:maj9')
        self.assertEqual(components['triad'], 'maj')
        self.assertEqual(components['7th'], '7')
        self.assertEqual(components['9th'], '9')

    def test_13_implies_b7_9_11(self):
        components = self.decomposer.decompose('C:13')
        self.assertEqual(components['7th'], 'b7')
        self.assertEqual(components['9th'], '9')
        self.assertEqual(components['11th'], '11')
        self.assertEqual(components['13th'], '13')

    def test_parenthetical_9_no_implied_7(self):
        """C:maj(9) = add 9, no implied 7th."""
        components = self.decomposer.decompose('C:maj(9)')
        self.assertEqual(components['triad'], 'maj')
        self.assertEqual(components['7th'], 'N')
        self.assertEqual(components['9th'], '9')

    def test_parenthetical_min_9_no_implied_7(self):
        components = self.decomposer.decompose('C:min(9)')
        self.assertEqual(components['triad'], 'min')
        self.assertEqual(components['7th'], 'N')
        self.assertEqual(components['9th'], '9')


if __name__ == '__main__':
    unittest.main(verbosity=2)
