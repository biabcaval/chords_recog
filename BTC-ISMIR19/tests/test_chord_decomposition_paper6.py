# encoding: utf-8
"""
Tests for the paper-faithful 6-component chord decomposition.

Covers vocabulary shape, the join/split bijection on root_triad, the 9->6
folding (notably the 6th -> bb7 fold), decompose_batch indices, and the
decompose -> reassemble roundtrip for representative chords.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import chord_decomposition_paper6 as p6
from utils.decomposition_registry import get_decomposition, normalize_scheme


def test_component_names_and_vocab_sizes():
    assert p6.COMPONENT_NAMES == ['root_triad', 'bass', '7th', '9th', '11th', '13th']
    sizes = p6.get_vocab_sizes()
    assert sizes['root_triad'] == 91  # 13 roots x 7 triads
    assert sizes['bass'] == 13
    assert sizes['7th'] == 4
    assert sizes['9th'] == 4
    assert sizes['11th'] == 3
    assert sizes['13th'] == 3
    # Vocab index maps must be bijective (unique labels).
    for comp, labels in p6.CHORD_VOCAB.items():
        assert len(labels) == len(set(labels)), f"duplicate labels in {comp}"


def test_join_split_bijection():
    for root_idx, root in enumerate(p6.ROOTS):
        for triad_idx, triad in enumerate(p6.TRIADS):
            joint = p6.join_root_triad(root, triad)
            if root == 'N':
                # All N-root combos canonicalize to the single no-chord tag 0.
                assert joint == 0
                continue
            r2, t2 = p6.split_root_triad(joint)
            assert (r2, t2) == (root, triad)


@pytest.mark.parametrize("chord,expected", [
    ('N', {'root_triad': 'N', '7th': 'N'}),
    ('C:maj', {'root_triad': 'C:maj', '7th': 'N'}),
    ('D:min', {'root_triad': 'D:min', '7th': 'N'}),
    ('C:maj7', {'root_triad': 'C:maj', '7th': '7'}),
    ('C:7', {'root_triad': 'C:maj', '7th': 'b7'}),
    ('D:min7', {'root_triad': 'D:min', '7th': 'b7'}),
    ('B:dim7', {'root_triad': 'B:dim', '7th': 'bb7'}),
])
def test_decompose_known_chords(chord, expected):
    comps = p6.ChordDecomposer().decompose(chord)
    for key, val in expected.items():
        assert comps[key] == val, f"{chord}: {key}={comps[key]} != {val}"


def test_sixth_folds_into_bb7():
    """Per the paper, the 6th is encoded as bb7 (on a maj/min triad)."""
    comps = p6.ChordDecomposer().decompose('C:maj6')
    assert comps['root_triad'] == 'C:maj'
    assert comps['7th'] == 'bb7'


def test_decompose_batch_indices_shape_and_range():
    labels = ['N', 'C:maj', 'D:min7', 'B:dim7', 'G:maj6', 'A:7']
    out = p6.ChordDecomposer().decompose_batch(labels)
    assert set(out.keys()) == set(p6.COMPONENT_NAMES)
    for comp in p6.COMPONENT_NAMES:
        arr = out[comp]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (len(labels),)
        assert arr.min() >= 0
        assert arr.max() < len(p6.CHORD_VOCAB[comp])


@pytest.mark.parametrize("chord", [
    'C:maj', 'D:min', 'C:maj7', 'C:7', 'D:min7', 'B:dim7',
    'C:maj6', 'A:min6', 'G:maj/B', 'E:min9', 'N',
])
def test_roundtrip_reassembles_equivalently(chord):
    """decompose -> indices -> reassemble must re-decompose to the same 6 comps."""
    dec = p6.ChordDecomposer()
    rea = p6.ChordReassembler()

    idx = {c: int(v[0]) for c, v in dec.decompose_batch([chord]).items()}
    rebuilt = rea.reassemble_from_indices(idx)

    # The reassembled label must decompose back to the same 6 components.
    again = {c: int(v[0]) for c, v in dec.decompose_batch([rebuilt]).items()}
    assert again == idx, f"{chord} -> {rebuilt}: {again} != {idx}"


def test_dim7_and_maj6_share_bb7_but_differ_by_triad():
    dec = p6.ChordDecomposer()
    dim7 = dec.decompose('B:dim7')
    maj6 = dec.decompose('C:maj6')
    assert dim7['7th'] == maj6['7th'] == 'bb7'
    assert dim7['root_triad'].endswith(':dim')
    assert maj6['root_triad'].endswith(':maj')

    rea = p6.ChordReassembler()
    # dim + bb7 -> dim7; maj + bb7 -> a sixth chord
    assert rea.reassemble(dim7) == 'B:dim7'
    assert rea.reassemble(maj6) == 'C:maj6'


def test_registry_selects_schemes():
    assert normalize_scheme('paper6') == 'paper6'
    assert normalize_scheme('6') == 'paper6'
    assert normalize_scheme('full9') == 'full9'
    assert normalize_scheme(None) == 'full9'

    six = get_decomposition('paper6')
    nine = get_decomposition('full9')
    assert len(six.COMPONENT_NAMES) == 6
    assert len(nine.COMPONENT_NAMES) == 9

    with pytest.raises(ValueError):
        normalize_scheme('bogus')
