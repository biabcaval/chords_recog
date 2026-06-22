# encoding: utf-8
"""
Paper-faithful 6-component chord decomposition (ChordFormer, 2502.11840v1).

The ChordFormer paper (Sec. III-A / III-D) represents a chord with a
*six*-dimensional vector, one softmax head per dimension:

    1. root + triad (JOINT)   -> e.g. "C#:aug", "D:min"
    2. bass                    -> {N, C, ..., B}
    3. 7th                     -> {N, 7, b7, bb7}   (bb7 also denotes the 6th)
    4. 9th                     -> {N, 9, #9, b9}
    5. 11th                    -> {N, 11, #11}
    6. 13th                    -> {N, 13, b13}

This differs from the project's default 9-component decomposition
(:mod:`utils.chord_decomposition`), which keeps ``root`` and ``triad`` as
separate heads and adds ``misc`` (power chord) and ``6th`` heads.

Rather than re-implement the (battle-tested) Harte parser, this module is a
THIN ADAPTER: it reuses the 9-component :class:`ChordDecomposer` /
:class:`ChordReassembler` and maps 9<->6 losslessly for everything the paper
models:

    * root_triad = JOIN(root, triad)         (cartesian, 13 x 7 = 91 tags)
    * the 6th    is folded into 7th = ``bb7`` (per the paper's footnote that
      "the double flat seventh (bb7) denotes the sixth"), distinguished from a
      diminished 7th by the triad (maj/min -> sixth, dim -> dim7).
    * power chords (``misc='5'``) have no head in the paper; the root is still
      preserved via the joint tag ``"<root>:N"`` (triad N).

The joint ``root_triad`` uses a cartesian index ``root_idx * 7 + triad_idx``
to stay consistent with the existing joint convention in
:mod:`models.harmonic_crf` (91 tags).

The public surface mirrors :mod:`utils.chord_decomposition` so it can be used
as a drop-in by the BEATs pipeline: ``COMPONENT_NAMES``, ``CHORD_VOCAB``,
``CHORD_VOCAB_IDX``, ``ChordDecomposer``, ``ChordReassembler``,
``get_vocab_sizes``.
"""

from typing import Dict, List

import numpy as np

from utils.chord_decomposition import (
    ChordDecomposer as _ChordDecomposer9,
    ChordReassembler as _ChordReassembler9,
)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

ROOTS = ['N', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
TRIADS = ['N', 'maj', 'min', 'dim', 'aug', 'sus2', 'sus4']

NUM_ROOTS = len(ROOTS)    # 13
NUM_TRIADS = len(TRIADS)  # 7


def _build_root_triad_vocab() -> List[str]:
    """Cartesian root x triad vocabulary (length 91).

    Index layout: ``root_idx * NUM_TRIADS + triad_idx``. The single no-chord
    label lives at index 0 (root='N', triad='N'). The other root='N' rows are
    musically impossible (never emitted by :meth:`ChordDecomposer.decompose`)
    but are kept as unique placeholders so ``CHORD_VOCAB_IDX`` stays bijective.
    """
    vocab: List[str] = []
    for r in ROOTS:
        for t in TRIADS:
            if r == 'N' and t == 'N':
                vocab.append('N')
            elif r == 'N':
                vocab.append(f'N:{t}')
            else:
                vocab.append(f'{r}:{t}')
    return vocab


ROOT_TRIAD_VOCAB = _build_root_triad_vocab()

CHORD_VOCAB: Dict[str, List[str]] = {
    'root_triad': ROOT_TRIAD_VOCAB,
    'bass': ['N', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
    '7th': ['N', '7', 'b7', 'bb7'],   # bb7 = diminished 7th OR (folded) the 6th
    '9th': ['N', '9', '#9', 'b9'],
    '11th': ['N', '11', '#11'],
    '13th': ['N', '13', 'b13'],
}

COMPONENT_NAMES = ['root_triad', 'bass', '7th', '9th', '11th', '13th']
NUM_COMPONENTS = len(COMPONENT_NAMES)

CHORD_VOCAB_IDX = {
    component: {label: idx for idx, label in enumerate(labels)}
    for component, labels in CHORD_VOCAB.items()
}

_ROOT_TO_IDX = {note: idx for idx, note in enumerate(ROOTS)}
_TRIAD_TO_IDX = {triad: idx for idx, triad in enumerate(TRIADS)}


def join_root_triad(root: str, triad: str) -> int:
    """Map a (root, triad) string pair to a joint index in ``[0, 91)``."""
    root_idx = _ROOT_TO_IDX.get(root, 0)
    triad_idx = _TRIAD_TO_IDX.get(triad, 0)
    if root_idx == 0:  # no root -> canonical no-chord tag
        return 0
    return root_idx * NUM_TRIADS + triad_idx


def split_root_triad(joint_idx: int) -> (str, str):
    """Inverse of :func:`join_root_triad`."""
    joint_idx = int(joint_idx)
    joint_idx = max(0, min(joint_idx, NUM_ROOTS * NUM_TRIADS - 1))
    root_idx, triad_idx = divmod(joint_idx, NUM_TRIADS)
    return ROOTS[root_idx], TRIADS[triad_idx]


# ---------------------------------------------------------------------------
# Decomposer (9 -> 6 adapter)
# ---------------------------------------------------------------------------

class ChordDecomposer:
    """Decompose chord labels into the paper's 6 components.

    Reuses the 9-component parser and folds the result down to 6 dimensions.
    """

    def __init__(self):
        self._inner = _ChordDecomposer9()
        self.vocab = CHORD_VOCAB
        self.vocab_idx = CHORD_VOCAB_IDX

    def decompose(self, chord_label: str) -> Dict[str, str]:
        """Return the 6 component *labels* (strings) for one chord."""
        nine = self._inner.decompose(chord_label)
        return self._fold(nine)

    @staticmethod
    def _fold(nine: Dict[str, str]) -> Dict[str, str]:
        root = nine.get('root', 'N')
        triad = nine.get('triad', 'N')
        seventh = nine.get('7th', 'N')
        sixth = nine.get('6th', 'N')

        # Fold the 6th into bb7 (paper convention) when no 7th is present.
        if sixth == '6' and seventh == 'N':
            seventh = 'bb7'

        joint_idx = join_root_triad(root, triad)
        return {
            'root_triad': ROOT_TRIAD_VOCAB[joint_idx],
            'bass': nine.get('bass', 'N'),
            '7th': seventh,
            '9th': nine.get('9th', 'N'),
            '11th': nine.get('11th', 'N'),
            '13th': nine.get('13th', 'N'),
        }

    def decompose_batch(self, chord_labels: List[str]) -> Dict[str, np.ndarray]:
        """Decompose a list of chord labels into per-component index arrays."""
        result: Dict[str, List[int]] = {c: [] for c in COMPONENT_NAMES}
        for label in chord_labels:
            comps = self.decompose(label)
            for component in COMPONENT_NAMES:
                idx = self.vocab_idx[component].get(comps[component], 0)
                result[component].append(idx)
        return {c: np.array(v, dtype=np.int64) for c, v in result.items()}

    def to_indices(self, components: Dict[str, str]) -> Dict[str, int]:
        return {
            component: self.vocab_idx[component].get(components.get(component, 'N'), 0)
            for component in COMPONENT_NAMES
        }


# ---------------------------------------------------------------------------
# Reassembler (6 -> 9 adapter)
# ---------------------------------------------------------------------------

class ChordReassembler:
    """Reassemble chord labels from the paper's 6 components."""

    def __init__(self):
        self._inner = _ChordReassembler9()
        self.vocab = CHORD_VOCAB
        self.vocab_idx = CHORD_VOCAB_IDX

    def reassemble(self, components: Dict[str, str]) -> str:
        """Reassemble from 6 component *label* strings."""
        root_triad = components.get('root_triad', 'N')
        seventh = components.get('7th', 'N')

        if root_triad == 'N' or ':' not in root_triad:
            root, triad = 'N', 'N'
        else:
            root, triad = root_triad.split(':', 1)

        nine = {
            'root': root,
            'bass': components.get('bass', 'N'),
            'triad': triad,
            'misc': 'N',
            '6th': 'N',
            '7th': 'N',
            '9th': components.get('9th', 'N'),
            '11th': components.get('11th', 'N'),
            '13th': components.get('13th', 'N'),
        }

        # Unfold bb7: on a maj/min triad it denotes the 6th; otherwise it is a
        # genuine (diminished) 7th interval.
        if seventh == 'bb7' and triad in ('maj', 'min'):
            nine['6th'] = '6'
        else:
            nine['7th'] = seventh

        return self._inner.reassemble(nine)

    def reassemble_from_indices(self, indices: Dict[str, int]) -> str:
        components: Dict[str, str] = {}
        for component in COMPONENT_NAMES:
            idx = indices.get(component, 0)
            if hasattr(idx, 'item'):
                idx = idx.item()
            idx = int(idx)
            idx = max(0, min(idx, len(self.vocab[component]) - 1))
            components[component] = self.vocab[component][idx]
        return self.reassemble(components)

    def reassemble_batch(self, indices_batch: Dict[str, np.ndarray]) -> List[str]:
        first = indices_batch[COMPONENT_NAMES[0]]
        if hasattr(first, 'numpy'):
            first = first.numpy()
        first = np.asarray(first)
        flat = {}
        for comp in COMPONENT_NAMES:
            arr = indices_batch[comp]
            if hasattr(arr, 'numpy'):
                arr = arr.numpy()
            flat[comp] = np.asarray(arr).flatten()
        return [
            self.reassemble_from_indices({c: flat[c][i] for c in COMPONENT_NAMES})
            for i in range(first.size)
        ]

    def reassemble_batch_2d(self, indices_batch: Dict[str, np.ndarray]) -> List[List[str]]:
        first = indices_batch[COMPONENT_NAMES[0]]
        if hasattr(first, 'numpy'):
            first = first.numpy()
        first = np.asarray(first)
        if first.ndim == 1:
            return [self.reassemble_batch(indices_batch)]

        arrs = {}
        for comp in COMPONENT_NAMES:
            arr = indices_batch[comp]
            if hasattr(arr, 'numpy'):
                arr = arr.numpy()
            arrs[comp] = np.asarray(arr)

        batch_size, seq_len = first.shape
        result: List[List[str]] = []
        for b in range(batch_size):
            seq = [
                self.reassemble_from_indices({c: arrs[c][b, t] for c in COMPONENT_NAMES})
                for t in range(seq_len)
            ]
            result.append(seq)
        return result


def get_vocab_sizes() -> Dict[str, int]:
    """Vocabulary size per component."""
    return {component: len(labels) for component, labels in CHORD_VOCAB.items()}
