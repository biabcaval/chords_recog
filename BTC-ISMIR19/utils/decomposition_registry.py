# encoding: utf-8
"""
Registry that selects a chord-decomposition scheme by name.

Two schemes are available:

* ``'full9'`` (default): the project's 9-component decomposition
  (root, bass, triad, misc, 6th, 7th, 9th, 11th, 13th) from
  :mod:`utils.chord_decomposition`.
* ``'paper6'``: the ChordFormer paper's 6-component decomposition
  (root_triad, bass, 7th, 9th, 11th, 13th) from
  :mod:`utils.chord_decomposition_paper6`.

Both modules expose the same surface (``COMPONENT_NAMES``, ``CHORD_VOCAB``,
``CHORD_VOCAB_IDX``, ``ChordDecomposer``, ``ChordReassembler``,
``get_vocab_sizes``), so callers can stay scheme-agnostic by going through
:func:`get_decomposition`.
"""

from types import SimpleNamespace

from utils import chord_decomposition as _full9
from utils import chord_decomposition_paper6 as _paper6

_ALIASES = {
    'full9': 'full9', '9': 'full9', 'nine': 'full9', 'default': 'full9',
    'paper6': 'paper6', '6': 'paper6', 'six': 'paper6', 'paper': 'paper6',
}

_MODULES = {'full9': _full9, 'paper6': _paper6}

DECOMPOSITION_CHOICES = ('full9', 'paper6')


def normalize_scheme(scheme: str) -> str:
    """Resolve a user-facing scheme name/alias to its canonical key."""
    if scheme is None:
        return 'full9'
    key = _ALIASES.get(str(scheme).strip().lower())
    if key is None:
        raise ValueError(
            f"Unknown decomposition scheme '{scheme}'. "
            f"Valid: {sorted(set(_ALIASES))}."
        )
    return key


def get_decomposition(scheme: str = 'full9') -> SimpleNamespace:
    """Return the decomposition module for ``scheme``.

    The returned object exposes ``COMPONENT_NAMES``, ``CHORD_VOCAB``,
    ``CHORD_VOCAB_IDX``, ``ChordDecomposer``, ``ChordReassembler`` and
    ``get_vocab_sizes`` for the requested scheme.
    """
    module = _MODULES[normalize_scheme(scheme)]
    return SimpleNamespace(
        scheme=normalize_scheme(scheme),
        COMPONENT_NAMES=module.COMPONENT_NAMES,
        CHORD_VOCAB=module.CHORD_VOCAB,
        CHORD_VOCAB_IDX=module.CHORD_VOCAB_IDX,
        ChordDecomposer=module.ChordDecomposer,
        ChordReassembler=module.ChordReassembler,
        get_vocab_sizes=module.get_vocab_sizes,
    )
