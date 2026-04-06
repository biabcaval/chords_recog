"""
Build a full-chord vocabulary from training data for the FullChordCRF.

Scans preprocessed .pt files, collects every unique chord label, decomposes
each into the 9-component representation, and produces a deterministic
label-to-index mapping plus a pre-computed component matrix.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from utils.chord_decomposition import (
    CHORD_VOCAB,
    CHORD_VOCAB_IDX,
    COMPONENT_NAMES,
    ChordDecomposer,
)
from utils.mir_eval_modules import idx2voca_chord

logger = logging.getLogger(__name__)


def _resolve_dataset_dir(root_dir: str, dataset_name: str,
                         mp3_string: str, feature_string: str) -> Optional[str]:
    """Return the path to the preprocessed directory, trying
    result_decomposed/ first, then result/."""
    for prefix in ('result_decomposed', 'result'):
        p = os.path.join(root_dir, prefix,
                         dataset_name + '_voca', mp3_string, feature_string)
        if os.path.isdir(p):
            return p
    return None


def build_vocab_from_pt_files(
    data_root: str,
    dataset_names: List[str],
    config,
    max_files_per_dataset: int = 0,
) -> Tuple[List[str], Dict[str, int], torch.Tensor]:
    """Scan training .pt files and build the full-chord vocabulary.

    Args:
        data_root: Root directory containing datasets.
        dataset_names: Which datasets to scan.
        config: HParams config (used for mp3/feature path strings).
        max_files_per_dataset: If >0, sample at most this many files per
            dataset (useful for quick testing).

    Returns:
        chord_vocab: Sorted list of unique chord label strings.
        chord_to_idx: Mapping from label to integer index.
        component_matrix: ``(len(chord_vocab), 9)`` int64 tensor where
            row *i* holds the 9 component indices for ``chord_vocab[i]``.
    """
    mp3_cfg = config.mp3
    feat_cfg = config.feature
    mp3_string = "%d_%.1f_%.1f" % (
        mp3_cfg['song_hz'], mp3_cfg['inst_len'], mp3_cfg['skip_interval'])
    feature_string = "cqt_%d_%d_%d" % (
        feat_cfg['n_bins'], feat_cfg['bins_per_octave'], feat_cfg['hop_length'])

    idx2chord = idx2voca_chord()
    unique_labels = set()

    for ds_name in dataset_names:
        ds_dir = _resolve_dataset_dir(data_root, ds_name,
                                      mp3_string, feature_string)
        if ds_dir is None:
            logger.warning(f"Dataset dir not found for '{ds_name}', skipping")
            continue

        pt_files = list(Path(ds_dir).rglob('*.pt'))
        if max_files_per_dataset > 0:
            pt_files = pt_files[:max_files_per_dataset]

        for pt_path in pt_files:
            try:
                data = torch.load(pt_path, map_location='cpu', weights_only=False)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue

            if 'original_chord_labels' in data:
                labels = data['original_chord_labels']
            elif 'original_chords' in data:
                labels = data['original_chords']
            elif 'chord' in data:
                chords = data['chord']
                if isinstance(chords, list) and chords and isinstance(chords[0], int):
                    labels = [idx2chord.get(c, 'N') for c in chords]
                else:
                    labels = list(chords)
            else:
                continue

            unique_labels.update(labels)

        logger.info(f"  {ds_name}: scanned {len(pt_files)} files, "
                    f"running unique count: {len(unique_labels)}")

    if 'N' not in unique_labels:
        unique_labels.add('N')

    chord_vocab = sorted(unique_labels)
    chord_to_idx = {label: i for i, label in enumerate(chord_vocab)}

    component_matrix = _build_component_matrix(chord_vocab)

    logger.info(f"Chord vocabulary: {len(chord_vocab)} unique labels")
    return chord_vocab, chord_to_idx, component_matrix


def _build_component_matrix(chord_vocab: List[str]) -> torch.Tensor:
    """Decompose every vocab entry and return the component index matrix."""
    decomposer = ChordDecomposer()
    n = len(chord_vocab)
    matrix = torch.zeros(n, len(COMPONENT_NAMES), dtype=torch.long)

    for i, label in enumerate(chord_vocab):
        components = decomposer.decompose(label)
        for j, comp_name in enumerate(COMPONENT_NAMES):
            comp_val = components.get(comp_name, 'N')
            idx = CHORD_VOCAB_IDX[comp_name].get(comp_val, 0)
            matrix[i, j] = idx

    return matrix


def validate_vocab(chord_vocab: List[str],
                   component_matrix: torch.Tensor) -> bool:
    """Verify round-trip consistency for every vocab entry.

    Decomposes each label independently and checks that the resulting
    indices match ``component_matrix``.  Logs warnings for mismatches.

    Returns True if all entries pass.
    """
    decomposer = ChordDecomposer()
    ok = True
    for i, label in enumerate(chord_vocab):
        components = decomposer.decompose(label)
        for j, comp_name in enumerate(COMPONENT_NAMES):
            comp_val = components.get(comp_name, 'N')
            expected_idx = CHORD_VOCAB_IDX[comp_name].get(comp_val, 0)
            if component_matrix[i, j].item() != expected_idx:
                logger.warning(
                    f"Vocab mismatch at [{i}] '{label}' component "
                    f"'{comp_name}': matrix={component_matrix[i, j].item()}, "
                    f"expected={expected_idx} ('{comp_val}')")
                ok = False
    if ok:
        logger.info("Vocab validation passed: all entries consistent")
    return ok


def build_component_tuple_index(
    component_matrix: torch.Tensor,
) -> Dict[tuple, int]:
    """Build a reverse lookup from component-index tuple to vocab index.

    Used during CRF training to convert per-frame 9-component labels
    into a single joint tag index.
    """
    mapping = {}
    for i in range(component_matrix.shape[0]):
        key = tuple(component_matrix[i].tolist())
        mapping[key] = i
    return mapping


def save_vocab(path: str, chord_vocab: List[str],
               chord_to_idx: Dict[str, int],
               component_matrix: torch.Tensor) -> None:
    torch.save({
        'chord_vocab': chord_vocab,
        'chord_to_idx': chord_to_idx,
        'component_matrix': component_matrix,
    }, path)
    logger.info(f"Vocab saved to {path} ({len(chord_vocab)} entries)")


def load_vocab(path: str):
    data = torch.load(path, map_location='cpu', weights_only=False)
    return (data['chord_vocab'],
            data['chord_to_idx'],
            data['component_matrix'])
