#!/usr/bin/env python3
"""
Validate that .pt files contain original_chord_labels with full extensions.

Compares the new ``original_chord_labels`` field against the legacy
``chord`` index field to quantify information gained by bypassing the
170-chord funnel.

Usage:
    python scripts/validate_chord_labels.py --data_root /path/to/datasets
    python scripts/validate_chord_labels.py --data_root /path/to/datasets --dataset balanced_v1_full
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.mir_eval_modules import idx2voca_chord
from utils.chord_decomposition import ChordDecomposer, COMPONENT_NAMES
from utils.chords import Chords


def validate_transpose():
    """Quick self-test for transpose_chord_label."""
    cases = [
        ('C:maj9/E',  2, 'D:maj9/F#'),
        ('A#:min7',  -3, 'G:min7'),
        ('N',         5, 'N'),
        ('X',         0, 'X'),
        ('Bb:7/Ab',   1, 'B:7/A'),
        ('C',         7, 'G'),
        ('C:min/Eb', 12, 'C:min/D#'),
        ('F#:dim7',  -1, 'F:dim7'),
    ]
    passed = 0
    for label, semi, expected in cases:
        result = Chords.transpose_chord_label(label, semi)
        ok = result == expected
        status = 'OK' if ok else 'FAIL'
        print(f"  {status}: transpose('{label}', {semi:+d}) -> '{result}'"
              f"{'  (expected: ' + expected + ')' if not ok else ''}")
        passed += int(ok)
    print(f"\n  Transpose tests: {passed}/{len(cases)} passed\n")
    return passed == len(cases)


def scan_pt_files(data_root, dataset=None):
    """Find .pt files in result/ or result_decomposed/."""
    root = Path(data_root)
    pt_files = []
    for subdir in ['result', 'result_decomposed']:
        base = root / subdir
        if not base.exists():
            continue
        for voca_dir in sorted(base.iterdir()):
            if not voca_dir.is_dir() or not voca_dir.name.endswith('_voca'):
                continue
            ds_name = voca_dir.name.replace('_voca', '')
            if dataset and ds_name != dataset:
                continue
            for pt in voca_dir.rglob('*.pt'):
                pt_files.append((subdir, ds_name, pt))
    return pt_files


def main():
    parser = argparse.ArgumentParser(description='Validate chord label fields in .pt files')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--dataset', default=None,
                        help='Restrict to a specific dataset name')
    parser.add_argument('--max_files', type=int, default=500,
                        help='Max files to sample per dataset (default: 500)')
    args = parser.parse_args()

    print("=" * 70)
    print("1. Transpose self-test")
    print("=" * 70)
    transpose_ok = validate_transpose()

    print("=" * 70)
    print("2. Scanning .pt files")
    print("=" * 70)
    pt_files = scan_pt_files(args.data_root, args.dataset)
    print(f"Found {len(pt_files)} .pt files\n")

    if not pt_files:
        print("No .pt files found. Nothing to validate.")
        return

    idx2chord = idx2voca_chord()
    decomposer = ChordDecomposer()

    datasets_seen = Counter()
    for _, ds, _ in pt_files:
        datasets_seen[ds] += 1

    for ds_name, total_count in datasets_seen.most_common():
        ds_files = [(s, d, p) for s, d, p in pt_files if d == ds_name]
        sample = ds_files[:args.max_files]

        has_original = 0
        has_index_only = 0
        total_frames = 0
        extension_frames = Counter()
        delta_frames = 0
        label_mismatches = []

        for subdir, _, pt_path in sample:
            try:
                data = torch.load(pt_path, map_location='cpu', weights_only=False)
            except Exception:
                continue

            if not isinstance(data, dict):
                continue

            original = data.get('original_chord_labels')
            chord_idx = data.get('chord')

            if original is not None:
                has_original += 1
                chord_labels = list(original)
            elif chord_idx is not None:
                has_index_only += 1
                if isinstance(chord_idx, list) and chord_idx and isinstance(chord_idx[0], int):
                    chord_labels = [idx2chord.get(c, 'N') for c in chord_idx]
                else:
                    continue
            else:
                continue

            total_frames += len(chord_labels)

            for lbl in chord_labels:
                d = decomposer.decompose(lbl)
                for ext in ('9th', '11th', '13th'):
                    if d.get(ext, 'N') != 'N':
                        extension_frames[ext] += 1
                if d.get('bass', 'N') != 'N':
                    extension_frames['bass_inversion'] += 1

            if original is not None and chord_idx is not None:
                idx_labels = []
                if isinstance(chord_idx, list) and chord_idx and isinstance(chord_idx[0], int):
                    idx_labels = [idx2chord.get(c, 'N') for c in chord_idx]
                if idx_labels and len(idx_labels) == len(original):
                    for orig_l, idx_l in zip(original, idx_labels):
                        d_orig = decomposer.decompose(orig_l)
                        d_idx = decomposer.decompose(idx_l)
                        if d_orig != d_idx:
                            delta_frames += 1
                            if len(label_mismatches) < 5:
                                label_mismatches.append((orig_l, idx_l, d_orig, d_idx))

        print(f"\n--- {ds_name} ({len(sample)}/{total_count} files sampled) ---")
        print(f"  Files with original_chord_labels: {has_original}")
        print(f"  Files with index-only fallback:   {has_index_only}")
        print(f"  Total frames sampled:             {total_frames}")

        if extension_frames:
            print(f"  Extension / inversion frames:")
            for ext, count in extension_frames.most_common():
                pct = 100 * count / total_frames if total_frames else 0
                print(f"    {ext}: {count} ({pct:.1f}%)")
        else:
            print(f"  No extensions or inversions detected (all N)")

        if delta_frames:
            print(f"  Decomposition delta (original vs 170-idx): {delta_frames} frames differ")
            for orig_l, idx_l, d_orig, d_idx in label_mismatches:
                diffs = {k: (d_orig[k], d_idx[k]) for k in d_orig if d_orig[k] != d_idx.get(k)}
                print(f"    '{orig_l}' vs '{idx_l}': {diffs}")

    print(f"\n{'='*70}")
    print("Validation complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
