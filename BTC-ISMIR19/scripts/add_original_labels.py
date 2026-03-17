#!/usr/bin/env python3
"""
Add original chord labels from .lab files to existing legacy .pt files.

The legacy preprocessing pipeline (generate_labels_features_voca) stores
170-class chord indices that discard extension info (9th, 11th, 13th).
This script reads the original .lab annotations and adds an
'original_chord_labels' field with the full chord strings, properly
transposed for each pitch-shift augmentation.

Target directory layout
-----------------------
  {data_root}/result/{dataset}_voca/{mp3_str}/{feat_str}/{song_name}/{stretch}_{shift}_{idx}.pt

Corresponding .lab files
------------------------
  {data_root}/{dataset}/annotations/{song_name}.lab

Usage:
    # Dry run (report matching stats, don't modify files)
    python scripts/add_original_labels.py --data_root /path/to/datasets --dry_run

    # Validate alignment against existing 170-class chord IDs
    python scripts/add_original_labels.py --data_root /path/to/datasets --validate

    # Process all files (add original_chord_labels)
    python scripts/add_original_labels.py --data_root /path/to/datasets --num_workers 8

    # Process specific feature config only
    python scripts/add_original_labels.py --data_root /path/to/datasets --feature_config cqt_252_36_2048
"""

import argparse
import os
import sys
from bisect import bisect_right
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.chord_decomposition import transpose_chord  # noqa: E402

# 170-class vocabulary (12 roots x 14 qualities + N + X)
_ROOT_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_QUALITY_LIST = [
    'min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7',
    'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4',
]


def _idx170_to_label(idx: int) -> str:
    """Convert a 170-class chord index back to a label string."""
    if idx == 169:
        return 'N'
    if idx == 168:
        return 'X'
    root = _ROOT_LIST[idx // 14]
    quality_idx = idx % 14
    if quality_idx == 1:
        return root  # plain major, no quality suffix
    return f"{root}:{_QUALITY_LIST[quality_idx]}"


# ---------------------------------------------------------------------------
# .lab file handling
# ---------------------------------------------------------------------------

def load_lab_file(lab_path: str):
    """Load chord annotations from a .lab file.

    Returns a sorted list of (start, end, chord_label).
    """
    annotations = []
    with open(lab_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    chord = parts[2]
                    annotations.append((start, end, chord))
                except ValueError:
                    continue
    annotations.sort(key=lambda x: x[0])
    return annotations


def _build_lookup(annotations):
    """Pre-compute sorted start times for bisect-based lookup."""
    starts = [a[0] for a in annotations]
    return starts


def chord_at_time(annotations, starts, t):
    """O(log n) chord lookup using bisect on pre-sorted start times."""
    idx = bisect_right(starts, t) - 1
    if idx < 0:
        return 'N'
    start, end, chord = annotations[idx]
    if t < end:
        return chord
    return 'N'


def chords_for_segment(annotations, starts, seg_start, n_frames, time_interval):
    """Get chord labels for each frame in a segment."""
    return [
        chord_at_time(annotations, starts, seg_start + i * time_interval)
        for i in range(n_frames)
    ]


# ---------------------------------------------------------------------------
# Filename / path parsing
# ---------------------------------------------------------------------------

def parse_pt_filename(filename: str):
    """Parse {stretch}_{shift}_{idx}.pt -> (stretch, shift, idx)."""
    name = Path(filename).stem
    parts = name.split('_')
    if len(parts) >= 3:
        try:
            stretch = float(parts[0])
            shift = int(parts[1])
            idx = int(parts[2])
            return stretch, shift, idx
        except ValueError:
            pass
    return 1.0, 0, 0


def discover_datasets(data_root: str, feature_config: str = None, mp3_str: str = '22050_10.0_5.0'):
    """Discover dataset directories and match songs to .lab files.

    Returns:
        list of (dataset_name, feat_dir, lab_dir) tuples
    """
    data_root = Path(data_root)
    result_dir = data_root / 'result'
    if not result_dir.exists():
        print(f"ERROR: result/ directory not found at {result_dir}")
        return []

    datasets = []
    for voca_dir in sorted(result_dir.iterdir()):
        if not voca_dir.is_dir() or not voca_dir.name.endswith('_voca'):
            continue
        dataset_name = voca_dir.name.replace('_voca', '')

        mp3_dir = voca_dir / mp3_str
        if not mp3_dir.exists():
            continue

        if feature_config:
            feat_dirs = [mp3_dir / feature_config]
        else:
            feat_dirs = sorted(mp3_dir.iterdir())

        for feat_dir in feat_dirs:
            if not feat_dir.is_dir():
                continue

            lab_dir = data_root / dataset_name / 'annotations'
            if not lab_dir.exists():
                print(f"WARNING: annotations dir not found for {dataset_name}: {lab_dir}")
                continue

            datasets.append((dataset_name, feat_dir, lab_dir))

    return datasets


def match_songs(feat_dir: Path, lab_dir: Path):
    """Match song folders in feat_dir to .lab files in lab_dir.

    Returns:
        matched: dict {song_name: (song_feat_dir, lab_path)}
        unmatched: list of song_names without a .lab file
    """
    lab_files = {}
    for f in lab_dir.iterdir():
        if f.suffix == '.lab':
            lab_files[f.stem] = f

    matched = {}
    unmatched = []
    for song_dir in sorted(feat_dir.iterdir()):
        if not song_dir.is_dir():
            continue
        song_name = song_dir.name
        if song_name in lab_files:
            matched[song_name] = (song_dir, lab_files[song_name])
        else:
            unmatched.append(song_name)

    return matched, unmatched


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_song(args):
    """Process all .pt files for a single song. Designed for Pool.map()."""
    song_name, song_dir, lab_path, time_interval, n_frames_expected, dry_run, validate = args

    pt_files = sorted(song_dir.glob('*.pt'))
    stats = {'updated': 0, 'skipped': 0, 'errors': 0, 'mismatches': 0, 'checked': 0}

    if dry_run:
        stats['updated'] = len(pt_files)
        return song_name, stats

    annotations = load_lab_file(str(lab_path))
    starts = _build_lookup(annotations)

    for pt_path in pt_files:
        try:
            stretch, shift, idx = parse_pt_filename(pt_path.name)

            data = torch.load(pt_path, map_location='cpu', weights_only=False)

            if 'original_chord_labels' in data and not validate:
                stats['skipped'] += 1
                continue

            etc = data.get('etc', '')
            if isinstance(etc, str) and '_' in etc:
                seg_start = float(etc.split('_')[0])
            else:
                stats['errors'] += 1
                continue

            # Determine n_frames from feature shape or fallback
            feature = data.get('feature')
            if feature is not None and hasattr(feature, 'shape'):
                if len(feature.shape) == 2:
                    n_frames = min(feature.shape)
                else:
                    n_frames = feature.shape[0]
            else:
                n_frames = n_frames_expected

            # Adjust for stretch (stretch_factors=[1.0] in practice, but be safe)
            effective_start = seg_start / stretch if stretch != 1.0 else seg_start
            effective_interval = time_interval / stretch if stretch != 1.0 else time_interval

            labels = chords_for_segment(annotations, starts, effective_start, n_frames, effective_interval)

            if shift != 0:
                labels = [transpose_chord(c, shift) for c in labels]

            if validate:
                existing_chords = data.get('chord', [])
                stats['checked'] += len(existing_chords)
                for i, (lab_chord, existing_idx) in enumerate(zip(labels, existing_chords)):
                    existing_idx = int(existing_idx)
                    expected_label = _idx170_to_label(existing_idx)
                    lab_root = lab_chord.split(':')[0].split('/')[0] if ':' in lab_chord or '/' in lab_chord else lab_chord
                    exp_root = expected_label.split(':')[0].split('/')[0] if ':' in expected_label or '/' in expected_label else expected_label
                    if lab_root == 'N' and exp_root == 'N':
                        continue
                    if lab_root != exp_root and lab_root != 'N' and exp_root != 'N':
                        stats['mismatches'] += 1
                continue

            data['original_chord_labels'] = labels
            torch.save(data, pt_path)
            stats['updated'] += 1

        except Exception as e:
            print(f"  Error processing {pt_path.name} in {song_name}: {e}")
            stats['errors'] += 1

    return song_name, stats


def main():
    parser = argparse.ArgumentParser(
        description='Add original chord labels from .lab files to legacy .pt files'
    )
    parser.add_argument('--data_root', required=True, help='Root directory with datasets')
    parser.add_argument('--feature_config', default=None,
                        help='Feature config to process (e.g. cqt_252_36_2048). Default: all.')
    parser.add_argument('--mp3_str', default='22050_10.0_5.0',
                        help='MP3 config string (default: 22050_10.0_5.0)')
    parser.add_argument('--hop_length', type=int, default=2048, help='Hop length in samples')
    parser.add_argument('--song_hz', type=int, default=22050, help='Sample rate')
    parser.add_argument('--n_frames', type=int, default=108, help='Expected frames per segment')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--dry_run', action='store_true', help='Report stats without modifying files')
    parser.add_argument('--validate', action='store_true',
                        help='Validate alignment against existing 170-class chord IDs')
    args = parser.parse_args()

    time_interval = args.hop_length / args.song_hz

    print(f"Configuration:")
    print(f"  data_root:      {args.data_root}")
    print(f"  feature_config: {args.feature_config or 'all'}")
    print(f"  time_interval:  {time_interval:.6f}s (hop={args.hop_length}, sr={args.song_hz})")
    print(f"  n_frames:       {args.n_frames}")
    print(f"  num_workers:    {args.num_workers}")
    print(f"  mode:           {'validate' if args.validate else 'dry_run' if args.dry_run else 'process'}")
    print()

    datasets = discover_datasets(args.data_root, args.feature_config, args.mp3_str)
    if not datasets:
        print("No datasets found.")
        return 1

    total_stats = {'updated': 0, 'skipped': 0, 'errors': 0, 'mismatches': 0, 'checked': 0}
    total_songs = 0
    total_unmatched = 0

    for dataset_name, feat_dir, lab_dir in datasets:
        print(f"=== {dataset_name} ({feat_dir.name}) ===")

        matched, unmatched = match_songs(feat_dir, lab_dir)
        total_unmatched += len(unmatched)

        print(f"  Songs matched: {len(matched)}, unmatched: {len(unmatched)}")

        if unmatched and len(unmatched) <= 5:
            print(f"  Unmatched: {unmatched}")
        elif unmatched:
            print(f"  First 5 unmatched: {unmatched[:5]}...")

        if not matched:
            continue

        work_items = [
            (song_name, song_dir, lab_path, time_interval, args.n_frames,
             args.dry_run, args.validate)
            for song_name, (song_dir, lab_path) in matched.items()
        ]
        total_songs += len(work_items)

        mode_label = 'validate' if args.validate else 'dry_run' if args.dry_run else 'process'
        desc = f"  {dataset_name} [{mode_label}]"

        if args.num_workers > 1 and len(work_items) > 1:
            results = []
            with Pool(processes=min(args.num_workers, len(work_items))) as pool:
                for result in tqdm(
                    pool.imap_unordered(process_song, work_items),
                    total=len(work_items), desc=desc, unit="song"
                ):
                    results.append(result)
        else:
            results = []
            for item in tqdm(work_items, desc=desc, unit="song"):
                results.append(process_song(item))

        ds_stats = {'updated': 0, 'skipped': 0, 'errors': 0, 'mismatches': 0, 'checked': 0}
        for song_name, stats in results:
            for k in ds_stats:
                ds_stats[k] += stats[k]

        for k in total_stats:
            total_stats[k] += ds_stats[k]

        print(f"  Results: updated={ds_stats['updated']}, skipped={ds_stats['skipped']}, "
              f"errors={ds_stats['errors']}")
        if args.validate:
            mismatch_rate = (ds_stats['mismatches'] / ds_stats['checked'] * 100) if ds_stats['checked'] else 0
            print(f"  Validation: {ds_stats['mismatches']} root mismatches / "
                  f"{ds_stats['checked']} checked ({mismatch_rate:.2f}%)")
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Songs processed: {total_songs}")
    print(f"  Songs unmatched: {total_unmatched}")
    print(f"  Files updated:   {total_stats['updated']}")
    print(f"  Files skipped:   {total_stats['skipped']}")
    print(f"  Errors:          {total_stats['errors']}")
    if args.validate:
        total_rate = (total_stats['mismatches'] / total_stats['checked'] * 100) if total_stats['checked'] else 0
        print(f"  Root mismatches: {total_stats['mismatches']} / {total_stats['checked']} ({total_rate:.2f}%)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
