#!/usr/bin/env python3
"""
Create a balanced dataset by selecting songs from multiple existing datasets.

Two input modes:
  A) --lab_dir   : directory with .lab files (uses those directly)
  B) --manifest  : text file listing paths (one per line) to .lab files;
                    the .lab and audio are sourced from the VM datasets

In both modes the script:
  1. Parses chord class distributions per song
  2. Searches for matching audio files across all existing datasets
  3. Splits into train/test with stratified class balancing
  4. Copies (or symlinks) audio + annotations into two new dataset folders

Usage:
    # Mode A: flat directory
    python scripts/create_balanced_dataset.py \
        --lab_dir /path/to/lab/files \
        --data_root /home/daniel.melo/datasets \
        --dataset_name balanced_v1

    # Mode B: manifest file (list of paths)
    python scripts/create_balanced_dataset.py \
        --manifest /path/to/balanced_selection.txt \
        --data_root /home/daniel.melo/datasets \
        --dataset_name balanced_v1
"""

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SEARCH_DATASETS = [
    'billboard',
    'dj_avan_songbook1',
    'dj_avan_songbook2',
    'jaah',
    'queen',
    'robbiewilliams',
    'rwc',
]

AUDIO_EXTENSIONS = ('.mp3', '.wav', '.flac', '.m4a')

MANIFEST_DIR_TO_DATASET = {
    'chords_billboard_verified': 'billboard',
    'chords_djavan_songbook1_verified': 'dj_avan_songbook1',
    'chords_djavan_songbook2_verified': 'dj_avan_songbook2',
    'chords_jaah_verified': 'jaah',
    'chords_queen_verified': 'queen',
    'chords_robbie_verified': 'robbiewilliams',
    'chords_rwc_verified': 'rwc',
}


def parse_manifest(manifest_path):
    """Parse a manifest text file into a list of (lab_filename, dataset_hint) tuples.

    Each line is an absolute path like:
      I:\\...\\chords_billboard_verified\\0006-Bette_Midler-The_Rose.lab

    The parent folder name is mapped to a dataset hint via MANIFEST_DIR_TO_DATASET.
    """
    entries = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            normalized = line.replace('\\', '/')
            parts = normalized.rstrip('/').split('/')
            lab_filename = parts[-1]
            parent_dir = parts[-2].lower() if len(parts) >= 2 else ''
            dataset_hint = MANIFEST_DIR_TO_DATASET.get(parent_dir)
            entries.append((lab_filename, dataset_hint))
    return entries


def find_lab_in_datasets(data_root, lab_filename, dataset_hint=None, datasets=None):
    """Find a .lab file in the annotations/ folders of existing datasets.

    Searches dataset_hint first (if given), then all others.
    Returns (dataset_name, full_lab_path) or (None, None).
    """
    if datasets is None:
        datasets = SEARCH_DATASETS

    search_order = []
    if dataset_hint and dataset_hint in datasets:
        search_order.append(dataset_hint)
    search_order.extend(d for d in datasets if d not in search_order)

    lab_stem = os.path.splitext(lab_filename)[0]
    lab_normalized = normalize_for_match(lab_stem)

    for ds in search_order:
        annot_dir = os.path.join(data_root, ds, 'annotations')
        if not os.path.isdir(annot_dir):
            continue
        for f in os.listdir(annot_dir):
            if not f.lower().endswith('.lab'):
                continue
            f_normalized = normalize_for_match(os.path.splitext(f)[0])
            if f_normalized == lab_normalized:
                return ds, os.path.join(annot_dir, f)

    return None, None


def parse_lab_file(lab_path):
    """Parse a .lab file and return list of (start, end, chord_label) tuples."""
    segments = []
    with open(lab_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1])
                chord = parts[2]
                segments.append((start, end, chord))
    return segments


QUALITY_FAMILIES = {
    'maj': 'maj', '1': 'maj',
    'min': 'min',
    '7': 'dom7',
    'maj7': 'maj7',
    'min7': 'min7',
    'dim': 'dim', 'dim7': 'dim',
    'aug': 'aug',
    'sus2': 'sus', 'sus4': 'sus',
    'hdim7': 'hdim7',
}

GENRE_GROUPS = {
    'billboard': 'pop_rock',
    'queen': 'pop_rock',
    'robbiewilliams': 'pop_rock',
    'rwc': 'pop_rock',
    'dj_avan_songbook1': 'mpb',
    'dj_avan_songbook2': 'mpb',
    'jaah': 'jazz',
}

QUALITY_FAMILY_NAMES = ['maj', 'min', 'dom7', 'min7', 'maj7',
                        'sus', 'dim', 'aug', 'hdim7', 'other', 'N']


def get_quality_family(chord):
    """Map a chord label to its quality family."""
    if chord == 'N':
        return 'N'
    parts = chord.split(':')
    if len(parts) < 2:
        return 'maj'
    q = parts[1].split('/')[0].split('(')[0]
    if q in QUALITY_FAMILIES:
        return QUALITY_FAMILIES[q]
    if 'min' in q:
        return 'min'
    if 'maj' in q:
        return 'maj'
    if '7' in q:
        return 'dom7'
    return 'other'


def compute_class_distribution(segments):
    """Compute weighted chord class distribution from segments (duration-weighted)."""
    dist = Counter()
    for start, end, chord in segments:
        duration = end - start
        if duration > 0:
            dist[chord] += duration
    return dist


def compute_quality_profile(segments):
    """Compute duration-weighted quality family distribution for a song.

    Returns a dict mapping quality family names to their percentage of total duration.
    This mirrors the approach from visualizer/data_loader.py compute_segment_stats.
    """
    family_dur = Counter()
    total_dur = 0
    for start, end, chord in segments:
        dur = end - start
        if dur > 0:
            family = get_quality_family(chord)
            family_dur[family] += dur
            total_dur += dur

    if total_dur == 0:
        return {f: 0.0 for f in QUALITY_FAMILY_NAMES}

    return {f: family_dur.get(f, 0) / total_dur for f in QUALITY_FAMILY_NAMES}


def get_dominant_class(distribution, exclude_n=True):
    """Return the chord class with the largest total duration."""
    filtered = {k: v for k, v in distribution.items()
                if not (exclude_n and k == 'N')}
    if not filtered:
        return 'N'
    return max(filtered, key=filtered.get)


def get_dominant_quality_family(quality_profile, exclude_n=True):
    """Return the quality family with highest duration share."""
    filtered = {k: v for k, v in quality_profile.items()
                if not (exclude_n and k == 'N')}
    if not filtered:
        return 'N'
    return max(filtered, key=filtered.get)


def normalize_for_match(name):
    """Normalize a filename stem for fuzzy matching."""
    return name.lower().replace(' ', '_')


def normalize_aggressive(name):
    """Aggressive normalization: spaces, hyphens, and diacritics all become underscores."""
    return name.lower().replace(' ', '_').replace('-', '_')


def find_audio_in_dataset(audio_dir, lab_stem):
    """
    Search for an audio file matching a .lab stem in a dataset's audio/ folder.
    Three-pass strategy (mirrors Preprocess.find_audio_path_generic):
      1. Exact basename match (case-insensitive, space→underscore)
      2. Prefix match (handles suffixes like jaah_018-artist_title)
      3. Numeric ID match (handles zero-padding differences)
    """
    if not os.path.isdir(audio_dir):
        return None

    lab_normalized = normalize_for_match(lab_stem)

    candidates = [f for f in os.listdir(audio_dir)
                  if f.lower().endswith(AUDIO_EXTENSIONS)]

    for filename in candidates:
        audio_normalized = normalize_for_match(os.path.splitext(filename)[0])
        if audio_normalized == lab_normalized:
            return filename

    for filename in candidates:
        audio_normalized = normalize_for_match(os.path.splitext(filename)[0])
        if (audio_normalized.startswith(lab_normalized + '-') or
                audio_normalized.startswith(lab_normalized + '_') or
                lab_normalized.startswith(audio_normalized + '-') or
                lab_normalized.startswith(audio_normalized + '_')):
            return filename

    lab_aggressive = normalize_aggressive(lab_stem)
    for filename in candidates:
        audio_aggressive = normalize_aggressive(os.path.splitext(filename)[0])
        if audio_aggressive == lab_aggressive:
            return filename

    lab_match = re.match(r'(.+?)(\d+)$', lab_normalized)
    if lab_match:
        lab_prefix, lab_num = lab_match.group(1), int(lab_match.group(2))
        for filename in candidates:
            audio_normalized = normalize_for_match(os.path.splitext(filename)[0])
            audio_match = re.match(r'(.+?)(\d+)(.*)', audio_normalized)
            if audio_match:
                audio_prefix = audio_match.group(1)
                audio_num = int(audio_match.group(2))
                if audio_prefix == lab_prefix and audio_num == lab_num:
                    return filename

    return None


def extract_dataset_hint(lab_stem):
    """Extract dataset name and numeric ID from specially-prefixed lab stems.

    Handles patterns like:
      - billboard_477  → ('billboard', 477)
      - hooktheory_09858_artist_song → ('hooktheory', 9858)
    Returns (dataset_hint, numeric_id) or (None, None).
    """
    m = re.match(r'^(billboard)_(\d+)$', lab_stem, re.IGNORECASE)
    if m:
        return m.group(1).lower(), int(m.group(2))

    m = re.match(r'^(hooktheory)_(\d+)_(.+)$', lab_stem, re.IGNORECASE)
    if m:
        return m.group(1).lower(), int(m.group(2))

    return None, None


def find_audio_by_numeric_id(audio_dir, numeric_id):
    """Find audio whose leading numeric ID matches (ignoring zero-padding).

    Matches e.g. numeric_id=477 against '0477-The_Hollies-Carrie-Anne.wav'.
    """
    if not os.path.isdir(audio_dir):
        return None
    candidates = [f for f in os.listdir(audio_dir)
                  if f.lower().endswith(AUDIO_EXTENSIONS)]
    for filename in candidates:
        m = re.match(r'^0*(\d+)', filename)
        if m and int(m.group(1)) == numeric_id:
            return filename
    return None


def search_audio_across_datasets(data_root, lab_stem, datasets=None):
    """Search all datasets for an audio file matching the lab stem.
    Returns (dataset_name, audio_filename) or (None, None).
    """
    if datasets is None:
        datasets = SEARCH_DATASETS

    dataset_hint, numeric_id = extract_dataset_hint(lab_stem)

    if dataset_hint and numeric_id is not None:
        if dataset_hint in datasets or dataset_hint == 'hooktheory':
            target = dataset_hint if dataset_hint != 'hooktheory' else None
            search_order = []
            if target:
                search_order.append(target)
            search_order.extend(d for d in datasets if d != target)

            for ds in search_order:
                audio_dir = os.path.join(data_root, ds, 'audio')
                match = find_audio_by_numeric_id(audio_dir, numeric_id)
                if match:
                    return ds, match
                match = find_audio_in_dataset(audio_dir, lab_stem)
                if match:
                    return ds, match
            return None, None

    for dataset in datasets:
        audio_dir = os.path.join(data_root, dataset, 'audio')
        match = find_audio_in_dataset(audio_dir, lab_stem)
        if match:
            return dataset, match
    return None, None


def stratified_split(song_data_list, test_ratio, seed):
    """Split songs into train/test keeping chord distributions balanced.

    Strategy:
      1. Group songs by genre (pop_rock / mpb / jazz) to preserve genre proportions
      2. Within each genre, sub-group by dominant quality family (maj / dom7 / min7 / ...)
      3. Draw test samples proportionally from each sub-group
      4. Print a comparison of train vs test quality distributions for verification

    Args:
        song_data_list: list of dicts, each with at least 'lab_stem', 'source_dataset',
                        'quality_profile', 'dominant_quality_family'
        test_ratio: fraction of songs for test
        seed: random seed

    Returns:
        (train_stems, test_stems): lists of lab_stem strings
    """
    rng = np.random.RandomState(seed)

    genre_quality_groups = defaultdict(list)
    for s in song_data_list:
        genre = GENRE_GROUPS.get(s['source_dataset'], 'unknown')
        dom_qf = s['dominant_quality_family']
        key = f"{genre}__{dom_qf}"
        genre_quality_groups[key].append(s['lab_stem'])

    train_stems = []
    test_stems = []

    for key in sorted(genre_quality_groups.keys()):
        group = genre_quality_groups[key]
        rng.shuffle(group)
        n_test = max(1, int(round(len(group) * test_ratio)))
        if len(group) <= 1:
            train_stems.extend(group)
        else:
            n_test = min(n_test, len(group) - 1)
            test_stems.extend(group[:n_test])
            train_stems.extend(group[n_test:])

    return train_stems, test_stems


def format_distribution(dist, top_n=15):
    """Format a Counter as a readable top-N list with percentages."""
    total = sum(dist.values())
    if total == 0:
        return "  (empty)"
    lines = []
    for chord, dur in dist.most_common(top_n):
        pct = 100.0 * dur / total
        lines.append(f"  {chord:>12s}: {pct:5.1f}%  ({dur:.1f}s)")
    if len(dist) > top_n:
        lines.append(f"  ... and {len(dist) - top_n} more classes")
    return "\n".join(lines)


def copy_or_link(src, dst, use_symlink=False):
    """Copy a file or create a symlink."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if use_symlink:
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def save_distribution_report(report_path, all_songs, train_set, test_set,
                             train_qf, test_qf, train_genres, test_genres,
                             args, dry_run=False):
    """Save a CSV report comparing train/test chord distributions."""
    import csv

    train_qf_total = sum(train_qf.values()) or 1
    test_qf_total = sum(test_qf.values()) or 1

    # Aggregate full chord distributions (not just quality families)
    train_chords = Counter()
    test_chords = Counter()
    for s in train_set.values():
        train_chords.update(s['distribution'])
    for s in test_set.values():
        test_chords.update(s['distribution'])

    all_chords_set = set(train_chords.keys()) | set(test_chords.keys())
    train_chord_total = sum(train_chords.values()) or 1
    test_chord_total = sum(test_chords.values()) or 1

    lines = []
    lines.append(f"# Distribution Report: {args.dataset_name}")
    lines.append(f"# Seed: {args.seed}, Test ratio: {args.test_ratio}")
    lines.append(f"# Train: {len(train_set)} songs, Test: {len(test_set)} songs")
    lines.append(f"# Total songs with audio: {len(all_songs)}")
    lines.append("")

    # Section 1: Genre balance
    lines.append("## Genre Balance")
    lines.append("genre,train_count,test_count,train_pct,test_pct")
    all_genres = sorted(set(list(train_genres.keys()) + list(test_genres.keys())))
    for g in all_genres:
        tr = train_genres.get(g, 0)
        te = test_genres.get(g, 0)
        tr_pct = round(100 * tr / len(train_set), 1) if train_set else 0
        te_pct = round(100 * te / len(test_set), 1) if test_set else 0
        lines.append(f"{g},{tr},{te},{tr_pct},{te_pct}")
    lines.append("")

    # Section 2: Quality family balance
    lines.append("## Quality Family Balance (% of duration)")
    lines.append("quality_family,train_pct,test_pct,diff")
    for fam in QUALITY_FAMILY_NAMES:
        tr_pct = round(100 * train_qf[fam] / train_qf_total, 2)
        te_pct = round(100 * test_qf[fam] / test_qf_total, 2)
        diff = round(te_pct - tr_pct, 2)
        if tr_pct > 0.05 or te_pct > 0.05:
            lines.append(f"{fam},{tr_pct},{te_pct},{diff:+.2f}")
    lines.append("")

    # Section 3: Full chord distribution (sorted by total duration)
    lines.append("## Full Chord Distribution (% of duration)")
    lines.append("chord,train_pct,test_pct,diff")
    sorted_chords = sorted(all_chords_set,
                           key=lambda c: -(train_chords.get(c, 0) + test_chords.get(c, 0)))
    for chord in sorted_chords:
        tr_pct = round(100 * train_chords.get(chord, 0) / train_chord_total, 3)
        te_pct = round(100 * test_chords.get(chord, 0) / test_chord_total, 3)
        diff = round(te_pct - tr_pct, 3)
        lines.append(f"{chord},{tr_pct},{te_pct},{diff:+.3f}")
    lines.append("")

    # Section 4: Per-song details
    lines.append("## Per-Song Details")
    lines.append("song,split,dataset,genre,dominant_chord,dominant_quality_family,duration_s")
    for name, info in sorted(train_set.items()):
        dur = round(sum(d for d in info['distribution'].values()), 1)
        lines.append(f"{name},train,{info['source_dataset']},{info['genre']},"
                     f"{info['dominant_class']},{info['dominant_quality_family']},{dur}")
    for name, info in sorted(test_set.items()):
        dur = round(sum(d for d in info['distribution'].values()), 1)
        lines.append(f"{name},test,{info['source_dataset']},{info['genre']},"
                     f"{info['dominant_class']},{info['dominant_quality_family']},{dur}")

    report_content = "\n".join(lines) + "\n"

    if dry_run:
        print(f"Would save distribution report to: {report_path}")
    else:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"Saved distribution report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create balanced train/test datasets from .lab files')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--lab_dir',
                             help='Directory containing .lab annotation files')
    input_group.add_argument('--manifest',
                             help='Text file with one .lab path per line (sources .lab from VM datasets)')
    parser.add_argument('--data_root', required=True,
                        help='Root directory of all datasets (e.g. /home/daniel.melo/datasets)')
    parser.add_argument('--output_dir', default=None,
                        help='Directory for output datasets (default: same as data_root). '
                             'E.g. /home/daniel.melo/datasets/personalized_datasets')
    parser.add_argument('--dataset_name', required=True,
                        help='Base name for the new dataset (creates {name}_train and {name}_test)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Fraction of songs for test set (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--symlink', action='store_true',
                        help='Create symlinks instead of copying audio files')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only show what would be done, do not copy files')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Restrict audio search to specific datasets')
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else data_root
    search_datasets = args.datasets or SEARCH_DATASETS

    # --- Build the list of (lab_filename, lab_path, dataset_hint) ---
    if args.manifest:
        manifest_entries = parse_manifest(args.manifest)
        print(f"Read {len(manifest_entries)} entries from manifest {args.manifest}")

        lab_entries = []
        lab_missing_on_vm = []
        for lab_filename, dataset_hint in manifest_entries:
            ds, lab_path = find_lab_in_datasets(
                data_root, lab_filename, dataset_hint, search_datasets)
            if lab_path:
                lab_entries.append((lab_filename, lab_path, ds))
            else:
                lab_missing_on_vm.append(lab_filename)

        if lab_missing_on_vm:
            print(f"\nWARNING: {len(lab_missing_on_vm)} .lab files not found in VM datasets:")
            for m in lab_missing_on_vm:
                print(f"  - {m}")
            print()

        print(f"Found {len(lab_entries)}/{len(manifest_entries)} .lab files on VM")
    else:
        lab_dir = os.path.abspath(args.lab_dir)
        lab_files = sorted([f for f in os.listdir(lab_dir)
                            if f.lower().endswith('.lab')])
        if not lab_files:
            print(f"ERROR: No .lab files found in {lab_dir}")
            sys.exit(1)
        lab_entries = [(f, os.path.join(lab_dir, f), None) for f in lab_files]
        print(f"Found {len(lab_entries)} .lab files in {lab_dir}")

    print(f"Searching audio in: {', '.join(search_datasets)}")
    print()

    # --- Phase 1: Parse .lab files and search for audio ---
    songs = []
    missing = []
    dataset_counts = Counter()

    for lab_filename, lab_path, dataset_hint in lab_entries:
        lab_stem = os.path.splitext(lab_filename)[0]

        segments = parse_lab_file(lab_path)
        distribution = compute_class_distribution(segments)
        dominant = get_dominant_class(distribution)

        if dataset_hint:
            audio_dir = os.path.join(data_root, dataset_hint, 'audio')
            audio_file = find_audio_in_dataset(audio_dir, lab_stem)
            src_dataset = dataset_hint if audio_file else None
            if not audio_file:
                src_dataset, audio_file = search_audio_across_datasets(
                    data_root, lab_stem, search_datasets)
        else:
            src_dataset, audio_file = search_audio_across_datasets(
                data_root, lab_stem, search_datasets)

        if audio_file:
            quality_profile = compute_quality_profile(segments)
            dom_qf = get_dominant_quality_family(quality_profile)
            genre = GENRE_GROUPS.get(src_dataset, 'unknown')
            songs.append({
                'lab_stem': lab_stem,
                'lab_file': lab_filename,
                'lab_path': lab_path,
                'segments': segments,
                'distribution': distribution,
                'dominant_class': dominant,
                'quality_profile': quality_profile,
                'dominant_quality_family': dom_qf,
                'genre': genre,
                'source_dataset': src_dataset,
                'audio_file': audio_file,
                'audio_path': os.path.join(data_root, src_dataset, 'audio', audio_file),
            })
            dataset_counts[src_dataset] += 1
        else:
            missing.append(lab_stem)

    print(f"=== Audio Search Results ===")
    print(f"Found: {len(songs)}/{len(lab_entries)} songs")
    for ds, count in sorted(dataset_counts.items()):
        print(f"  {ds}: {count} songs")
    if missing:
        print(f"\nMissing audio for {len(missing)} songs:")
        for m in missing:
            print(f"  - {m}")
    print()

    if not songs:
        print("ERROR: No songs found with matching audio. Aborting.")
        sys.exit(1)

    # --- Phase 2: Quality family distribution ---
    total_dist = Counter()
    for s in songs:
        total_dist.update(s['distribution'])

    print("=== Overall Chord Distribution (duration-weighted, top 15) ===")
    print(format_distribution(total_dist))
    print()

    total_qf = Counter()
    for s in songs:
        for fam, pct in s['quality_profile'].items():
            total_qf[fam] += pct * sum(d for d in s['distribution'].values())

    print("=== Quality Family Distribution ===")
    qf_total = sum(total_qf.values())
    for fam in QUALITY_FAMILY_NAMES:
        if total_qf[fam] > 0:
            print(f"  {fam:>10s}: {100*total_qf[fam]/qf_total:5.1f}%")

    genre_counts = Counter(s['genre'] for s in songs)
    print(f"\n=== Genre Groups ===")
    for g, c in sorted(genre_counts.items()):
        print(f"  {g}: {c} songs")
    print()

    # --- Phase 3: Distribution-based stratified split ---
    train_names, test_names = stratified_split(
        songs, args.test_ratio, args.seed)

    train_set = {s['lab_stem']: s for s in songs if s['lab_stem'] in set(train_names)}
    test_set = {s['lab_stem']: s for s in songs if s['lab_stem'] in set(test_names)}

    # Quality family comparison
    def aggregate_qf(song_set):
        agg = Counter()
        for s in song_set.values():
            for fam, pct in s['quality_profile'].items():
                agg[fam] += pct * sum(d for d in s['distribution'].values())
        return agg

    train_qf = aggregate_qf(train_set)
    test_qf = aggregate_qf(test_set)
    train_qf_total = sum(train_qf.values())
    test_qf_total = sum(test_qf.values())

    train_genres = Counter(s['genre'] for s in train_set.values())
    test_genres = Counter(s['genre'] for s in test_set.values())

    print(f"=== Split: {len(train_set)} train / {len(test_set)} test ===")

    print(f"\n  Genre balance:")
    print(f"  {'Genre':<12s} {'Train':>8s} {'Test':>8s} {'Train%':>8s} {'Test%':>8s}")
    for g in sorted(genre_counts.keys()):
        tr = train_genres.get(g, 0)
        te = test_genres.get(g, 0)
        tr_pct = 100 * tr / len(train_set) if train_set else 0
        te_pct = 100 * te / len(test_set) if test_set else 0
        print(f"  {g:<12s} {tr:>8d} {te:>8d} {tr_pct:>7.1f}% {te_pct:>7.1f}%")

    print(f"\n  Quality family balance (% of duration):")
    print(f"  {'Family':<12s} {'Train%':>8s} {'Test%':>8s} {'Diff':>8s}")
    for fam in QUALITY_FAMILY_NAMES:
        tr_pct = 100 * train_qf[fam] / train_qf_total if train_qf_total else 0
        te_pct = 100 * test_qf[fam] / test_qf_total if test_qf_total else 0
        diff = te_pct - tr_pct
        if tr_pct > 0.1 or te_pct > 0.1:
            print(f"  {fam:<12s} {tr_pct:>7.1f}% {te_pct:>7.1f}% {diff:>+7.1f}%")
    print()

    # --- Phase 4: Create dataset folders ---
    train_dir = os.path.join(output_dir, f"{args.dataset_name}_train")
    test_dir = os.path.join(output_dir, f"{args.dataset_name}_test")

    for split_name, split_dir, split_songs in [
        ('train', train_dir, train_set),
        ('test', test_dir, test_set),
    ]:
        audio_out = os.path.join(split_dir, 'audio')
        annot_out = os.path.join(split_dir, 'annotations')

        if args.dry_run:
            print(f"Would create {split_name}:")
            print(f"  {audio_out}  ({len(split_songs)} files)")
            print(f"  {annot_out}  ({len(split_songs)} files)")
        else:
            os.makedirs(audio_out, exist_ok=True)
            os.makedirs(annot_out, exist_ok=True)

            for name, info in sorted(split_songs.items()):
                src_audio = info['audio_path']
                dst_audio = os.path.join(audio_out, info['audio_file'])
                copy_or_link(src_audio, dst_audio, use_symlink=args.symlink)

                src_lab = info['lab_path']
                dst_lab = os.path.join(annot_out, info['lab_file'])
                shutil.copy2(src_lab, dst_lab)

            manifest = {
                'dataset_name': f"{args.dataset_name}_{split_name}",
                'split': split_name,
                'seed': args.seed,
                'test_ratio': args.test_ratio,
                'num_songs': len(split_songs),
                'songs': [
                    {
                        'name': name,
                        'source_dataset': info['source_dataset'],
                        'genre': info['genre'],
                        'audio_file': info['audio_file'],
                        'lab_file': info['lab_file'],
                        'dominant_class': info['dominant_class'],
                        'dominant_quality_family': info['dominant_quality_family'],
                    }
                    for name, info in sorted(split_songs.items())
                ],
            }
            manifest_path = os.path.join(split_dir, 'manifest.json')
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            print(f"Created {split_name}: {len(split_songs)} songs in {split_dir}")

    # --- Phase 5: Save distribution report ---
    report_path = os.path.join(
        output_dir if args.dry_run else train_dir,
        f"{args.dataset_name}_distribution_report.csv")
    save_distribution_report(
        report_path, songs, train_set, test_set,
        train_qf, test_qf, train_genres, test_genres,
        args, dry_run=args.dry_run)

    if not args.dry_run:
        print()
        print("=== Done! ===")

    # --- Phase 6: Registration instructions ---
    train_name = f"{args.dataset_name}_train"
    test_name = f"{args.dataset_name}_test"

    print()
    print("=" * 60)
    print("NEXT STEPS - Register the new datasets in the project:")
    print("=" * 60)
    print()
    print(f"1. In utils/preprocess.py, add to generic_datasets list:")
    print(f"   '{train_name}', '{test_name}'")
    print()
    print(f"2. In run_config.yaml, add to experiment.dataset_names (for training):")
    print(f"   - {train_name}")
    print()
    print(f"3. For inference testing, use:")
    print(f"   --test_dataset {test_name}")
    print()
    print(f"4. Run preprocessing:")
    print(f"   python scripts/preprocess_datasets.py  (will pick up new datasets from config)")
    print()


if __name__ == '__main__':
    main()
