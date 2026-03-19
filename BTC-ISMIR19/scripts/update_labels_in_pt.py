#!/usr/bin/env python
# encoding: utf-8
"""
Update chord labels in existing preprocessed .pt files without recomputing
CQT features.  This is ~20x faster than full reprocessing.

Also handles dataset renames (dj_avan -> dj_avan_songbook1 / dj_avan_songbook2).

After updating the Step-1 .pt files, Step 2 (decomposition to 9 components)
is re-run automatically unless --skip_decompose is passed.

Usage:
    # Update all datasets:
    python scripts/update_labels_in_pt.py \\
        --data_root /home/daniel.melo/datasets --config run_config.yaml

    # Update specific datasets only:
    python scripts/update_labels_in_pt.py \\
        --data_root /home/daniel.melo/datasets --datasets billboard rwc
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.chords import Chords
from utils.hparams import HParams
from utils.preprocess import FeatureTypes

# ---------------------------------------------------------------------------
# Dataset mapping: new_name -> (source_pt_name, annotations_subdir, filter_fn)
#
# For most datasets source_pt_name == new_name (update in place).
# For the Djavan split the source is the old "dj_avan" directory, and a filter
# selects only the songs belonging to each songbook.
# ---------------------------------------------------------------------------
DATASET_MAPPING = {
    'billboard':         ('billboard',     'billboard/annotations',         None),
    'jaah':              ('jaah',          'jaah/annotations',              None),
    'rwc':               ('rwc',           'rwc/annotations',               None),
    'queen':             ('queen',         'queen/annotations',             None),
    'robbiewilliams':    ('robbiewilliams','robbiewilliams/annotations',    None),
    'dj_avan_songbook1': ('dj_avan',       'dj_avan_songbook1/annotations', lambda n: 'Songbook1' in n),
    'dj_avan_songbook2': ('dj_avan',       'dj_avan_songbook2/annotations', lambda n: 'Songbook2' in n),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_pt_filename(filename):
    """Extract (stretch_factor, shift_factor) from a .pt filename.

    Filename format: ``{stretch:.2f}_{shift:d}_{idx:d}.pt``
    Example: ``1.00_3_5.pt`` -> (1.0, 3)
    """
    parts = Path(filename).stem.split('_')
    return float(parts[0]), int(parts[1])


def parse_etc_field(etc_str):
    """Extract (start_sec, end_sec) from the ``etc`` field inside a .pt."""
    parts = etc_str.split('_')
    return float(parts[0]), float(parts[1])


def find_lab_for_song(song_name, annotations_dir):
    """Find the .lab file that matches *song_name* (case-insensitive)."""
    exact = os.path.join(annotations_dir, song_name + '.lab')
    if os.path.exists(exact):
        return exact

    norm = song_name.lower().replace(' ', '_')
    for f in os.listdir(annotations_dir):
        if f.lower().endswith('.lab'):
            if os.path.splitext(f)[0].lower().replace(' ', '_') == norm:
                return os.path.join(annotations_dir, f)
    return None


# ---------------------------------------------------------------------------
# Core: recompute chord labels for one segment
# ---------------------------------------------------------------------------

def recompute_labels(chord_info, start_sec, stretch_factor, shift_factor,
                     time_interval, n_frames):
    """Re-derive chord/root/quality/bass lists for a time window.

    Mirrors the inner loop of ``Preprocess.generate_labels_features_voca``
    but skips CQT computation.
    """
    ci = chord_info.copy()
    ci['start'] = ci['start'] * (1.0 / stretch_factor)
    ci['end']   = ci['end']   * (1.0 / stretch_factor)

    chords, roots, qualities, basses = [], [], [], []
    cur = start_sec

    for _ in range(n_frames):
        try:
            avail = ci.loc[(ci['start'] <= cur) & (ci['end'] > cur + time_interval)]
            if len(avail) == 0:
                avail = ci.loc[
                    ((ci['start'] >= cur) & (ci['start'] <= cur + time_interval)) |
                    ((ci['end']   >= cur) & (ci['end']   <= cur + time_interval))
                ]

            if len(avail) == 1:
                chord   = avail['chord_id'].iloc[0]
                root    = avail['root'].iloc[0]
                quality = avail['quality'].iloc[0]
                bass    = avail['bass'].iloc[0]
            elif len(avail) > 1:
                avail = avail.copy()
                avail['max_start']    = avail['start'].clip(lower=cur)
                avail['min_end']      = avail['end'].clip(upper=cur + time_interval)
                avail['chord_length'] = avail['min_end'] - avail['max_start']
                idx = avail['chord_length'].idxmax()
                chord   = avail.loc[idx, 'chord_id']
                root    = avail.loc[idx, 'root']
                quality = avail.loc[idx, 'quality']
                bass    = avail.loc[idx, 'bass']
            else:
                chord, root, quality, bass = 169, 12, 14, 12
        except Exception:
            chord, root, quality, bass = 169, 12, 14, 12

        if chord != 169 and chord != 168:
            chord = (chord + shift_factor * 14) % 168
        if root != 12:
            root = (root + shift_factor) % 12
        if bass != 12:
            bass = (bass + shift_factor) % 12

        chords.append(chord)
        roots.append(root)
        qualities.append(quality)
        basses.append(bass)
        cur += time_interval

    return chords, roots, qualities, basses


# ---------------------------------------------------------------------------
# Per-song worker (runs in child process when num_workers > 1)
# ---------------------------------------------------------------------------

def _process_song(song_dir_str, ann_dir, target_dir, time_interval):
    """Process all .pt files for a single song. Returns (updated, skipped)."""
    song_dir = Path(song_dir_str)
    chord_class = Chords()

    lab_path = find_lab_for_song(song_dir.name, ann_dir)
    if lab_path is None:
        return 0, len(list(song_dir.glob('*.pt')))

    try:
        chord_info = chord_class.get_converted_chord_voca(lab_path)
    except Exception as e:
        print(f"    Error reading {lab_path}: {e}")
        return 0, len(list(song_dir.glob('*.pt')))

    out_song_dir = Path(target_dir) / song_dir.name
    out_song_dir.mkdir(parents=True, exist_ok=True)

    updated = 0
    skipped = 0

    for pt_file in sorted(song_dir.glob('*.pt')):
        try:
            data = torch.load(pt_file, map_location='cpu', weights_only=False)

            etc = data.get('etc', '')
            if not etc or '_' not in etc:
                skipped += 1
                continue

            start_sec, _ = parse_etc_field(etc)
            stretch, shift = parse_pt_filename(pt_file.name)
            n_frames = len(data['chord'])

            ch, ro, qu, ba = recompute_labels(
                chord_info, start_sec, stretch, shift,
                time_interval, n_frames,
            )

            data['chord']   = ch
            data['root']    = ro
            data['quality'] = qu
            data['bass']    = ba

            torch.save(data, out_song_dir / pt_file.name)
            updated += 1
        except Exception as e:
            print(f"    Error {pt_file.name}: {e}")
            skipped += 1

    return updated, skipped


# ---------------------------------------------------------------------------
# Per-dataset driver
# ---------------------------------------------------------------------------

def update_dataset(data_root, source_name, target_name,
                   annotations_subdir, song_filter,
                   mp3_string, feature_string, time_interval,
                   num_workers=1):
    """Update labels for every .pt in one dataset."""

    source_dir = os.path.join(data_root, 'result',
                              source_name + '_voca', mp3_string, feature_string)
    target_dir = os.path.join(data_root, 'result',
                              target_name + '_voca', mp3_string, feature_string)
    ann_dir    = os.path.join(data_root, annotations_subdir)

    if not os.path.exists(source_dir):
        print(f"  Source not found: {source_dir}")
        return 0, 0
    if not os.path.exists(ann_dir):
        print(f"  Annotations not found: {ann_dir}")
        return 0, 0

    song_dirs = sorted([d for d in Path(source_dir).iterdir() if d.is_dir()])
    if song_filter:
        song_dirs = [d for d in song_dirs if song_filter(d.name)]

    if not song_dirs:
        print(f"  No songs found in {source_dir}")
        return 0, 0

    total_updated = 0
    total_skipped = 0

    if num_workers <= 1:
        for song_dir in tqdm(song_dirs, desc=f"  {target_name}"):
            up, sk = _process_song(str(song_dir), ann_dir, target_dir, time_interval)
            total_updated += up
            total_skipped += sk
    else:
        workers = min(num_workers, len(song_dirs))
        print(f"  Using {workers} parallel workers for {len(song_dirs)} songs")
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for song_dir in song_dirs:
                fut = pool.submit(_process_song, str(song_dir), ann_dir,
                                  target_dir, time_interval)
                futures[fut] = song_dir.name

            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"  {target_name}"):
                up, sk = fut.result()
                total_updated += up
                total_skipped += sk

    return total_updated, total_skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Update chord labels in .pt files (keeps CQT features intact)'
    )
    parser.add_argument('--config', type=str, default='run_config.yaml')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory with datasets (default: from config)')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=list(DATASET_MAPPING.keys()),
                        help='Datasets to update (default: all)')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                        help='Parallel workers per dataset (default: all CPUs)')
    parser.add_argument('--skip_decompose', action='store_true',
                        help='Skip Step 2 (decomposition to 9 components)')

    args = parser.parse_args()

    config = HParams.load(args.config)
    data_root = args.data_root or config.path['root_path']

    mp3_cfg = config.mp3
    feat_cfg = config.feature
    mp3_string = "%d_%.1f_%.1f" % (mp3_cfg['song_hz'], mp3_cfg['inst_len'],
                                    mp3_cfg['skip_interval'])
    feature_string = "%s_%d_%d_%d" % (FeatureTypes.cqt.value, feat_cfg['n_bins'],
                                       feat_cfg['bins_per_octave'], feat_cfg['hop_length'])
    time_interval = feat_cfg['hop_length'] / mp3_cfg['song_hz']

    print("=" * 70)
    print("STEP 1: Update labels in .pt files (CQT features preserved)")
    print("=" * 70)
    print(f"Data root : {data_root}")
    print(f"Datasets  : {args.datasets}")
    print(f"Workers   : {args.num_workers}")
    print(f"Config    : {mp3_string} / {feature_string}")
    print()

    total_updated = 0
    total_skipped = 0

    for ds in args.datasets:
        if ds not in DATASET_MAPPING:
            print(f"  Unknown dataset: {ds}")
            continue

        src, ann_sub, filt = DATASET_MAPPING[ds]
        print(f"\n{ds} (source: {src})...")

        up, sk = update_dataset(
            data_root, src, ds, ann_sub, filt,
            mp3_string, feature_string, time_interval,
            num_workers=args.num_workers,
        )
        print(f"  Updated: {up}  |  Skipped: {sk}")
        total_updated += up
        total_skipped += sk

    print(f"\n{'=' * 70}")
    print(f"Step 1 done.  Updated: {total_updated}  |  Skipped: {total_skipped}")

    if args.skip_decompose:
        print("Step 2 skipped (--skip_decompose).")
        return

    # Step 2: re-run decomposition
    print(f"\n{'=' * 70}")
    print("STEP 2: Decompose to 9-component format")
    print("=" * 70)

    from scripts.preprocess_decomposed import decompose_preprocessed_data

    for ds in args.datasets:
        src_dir = os.path.join(data_root, 'result',
                               ds + '_voca', mp3_string, feature_string)
        out_dir = os.path.join(data_root, 'result_decomposed',
                               ds + '_voca', mp3_string, feature_string)
        if os.path.exists(src_dir):
            print(f"\n  Decomposing {ds}...")
            decompose_preprocessed_data(src_dir, out_dir, force=True)
        else:
            print(f"  Skipping {ds} — {src_dir} not found")

    print(f"\n{'=' * 70}")
    print("All done! Labels updated and decomposed data regenerated.")
    print("=" * 70)


if __name__ == '__main__':
    main()
