#!/usr/bin/env python
"""
Generate evaluation metrics CSV by comparing inference .lab files against
ground-truth .lab files.

Usage:
    python generate_metrics_csv.py \
        --inference_dir ./inferences_decomposed/inference_chordformer_test_Rw \
        --gt_dir /home/daniel.melo/datasets/rwc/annotations \
        --output_dir ./metrics_results \
        --prefix chordformer_rwc
"""

import os
import re
import argparse
import glob
import numpy as np
import pandas as pd
import mir_eval
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Chord label normalization (mir_eval compatibility)
# ---------------------------------------------------------------------------

_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_NOTE_TO_IDX = {n: i for i, n in enumerate(_NOTE_NAMES)}
_FLAT_TO_SHARP = {
    'Cb': 'B', 'Db': 'C#', 'Eb': 'D#', 'Fb': 'E',
    'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
}
_SEMITONE_TO_DEGREE = {
    0: '1', 1: 'b2', 2: '2', 3: 'b3', 4: '3', 5: '4',
    6: 'b5', 7: '5', 8: 'b6', 9: '6', 10: 'b7', 11: '7',
}


def _absolute_bass_to_degree(root_str, bass_str):
    """Convert an absolute bass note name to a scale degree relative to root.

    Returns the degree string (e.g. 'b7') or None if either note is
    unrecognised.
    """
    r = _FLAT_TO_SHARP.get(root_str, root_str)
    b = _FLAT_TO_SHARP.get(bass_str, bass_str)
    r_idx = _NOTE_TO_IDX.get(r)
    b_idx = _NOTE_TO_IDX.get(b)
    if r_idx is None or b_idx is None:
        return None
    interval = (b_idx - r_idx) % 12
    return _SEMITONE_TO_DEGREE.get(interval)


def normalize_chord_for_mir_eval(label):
    """Rewrite a chord label so that mir_eval.chord can parse it.

    Fixes three classes of issues produced by ChordReassembler:
      1. Multiple parenthetical groups  e.g. sus4(b7)(9) -> sus4(b7,9)
      2. Unrecognised shorthands        e.g. aug7        -> aug(b7)
      3. Absolute bass note names       e.g. A:maj/G     -> A:maj/b7
    """
    if label in ('N', 'X', '') or ':' not in label:
        return label

    # Separate optional bass suffix  "/E", "/F#", …
    bass_suffix = ''
    slash_idx = label.rfind('/')
    if slash_idx > label.rfind(')'):
        bass_suffix = label[slash_idx:]
        main = label[:slash_idx]
    else:
        main = label

    # aug7 -> aug(b7)   (mir_eval doesn't know the "aug7" shorthand)
    main = re.sub(r':aug7\b', ':aug(b7)', main)

    # Merge multiple (...) groups into one comma-separated group
    groups = re.findall(r'\(([^)]+)\)', main)
    if len(groups) > 1:
        base = re.sub(r'\([^)]+\)', '', main)
        main = f"{base}({','.join(groups)})"

    # Convert absolute bass note to relative degree for mir_eval.
    # e.g. A:maj/G -> A:maj/b7,  E:7/D -> E:7/b7
    if bass_suffix:
        bass_note = bass_suffix[1:]  # strip leading '/'
        root_str = main.split(':')[0]
        degree = _absolute_bass_to_degree(root_str, bass_note)
        if degree is not None:
            bass_suffix = f"/{degree}"

    return main + bass_suffix


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def parse_lab_file(filepath):
    """Parse a .lab file and return intervals and labels."""
    intervals = []
    labels = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1])
                chord = ' '.join(parts[2:])
                intervals.append([start, end])
                labels.append(chord)
    return np.array(intervals), labels


def normalize_filename(filename):
    """Normalize filename for matching between inference and ground truth."""
    normalized = filename.lower()
    normalized = normalized.replace(' ', '_')
    normalized = normalized.replace('-', '_')
    return normalized


def build_gt_filename_map(ground_truth_dir):
    """Build a mapping from normalized filenames to actual ground truth paths."""
    gt_files = glob.glob(os.path.join(ground_truth_dir, "*.lab"))
    gt_map = {}
    for gt_file in gt_files:
        basename = os.path.basename(gt_file)
        normalized = normalize_filename(basename)
        gt_map[normalized] = gt_file
    return gt_map


# ---------------------------------------------------------------------------
# Metrics calculation
# ---------------------------------------------------------------------------

def calculate_metrics(inference_dir, ground_truth_dir):
    """Compare every .lab in inference_dir against its match in ground_truth_dir.

    Returns a dict of aggregated metrics (means, stds, WCSRs), or None if no
    tracks could be matched.
    """
    inference_files = glob.glob(os.path.join(inference_dir, "*.lab"))
    if not inference_files:
        return None, None

    gt_filename_map = build_gt_filename_map(ground_truth_dir)
    all_scores = defaultdict(list)
    track_names = []

    for inf_file in inference_files:
        track_id = os.path.basename(inf_file)
        normalized_track_id = normalize_filename(track_id)

        gt_file = os.path.join(ground_truth_dir, track_id)
        if not os.path.exists(gt_file):
            gt_file = gt_filename_map.get(normalized_track_id)

        if gt_file is None or not os.path.exists(gt_file):
            continue

        try:
            ref_intervals, ref_labels = parse_lab_file(gt_file)
            est_intervals, est_labels = parse_lab_file(inf_file)

            ref_labels = [normalize_chord_for_mir_eval(l) for l in ref_labels]
            est_labels = [normalize_chord_for_mir_eval(l) for l in est_labels]

            if len(ref_intervals) == 0 or len(est_intervals) == 0:
                continue

            scores = mir_eval.chord.evaluate(
                ref_intervals, ref_labels, est_intervals, est_labels
            )

            track_names.append(os.path.splitext(track_id)[0])
            for metric, value in scores.items():
                all_scores[metric].append(value)

            all_scores['num_predictions'].append(len(est_labels))
            all_scores['num_ground_truth'].append(len(ref_labels))
            all_scores['duration_seconds'].append(ref_intervals[-1][1])

            est_dur = est_intervals[-1][1]
            ref_dur = ref_intervals[-1][1]
            all_scores['pred_changes_per_min'].append(
                len(est_labels) / (est_dur / 60) if est_dur > 0 else 0
            )
            all_scores['gt_changes_per_min'].append(
                len(ref_labels) / (ref_dur / 60) if ref_dur > 0 else 0
            )

        except Exception as e:
            print(f"  Error processing {track_id}: {e}")
            continue

    if not all_scores:
        return None, None

    # --- Per-track DataFrame (detailed) ---
    per_track = {metric: values for metric, values in all_scores.items()}
    df_per_track = pd.DataFrame(per_track, index=track_names)

    # --- Aggregate summary ---
    mean_metrics = {k: np.mean(v) for k, v in all_scores.items()}
    std_metrics = {f"{k}_std": np.std(v) for k, v in all_scores.items()}

    wcsr_keys = ['root', 'majmin', 'thirds', 'triads', 'tetrads', 'sevenths', 'mirex']
    durations = np.array(all_scores['duration_seconds'])
    total_duration = np.sum(durations)

    wcsr = {}
    for m in wcsr_keys:
        if m in all_scores:
            s = np.array(all_scores[m])
            wcsr[f'{m}_wcsr'] = (np.sum(s * durations) / total_duration * 100
                                 if total_duration > 0 else 0)

    combined = {**mean_metrics, **std_metrics, **wcsr}
    combined['num_tracks'] = len(track_names)

    return df_per_track, combined


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

METRIC_ORDER = [
    'root', 'majmin', 'thirds', 'triads', 'tetrads', 'sevenths',
    'root_wcsr', 'majmin_wcsr', 'thirds_wcsr', 'triads_wcsr',
    'tetrads_wcsr', 'sevenths_wcsr', 'mirex_wcsr',
    'overseg', 'underseg', 'seg',
    'num_tracks', 'duration_seconds',
    'num_predictions', 'num_ground_truth',
    'pred_changes_per_min', 'gt_changes_per_min',
]

COLUMN_LABELS = {
    'root': 'Root', 'majmin': 'Maj/Min', 'thirds': 'Thirds',
    'triads': 'Triads', 'tetrads': 'Tetrads', 'sevenths': 'Sevenths',
    'mirex': 'MIREX',
    'root_wcsr': 'Root WCSR (%)', 'majmin_wcsr': 'Maj/Min WCSR (%)',
    'thirds_wcsr': 'Thirds WCSR (%)', 'triads_wcsr': 'Triads WCSR (%)',
    'tetrads_wcsr': 'Tetrads WCSR (%)', 'sevenths_wcsr': 'Sevenths WCSR (%)',
    'mirex_wcsr': 'MIREX WCSR (%)',
    'overseg': 'Over-seg', 'underseg': 'Under-seg', 'seg': 'Seg',
    'num_tracks': 'Tracks', 'duration_seconds': 'Avg Duration (s)',
    'num_predictions': 'Avg Preds', 'num_ground_truth': 'Avg GT',
    'pred_changes_per_min': 'Pred Chg/min', 'gt_changes_per_min': 'GT Chg/min',
}


def build_summary_row(combined, label):
    """Return a single-row DataFrame with ordered & renamed columns."""
    ordered = [m for m in METRIC_ORDER if m in combined]
    cols_with_std = []
    for m in ordered:
        cols_with_std.append(m)
        std_key = f"{m}_std"
        if std_key in combined:
            cols_with_std.append(std_key)

    row = {k: combined[k] for k in cols_with_std}
    df = pd.DataFrame([row], index=[label])

    rename = {}
    for c in df.columns:
        base = c.replace('_std', '')
        base_label = COLUMN_LABELS.get(base, base)
        if c.endswith('_std'):
            rename[c] = f"{base_label} (Std)"
        else:
            rename[c] = f"{base_label} (Mean)" if f"{c}_std" in combined else base_label
    df = df.rename(columns=rename)

    for col in df.columns:
        if 'Tracks' in col:
            df[col] = df[col].apply(lambda x: f"{x:.0f}")
        elif 'WCSR' in col:
            df[col] = df[col].apply(lambda x: f"{x:.2f}")
        elif any(k in col for k in ('Chg', 'Duration', 'Preds', 'GT')):
            df[col] = df[col].apply(lambda x: f"{x:.2f}")
        else:
            df[col] = df[col].apply(lambda x: f"{x:.4f}")

    return df


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare inference .lab files against ground-truth and "
                    "generate evaluation metrics CSVs."
    )
    parser.add_argument(
        "--inference_dir", type=str, required=True,
        help="Directory with predicted .lab files",
    )
    parser.add_argument(
        "--gt_dir", type=str, required=True,
        help="Directory with ground-truth .lab files",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./metrics_results",
        help="Where to save CSV results (default: ./metrics_results)",
    )
    parser.add_argument(
        "--prefix", type=str, default="metrics",
        help="Filename prefix for the CSVs (default: metrics)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.inference_dir):
        raise FileNotFoundError(f"Inference dir not found: {args.inference_dir}")
    if not os.path.isdir(args.gt_dir):
        raise FileNotFoundError(f"Ground-truth dir not found: {args.gt_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"Inference dir : {args.inference_dir}")
    print(f"Ground-truth  : {args.gt_dir}")
    print("=" * 70)

    df_per_track, combined = calculate_metrics(args.inference_dir, args.gt_dir)

    if combined is None:
        print("\nNo tracks matched between inference and ground-truth directories.")
        return

    exp_label = os.path.basename(args.inference_dir.rstrip("/\\"))

    # Per-track detailed CSV
    detailed_path = os.path.join(args.output_dir, f"{args.prefix}_per_track.csv")
    df_per_track.to_csv(detailed_path)
    print(f"\n✓ Per-track metrics : {detailed_path}")

    # Summary CSV
    df_summary = build_summary_row(combined, exp_label)
    summary_path = os.path.join(args.output_dir, f"{args.prefix}_summary.csv")
    df_summary.to_csv(summary_path)
    print(f"✓ Summary metrics   : {summary_path}")

    print(f"\n{df_summary.to_string()}")
    print(f"\nMatched {combined['num_tracks']:.0f} tracks.")
    print("=" * 70)


if __name__ == "__main__":
    main()
