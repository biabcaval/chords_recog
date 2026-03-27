#!/usr/bin/env python
"""
Detailed frame-level evaluation: Accuracy, Precision, Recall, F1 and
Confusion Matrices for chord recognition, at Root and Quality granularity.

Compares inference .lab files against ground-truth .lab files by sampling
at a fixed frame rate (default 100 fps = 10 ms) and computing standard
classification metrics via scikit-learn.

Usage:
    python generate_detailed_metrics.py \
        --inference_dir ./inferences_decomposed/inference_chordformer_test_Dj2 \
        --gt_dir /home/daniel.melo/datasets/dj_avan_songbook2/annotations \
        --output_dir ./detailed_metrics \
        --prefix BiQuRoRwJaDj1_testDj2
"""

import os
import re
import argparse
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────────

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_TO_SHARP = {
    "Cb": "B", "Db": "C#", "Eb": "D#", "Fb": "E",
    "Gb": "F#", "Ab": "G#", "Bb": "A#",
}

QUALITY_CANON = {
    "maj": "maj", "min": "min", "dim": "dim", "aug": "aug",
    "sus2": "sus2", "sus4": "sus4",
    "7": "7", "maj7": "maj7", "min7": "min7", "dim7": "dim7",
    "hdim7": "hdim7", "minmaj7": "minmaj7", "aug7": "aug7",
    "maj6": "maj6", "min6": "min6",
    "9": "9", "maj9": "maj9", "min9": "min9",
    "sus": "sus4",
    "": "maj",
}

ROOT_ORDER = ["N"] + _NOTE_NAMES

# ───────────────────────────────────────────────────────────────────────
# .lab parsing (shared logic with generate_metrics_csv.py)
# ───────────────────────────────────────────────────────────────────────

def parse_lab_file(filepath):
    """Return (intervals np.array, labels list) from a .lab file."""
    intervals, labels = [], []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                intervals.append([float(parts[0]), float(parts[1])])
                labels.append(" ".join(parts[2:]))
    return np.array(intervals) if intervals else np.empty((0, 2)), labels


def normalize_filename(filename):
    return filename.lower().replace(" ", "_").replace("-", "_")


def build_gt_filename_map(gt_dir):
    gt_map = {}
    for f in glob.glob(os.path.join(gt_dir, "*.lab")):
        gt_map[normalize_filename(os.path.basename(f))] = f
    return gt_map


# ───────────────────────────────────────────────────────────────────────
# Chord label → (root, quality)
# ───────────────────────────────────────────────────────────────────────

def _normalize_root(root_str):
    """Normalize flats to sharps and validate."""
    root = _FLAT_TO_SHARP.get(root_str, root_str)
    if root in _NOTE_NAMES:
        return root
    return None


def _extract_quality(shorthand):
    """Map a mir_eval-style shorthand string to a canonical quality label."""
    sh = shorthand.split("/")[0]  # strip bass
    sh = re.sub(r"\([^)]*\)", "", sh)  # strip degree extensions in parens
    sh = sh.strip()
    if sh in QUALITY_CANON:
        return QUALITY_CANON[sh]
    for key in sorted(QUALITY_CANON, key=len, reverse=True):
        if sh.startswith(key) and key:
            return QUALITY_CANON[key]
    return sh if sh else "maj"


def chord_to_root_quality(label):
    """Parse a chord label into (root, quality).

    Returns ("N", "N") for no-chord / unknown.
    """
    if label in ("N", "X", ""):
        return "N", "N"
    if ":" not in label:
        root = _normalize_root(label.split("/")[0])
        return (root, "maj") if root else ("N", "N")
    root_str, rest = label.split(":", 1)
    root = _normalize_root(root_str)
    if root is None:
        return "N", "N"
    quality = _extract_quality(rest)
    return root, quality


# ───────────────────────────────────────────────────────────────────────
# Frame-level sampling
# ───────────────────────────────────────────────────────────────────────

def sample_labels_at_frames(intervals, labels, fps):
    """Sample chord labels at fixed frame rate across the annotation span."""
    if len(intervals) == 0:
        return np.array([]), []
    end_time = intervals[-1, 1]
    n_frames = int(np.ceil(end_time * fps))
    if n_frames == 0:
        return np.array([]), []
    times = np.arange(n_frames) / fps
    sampled = []
    idx = 0
    for t in times:
        while idx < len(intervals) - 1 and t >= intervals[idx, 1]:
            idx += 1
        if intervals[idx, 0] <= t < intervals[idx, 1]:
            sampled.append(labels[idx])
        else:
            sampled.append("N")
    return times, sampled


# ───────────────────────────────────────────────────────────────────────
# Metrics computation
# ───────────────────────────────────────────────────────────────────────

def compute_frame_metrics(inference_dir, gt_dir, fps=100):
    """Collect frame-level root and quality labels across all matched tracks.

    Returns (all_ref_roots, all_est_roots, all_ref_quals, all_est_quals,
             track_stats) or Nones if no tracks matched.
    """
    inference_files = glob.glob(os.path.join(inference_dir, "*.lab"))
    if not inference_files:
        return None

    gt_map = build_gt_filename_map(gt_dir)

    all_ref_roots, all_est_roots = [], []
    all_ref_quals, all_est_quals = [], []
    track_stats = []

    for inf_file in sorted(inference_files):
        track_id = os.path.basename(inf_file)
        norm_id = normalize_filename(track_id)

        gt_file = os.path.join(gt_dir, track_id)
        if not os.path.exists(gt_file):
            gt_file = gt_map.get(norm_id)
        if gt_file is None or not os.path.exists(gt_file):
            print(f"  [skip] no GT for {track_id}")
            continue

        try:
            ref_intervals, ref_labels = parse_lab_file(gt_file)
            est_intervals, est_labels = parse_lab_file(inf_file)
            if len(ref_intervals) == 0 or len(est_intervals) == 0:
                continue

            common_end = min(ref_intervals[-1, 1], est_intervals[-1, 1])
            if common_end <= 0:
                continue

            _, ref_sampled = sample_labels_at_frames(ref_intervals, ref_labels, fps)
            _, est_sampled = sample_labels_at_frames(est_intervals, est_labels, fps)

            n = min(len(ref_sampled), len(est_sampled))
            ref_sampled = ref_sampled[:n]
            est_sampled = est_sampled[:n]

            ref_rq = [chord_to_root_quality(l) for l in ref_sampled]
            est_rq = [chord_to_root_quality(l) for l in est_sampled]

            ref_r = [r for r, _ in ref_rq]
            est_r = [r for r, _ in est_rq]
            ref_q = [q for _, q in ref_rq]
            est_q = [q for _, q in est_rq]

            all_ref_roots.extend(ref_r)
            all_est_roots.extend(est_r)
            all_ref_quals.extend(ref_q)
            all_est_quals.extend(est_q)

            root_acc = accuracy_score(ref_r, est_r)
            qual_acc = accuracy_score(ref_q, est_q)
            track_stats.append({
                "track": os.path.splitext(track_id)[0],
                "frames": n,
                "duration_s": n / fps,
                "root_accuracy": root_acc,
                "quality_accuracy": qual_acc,
            })

        except Exception as e:
            print(f"  [error] {track_id}: {e}")
            continue

    if not all_ref_roots:
        return None

    return all_ref_roots, all_est_roots, all_ref_quals, all_est_quals, track_stats


def classification_metrics(y_true, y_pred, class_order=None):
    """Compute accuracy + per-class and averaged P/R/F1.

    Returns (accuracy, df_per_class, macro, micro, weighted) where
    macro/micro/weighted are dicts with keys precision, recall, f1.
    """
    labels = class_order
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    present = sorted(set(y_true) | set(y_pred))
    labels = [l for l in labels if l in present]

    acc = accuracy_score(y_true, y_pred)

    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    df = pd.DataFrame({
        "class": labels,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": sup.astype(int),
    }).set_index("class")

    def _avg(avg_type):
        pa, ra, fa, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=avg_type, zero_division=0
        )
        return {"precision": pa, "recall": ra, "f1": fa}

    return acc, df, _avg("macro"), _avg("micro"), _avg("weighted")


def build_confusion(y_true, y_pred, class_order=None):
    """Return a labeled confusion-matrix DataFrame."""
    labels = class_order
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    present = sorted(set(y_true) | set(y_pred))
    labels = [l for l in labels if l in present]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


# ───────────────────────────────────────────────────────────────────────
# Plotting
# ───────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm_df, title, output_path, figsize=None, annot=True):
    """Save a confusion-matrix heatmap as PNG."""
    n = len(cm_df)
    if figsize is None:
        side = max(6, n * 0.55)
        figsize = (side, side)
    fig, ax = plt.subplots(figsize=figsize)

    fmt = "d" if cm_df.values.max() < 1e6 else ".1e"
    annot_kws = {"size": max(6, 10 - n // 8)}

    sns.heatmap(
        cm_df, annot=annot and n <= 40, fmt=fmt, cmap="Blues",
        linewidths=0.4, ax=ax, annot_kws=annot_kws,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.tick_params(axis="both", labelsize=max(6, 10 - n // 10))
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_per_class_f1(df_per_class, title, output_path):
    """Horizontal bar chart of per-class F1 scores."""
    df_sorted = df_per_class.sort_values("f1", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df_sorted) * 0.35)))
    colors = plt.cm.RdYlGn(df_sorted["f1"].values)
    ax.barh(df_sorted.index, df_sorted["f1"], color=colors, edgecolor="gray", linewidth=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 Score", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax.text(row["f1"] + 0.01, i, f"{row['f1']:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Frame-level Accuracy / P / R / F1 and Confusion Matrices "
                    "for chord recognition, at Root and Quality granularity."
    )
    p.add_argument("--inference_dir", required=True, help="Dir with predicted .lab files")
    p.add_argument("--gt_dir", required=True, help="Dir with ground-truth .lab files")
    p.add_argument("--output_dir", default="./detailed_metrics", help="Output directory")
    p.add_argument("--prefix", default="detailed", help="Filename prefix for outputs")
    p.add_argument("--fps", type=int, default=100,
                   help="Sampling rate in frames per second (default 100 = 10 ms)")
    p.add_argument("--no_plots", action="store_true", help="Skip PNG generation")
    return p.parse_args()


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not os.path.isdir(args.inference_dir):
        raise FileNotFoundError(f"Inference dir not found: {args.inference_dir}")
    if not os.path.isdir(args.gt_dir):
        raise FileNotFoundError(f"GT dir not found: {args.gt_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"  Inference dir : {args.inference_dir}")
    print(f"  Ground-truth  : {args.gt_dir}")
    print(f"  FPS           : {args.fps}")
    print("=" * 70)

    result = compute_frame_metrics(args.inference_dir, args.gt_dir, fps=args.fps)
    if result is None:
        print("\nNo tracks matched.")
        return

    ref_roots, est_roots, ref_quals, est_quals, track_stats = result
    total_frames = len(ref_roots)
    n_tracks = len(track_stats)

    print(f"\nMatched {n_tracks} tracks  |  {total_frames:,} frames "
          f"({total_frames / args.fps:.1f} s)\n")

    # ── Per-track stats ──────────────────────────────────────────────
    df_tracks = pd.DataFrame(track_stats).set_index("track")
    track_path = os.path.join(args.output_dir, f"{args.prefix}_per_track.csv")
    df_tracks.to_csv(track_path)
    print(f"  Per-track stats     : {track_path}")

    # ── ROOT metrics ─────────────────────────────────────────────────
    root_acc, root_cls, root_macro, root_micro, root_wtd = classification_metrics(
        ref_roots, est_roots, class_order=ROOT_ORDER
    )
    root_cls_path = os.path.join(args.output_dir, f"{args.prefix}_root_per_class.csv")
    root_cls.to_csv(root_cls_path)
    print(f"  Root per-class      : {root_cls_path}")

    root_cm = build_confusion(ref_roots, est_roots, class_order=ROOT_ORDER)
    root_cm_path = os.path.join(args.output_dir, f"{args.prefix}_root_confusion_matrix.csv")
    root_cm.to_csv(root_cm_path)
    print(f"  Root confusion mat  : {root_cm_path}")

    if not args.no_plots:
        plot_confusion_matrix(
            root_cm, "Confusion Matrix — Root",
            os.path.join(args.output_dir, f"{args.prefix}_root_confusion_matrix.png"),
        )
        plot_per_class_f1(
            root_cls, "Per-class F1 — Root",
            os.path.join(args.output_dir, f"{args.prefix}_root_f1_bar.png"),
        )

    # ── QUALITY metrics ──────────────────────────────────────────────
    qual_acc, qual_cls, qual_macro, qual_micro, qual_wtd = classification_metrics(
        ref_quals, est_quals
    )
    qual_cls_path = os.path.join(args.output_dir, f"{args.prefix}_quality_per_class.csv")
    qual_cls.to_csv(qual_cls_path)
    print(f"  Quality per-class   : {qual_cls_path}")

    qual_cm = build_confusion(ref_quals, est_quals)
    qual_cm_path = os.path.join(args.output_dir, f"{args.prefix}_quality_confusion_matrix.csv")
    qual_cm.to_csv(qual_cm_path)
    print(f"  Quality confusion   : {qual_cm_path}")

    if not args.no_plots:
        plot_confusion_matrix(
            qual_cm, "Confusion Matrix — Quality",
            os.path.join(args.output_dir, f"{args.prefix}_quality_confusion_matrix.png"),
        )
        plot_per_class_f1(
            qual_cls, "Per-class F1 — Quality",
            os.path.join(args.output_dir, f"{args.prefix}_quality_f1_bar.png"),
        )

    # ── Summary ──────────────────────────────────────────────────────
    summary = {
        "num_tracks": n_tracks,
        "total_frames": total_frames,
        "total_duration_s": total_frames / args.fps,
        "root_accuracy": root_acc,
        "root_precision_macro": root_macro["precision"],
        "root_recall_macro": root_macro["recall"],
        "root_f1_macro": root_macro["f1"],
        "root_precision_micro": root_micro["precision"],
        "root_recall_micro": root_micro["recall"],
        "root_f1_micro": root_micro["f1"],
        "root_precision_weighted": root_wtd["precision"],
        "root_recall_weighted": root_wtd["recall"],
        "root_f1_weighted": root_wtd["f1"],
        "quality_accuracy": qual_acc,
        "quality_precision_macro": qual_macro["precision"],
        "quality_recall_macro": qual_macro["recall"],
        "quality_f1_macro": qual_macro["f1"],
        "quality_precision_micro": qual_micro["precision"],
        "quality_recall_micro": qual_micro["recall"],
        "quality_f1_micro": qual_micro["f1"],
        "quality_precision_weighted": qual_wtd["precision"],
        "quality_recall_weighted": qual_wtd["recall"],
        "quality_f1_weighted": qual_wtd["f1"],
    }
    df_summary = pd.DataFrame([summary])
    summary_path = os.path.join(args.output_dir, f"{args.prefix}_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"  Summary             : {summary_path}")

    # ── Console output ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ROOT")
    print(f"    Accuracy : {root_acc:.4f}")
    print(f"    Macro    : P={root_macro['precision']:.4f}  "
          f"R={root_macro['recall']:.4f}  F1={root_macro['f1']:.4f}")
    print(f"    Weighted : P={root_wtd['precision']:.4f}  "
          f"R={root_wtd['recall']:.4f}  F1={root_wtd['f1']:.4f}")
    print()
    print("  QUALITY")
    print(f"    Accuracy : {qual_acc:.4f}")
    print(f"    Macro    : P={qual_macro['precision']:.4f}  "
          f"R={qual_macro['recall']:.4f}  F1={qual_macro['f1']:.4f}")
    print(f"    Weighted : P={qual_wtd['precision']:.4f}  "
          f"R={qual_wtd['recall']:.4f}  F1={qual_wtd['f1']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
