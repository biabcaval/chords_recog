#!/usr/bin/env python3
"""
Compute stratified k-fold assignments based on chord class distribution.

Produces a fold_assignments.json file that audio_dataset.py uses to split
songs into balanced folds instead of the default alphabetical contiguous blocks.

Usage (on the VM where preprocessed .pt files live):

    ~/miniconda3/envs/BTC_py310/bin/python scripts/compute_stratified_folds.py \
        --data_root ~/datasets \
        --datasets billboard dj_avan_songbook1 dj_avan_songbook2 jaah queen robbiewilliams rwc \
        --n_splits 5 --seed 42

The output file is written to <data_root>/fold_assignments.json.
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: torch not available. Use the correct conda env.")
    sys.exit(1)

try:
    from sortedcontainers import SortedList
except ImportError:
    SortedList = lambda x: sorted(x)


MP3_STRING = "22050_10.0_5.0"
FEATURE_STRING = "cqt_252_36_2048"


def resolve_dataset_path(data_root, dataset_name):
    voca_name = dataset_name + "_voca"
    path = os.path.join(data_root, "result_decomposed", voca_name, MP3_STRING, FEATURE_STRING)
    if os.path.isdir(path):
        return path
    path = os.path.join(data_root, "result", voca_name, MP3_STRING, FEATURE_STRING)
    if os.path.isdir(path):
        return path
    return None


def collect_songs(data_root, dataset_names):
    """Collect song paths (mirrors audio_dataset.get_paths_voca)."""
    temp = {}
    used_song_names = []

    for name in dataset_names:
        ds_path = resolve_dataset_path(data_root, name)
        if ds_path is None:
            print(f"  WARNING: dataset '{name}' not found, skipping")
            continue
        for song_name in os.listdir(ds_path):
            song_dir = os.path.join(ds_path, song_name)
            if not os.path.isdir(song_dir):
                continue
            pts = [os.path.join(song_dir, f) for f in os.listdir(song_dir) if f.endswith(".pt")]
            if pts:
                used_song_names.append(song_name)
            temp[song_name] = pts

    song_names = SortedList(used_song_names)
    return song_names, temp


def compute_song_histograms(song_names, temp):
    """Load non-augmented .pt files and count chord labels per unique song."""
    unique_songs = sorted(set(song_names))
    histograms = {}
    all_labels_set = set()
    total = len(unique_songs)

    for i, song in enumerate(unique_songs):
        pts = temp.get(song, [])
        non_aug = [p for p in pts if "1.00_0" in os.path.basename(p)]
        counter = Counter()

        for pt_path in non_aug:
            try:
                data = torch.load(pt_path, weights_only=False)
            except Exception:
                continue

            labels = data.get("original_chord_labels") or data.get("original_chords")
            if labels is None:
                chord_ids = data.get("chord")
                if chord_ids is not None:
                    labels = [str(c) for c in chord_ids]
            if labels:
                counter.update(labels)

        histograms[song] = counter
        all_labels_set.update(counter.keys())

        if (i + 1) % 100 == 0 or i + 1 == total:
            print(f"  [{i+1}/{total}] songs processed", flush=True)

    all_labels = sorted(all_labels_set)
    return histograms, all_labels


def histogram_to_array(counter, label_list):
    return np.array([counter.get(l, 0) for l in label_list], dtype=np.float64)


def compute_stratified_folds(song_names, histograms, label_list, n_splits, seed):
    """Greedy distribution matching with capacity + frame balance."""
    rng = np.random.RandomState(seed)

    unique_songs = sorted(set(song_names))
    rng.shuffle(unique_songs)

    song_arrays = {s: histogram_to_array(histograms[s], label_list) for s in unique_songs}
    global_hist = np.sum(list(song_arrays.values()), axis=0)
    global_pct = global_hist / (global_hist.sum() + 1e-10)
    frame_target = global_hist.sum() / n_splits
    max_per_fold = -(-len(unique_songs) // n_splits)

    fold_hists = [np.zeros_like(global_hist) for _ in range(n_splits)]
    fold_counts = [0] * n_splits
    assignments = {}

    for song in unique_songs:
        h = song_arrays[song]
        best_fold = 0
        best_score = float("inf")
        for f in range(n_splits):
            if fold_counts[f] >= max_per_fold:
                continue
            candidate = fold_hists[f] + h
            cand_total = candidate.sum()
            cand_pct = candidate / (cand_total + 1e-10)
            dist_score = np.sum((cand_pct - global_pct) ** 2)
            frame_score = ((cand_total - frame_target) / (frame_target + 1e-10)) ** 2
            score = dist_score + frame_score
            if score < best_score:
                best_score = score
                best_fold = f
        assignments[song] = best_fold
        fold_hists[best_fold] += h
        fold_counts[best_fold] += 1

    return assignments, fold_hists, fold_counts


def print_report(assignments, fold_hists, fold_counts, label_list, n_splits):
    global_raw = np.sum(fold_hists, axis=0)
    global_total = global_raw.sum()
    global_pct = 100.0 * global_raw / (global_total + 1e-10)
    top_indices = np.argsort(global_raw)[-10:][::-1]

    print(f"\n{'Fold':<6} {'Songs':>6} {'Frames':>10}  ", end="")
    print("  ".join(f"{label_list[i]:>8}" for i in top_indices))

    for k in range(n_splits):
        total = fold_hists[k].sum()
        pct = 100.0 * fold_hists[k] / (total + 1e-10)
        row = f"  {k:<4} {fold_counts[k]:>6} {int(total):>10}  "
        row += "  ".join(f"{pct[i]:>7.1f}%" for i in top_indices)
        print(row)

    frames = [int(fold_hists[k].sum()) for k in range(n_splits)]
    print(f"\nFrames ratio (max/min): {max(frames)/(min(frames)+1):.2f}")

    chi2_list = []
    for k in range(n_splits):
        fold_pct = 100.0 * fold_hists[k] / (fold_hists[k].sum() + 1e-10)
        chi2 = np.sum((fold_pct - global_pct) ** 2 / (global_pct + 1e-10))
        chi2_list.append(chi2)
    print(f"Chi-squared medio: {np.mean(chi2_list):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute stratified k-fold assignments by chord distribution")
    parser.add_argument("--data_root", required=True,
                        help="Root datasets directory (e.g. ~/datasets)")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Dataset names (e.g. billboard jaah rwc)")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="Output path (default: <data_root>/fold_assignments.json)")
    args = parser.parse_args()

    data_root = os.path.expanduser(args.data_root)
    output_path = args.output or os.path.join(data_root, "fold_assignments.json")

    print("Collecting songs...")
    song_names, temp = collect_songs(data_root, args.datasets)
    print(f"Total song entries: {len(song_names)}  (unique: {len(set(song_names))})")

    print("\nComputing chord histograms...")
    histograms, label_list = compute_song_histograms(song_names, temp)
    print(f"Chord vocabulary: {len(label_list)} classes")

    print("\nComputing stratified fold assignments...")
    assignments, fold_hists, fold_counts = compute_stratified_folds(
        song_names, histograms, label_list, args.n_splits, args.seed)

    print_report(assignments, fold_hists, fold_counts, label_list, args.n_splits)

    result = {
        "n_splits": args.n_splits,
        "seed": args.seed,
        "datasets": args.datasets,
        "total_songs": len(set(song_names)),
        "chord_vocab_size": len(label_list),
        "songs": assignments,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved fold assignments to: {output_path}")


if __name__ == "__main__":
    main()
