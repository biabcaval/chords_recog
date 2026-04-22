#!/usr/bin/env python3
"""
Diagnostic script: compare chord-class distribution across k-folds.

Simulates two split strategies side-by-side:
  A) Current (alphabetical contiguous blocks via SortedList)
  B) Proposed (greedy distribution matching by chord histogram)

Run on the VM where preprocessed .pt files live:

    ~/miniconda3/envs/BTC_py310/bin/python scripts/analyze_fold_distribution.py \
        --data_root ~/datasets \
        --datasets billboard dj_avan_songbook1 dj_avan_songbook2 jaah queen robbiewilliams rwc \
        --n_splits 5 --seed 42 --top_n 15
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: torch not available. Use the correct conda env.")
    sys.exit(1)

try:
    from sortedcontainers import SortedList
except ImportError:
    # Minimal fallback: plain sorted list
    SortedList = lambda x: sorted(x)

# ── path resolution (mirrors audio_dataset.get_paths_voca) ──────────────

MP3_STRING = "22050_10.0_5.0"
FEATURE_STRING = "cqt_252_36_2048"


def resolve_dataset_path(data_root, dataset_name):
    """Try result_decomposed first, fallback to result."""
    voca_name = dataset_name + "_voca"
    path = os.path.join(data_root, "result_decomposed", voca_name, MP3_STRING, FEATURE_STRING)
    if os.path.isdir(path):
        return path
    path = os.path.join(data_root, "result", voca_name, MP3_STRING, FEATURE_STRING)
    if os.path.isdir(path):
        return path
    return None


def collect_songs(data_root, dataset_names):
    """Collect song paths exactly as get_paths_voca does.

    Returns:
        song_names: SortedList (may contain duplicates across datasets)
        temp: dict  song_name -> list of .pt paths
    """
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


# ── histogram computation ───────────────────────────────────────────────

def compute_song_histograms(song_names, temp):
    """Load non-augmented .pt files and count chord labels per unique song.

    Returns:
        histograms: dict  song_name -> Counter {chord_label: frame_count}
        all_labels: sorted list of all chord labels seen
    """
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
    """Convert a Counter to a numpy array aligned with label_list."""
    return np.array([counter.get(l, 0) for l in label_list], dtype=np.float64)


# ── split methods ───────────────────────────────────────────────────────

def split_alphabetical(song_names, n_splits):
    """Current method: contiguous blocks on SortedList."""
    total = len(song_names)
    quotient = total // n_splits
    remainder = total % n_splits
    boundaries = [0]
    for i in range(n_splits):
        size = quotient + (1 if i < remainder else 0)
        boundaries.append(boundaries[-1] + size)

    folds = {}
    for k in range(n_splits):
        for idx in range(boundaries[k], boundaries[k + 1]):
            folds[song_names[idx]] = k
    return folds


def split_stratified_greedy(song_names, histograms, label_list, n_splits, seed):
    """Proposed method: greedy distribution matching with capacity constraint.

    Each fold is capped at ceil(N/k) songs so sizes stay balanced.
    Scoring uses L2 distance from the target histogram (global/k),
    which keeps common classes well-balanced without over-penalizing
    rare ones the way chi-squared does.
    """
    rng = np.random.RandomState(seed)

    unique_songs = sorted(set(song_names))
    rng.shuffle(unique_songs)

    song_arrays = {s: histogram_to_array(histograms[s], label_list) for s in unique_songs}
    global_hist = np.sum(list(song_arrays.values()), axis=0)
    target = global_hist / n_splits

    max_per_fold = -(-len(unique_songs) // n_splits)  # ceil division
    frame_target = global_hist.sum() / n_splits
    global_pct = global_hist / (global_hist.sum() + 1e-10)

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

    return assignments


# ── reporting ───────────────────────────────────────────────────────────

def compute_fold_stats(song_names, folds, histograms, label_list, n_splits):
    """Aggregate per-fold statistics.

    Returns list of dicts (one per fold) with keys:
        n_songs, total_frames, distribution (np.array percentages)
    """
    stats = []
    for k in range(n_splits):
        fold_songs = [s for s in song_names if folds.get(s) == k]
        fold_unique = sorted(set(fold_songs))
        agg = np.zeros(len(label_list), dtype=np.float64)
        for s in fold_unique:
            agg += histogram_to_array(histograms[s], label_list)
        total = agg.sum()
        pct = 100.0 * agg / (total + 1e-10)
        stats.append({
            "n_songs": len(fold_songs),
            "n_unique": len(fold_unique),
            "total_frames": int(total),
            "distribution": pct,
            "raw": agg,
        })
    return stats


def print_report(title, stats, label_list, top_n):
    n_splits = len(stats)
    global_raw = np.sum([s["raw"] for s in stats], axis=0)
    global_total = global_raw.sum()
    global_pct = 100.0 * global_raw / (global_total + 1e-10)

    top_indices = np.argsort(global_raw)[-top_n:][::-1]
    top_labels = [label_list[i] for i in top_indices]

    print()
    print("=" * 90)
    print(f"  {title}")
    print("=" * 90)

    header = f"{'Fold':<6} {'Songs':>6} {'Frames':>10}  "
    header += "  ".join(f"{l:>8}" for l in top_labels)
    print(header)
    print("-" * len(header))

    for k, s in enumerate(stats):
        row = f"  {k:<4} {s['n_songs']:>6} {s['total_frames']:>10}  "
        row += "  ".join(f"{s['distribution'][i]:>7.1f}%" for i in top_indices)
        print(row)

    row_global = f"{'ALL':<6} {sum(s['n_songs'] for s in stats):>6} {int(global_total):>10}  "
    row_global += "  ".join(f"{global_pct[i]:>7.1f}%" for i in top_indices)
    print("-" * len(header))
    print(row_global)

    # Max deviation
    print()
    print("Max absolute deviation from global (pp):")
    deviations = []
    for i in top_indices:
        vals = [s["distribution"][i] for s in stats]
        dev = max(abs(v - global_pct[i]) for v in vals)
        deviations.append((label_list[i], dev))
    deviations.sort(key=lambda x: -x[1])
    for label, dev in deviations[:10]:
        print(f"  {label:>12s}: {dev:+.2f} pp")

    # Aggregate chi-squared
    chi2_list = []
    for s in stats:
        fold_pct = s["distribution"]
        chi2 = np.sum((fold_pct - global_pct) ** 2 / (global_pct + 1e-10))
        chi2_list.append(chi2)
    print(f"\nChi-squared medio entre folds e global: {np.mean(chi2_list):.4f}")

    # Frames balance
    frames = [s["total_frames"] for s in stats]
    print(f"Frames por fold: min={min(frames)}, max={max(frames)}, ratio max/min={max(frames)/(min(frames)+1):.2f}")
    print()


# ── main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic: compare k-fold chord distribution (alphabetical vs stratified)")
    parser.add_argument("--data_root", required=True,
                        help="Root datasets directory (e.g. ~/datasets)")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Dataset names (e.g. billboard jaah rwc)")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_n", type=int, default=15,
                        help="Number of top chord classes to show in the report")
    args = parser.parse_args()

    data_root = os.path.expanduser(args.data_root)

    print("Collecting songs...")
    song_names, temp = collect_songs(data_root, args.datasets)
    print(f"Total song entries: {len(song_names)}  (unique: {len(set(song_names))})")

    print("\nComputing chord histograms (loading non-augmented .pt files)...")
    histograms, label_list = compute_song_histograms(song_names, temp)
    print(f"Chord vocabulary size: {len(label_list)}")

    total_frames = sum(h.total() for h in histograms.values())
    print(f"Total frames across all songs: {total_frames}")

    # Method A: current alphabetical
    folds_alpha = split_alphabetical(song_names, args.n_splits)
    stats_alpha = compute_fold_stats(song_names, folds_alpha, histograms, label_list, args.n_splits)
    print_report("METODO ATUAL (ALFABETICO - blocos contiguos)", stats_alpha, label_list, args.top_n)

    # Method B: proposed stratified greedy
    folds_strat = split_stratified_greedy(song_names, histograms, label_list, args.n_splits, args.seed)
    stats_strat = compute_fold_stats(song_names, folds_strat, histograms, label_list, args.n_splits)
    print_report("METODO PROPOSTO (ESTRATIFICADO - greedy matching)", stats_strat, label_list, args.top_n)

    # Summary comparison
    chi2_alpha = np.mean([
        np.sum((s["distribution"] - 100.0 * np.sum([st["raw"] for st in stats_alpha], axis=0) /
                (sum(st["total_frames"] for st in stats_alpha) + 1e-10)) ** 2 /
               (100.0 * np.sum([st["raw"] for st in stats_alpha], axis=0) /
                (sum(st["total_frames"] for st in stats_alpha) + 1e-10) + 1e-10))
        for s in stats_alpha
    ])
    chi2_strat = np.mean([
        np.sum((s["distribution"] - 100.0 * np.sum([st["raw"] for st in stats_strat], axis=0) /
                (sum(st["total_frames"] for st in stats_strat) + 1e-10)) ** 2 /
               (100.0 * np.sum([st["raw"] for st in stats_strat], axis=0) /
                (sum(st["total_frames"] for st in stats_strat) + 1e-10) + 1e-10))
        for s in stats_strat
    ])

    print("=" * 90)
    print("  RESUMO COMPARATIVO")
    print("=" * 90)
    print(f"  Chi-squared medio  -  Alfabetico: {chi2_alpha:.4f}   Estratificado: {chi2_strat:.4f}")
    improvement = (1 - chi2_strat / (chi2_alpha + 1e-10)) * 100
    print(f"  Reducao de divergencia: {improvement:.1f}%")

    frames_alpha = [s["total_frames"] for s in stats_alpha]
    frames_strat = [s["total_frames"] for s in stats_strat]
    print(f"  Frames ratio (max/min)  -  Alfabetico: {max(frames_alpha)/(min(frames_alpha)+1):.2f}"
          f"   Estratificado: {max(frames_strat)/(min(frames_strat)+1):.2f}")
    print()


if __name__ == "__main__":
    main()
