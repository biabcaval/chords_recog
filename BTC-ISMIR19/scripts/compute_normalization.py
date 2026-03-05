#!/usr/bin/env python
"""
Compute global mean and std of log-CQT features across ALL data of the
specified datasets (no k-fold split — uses 100% of the data).

Saves the result as a .pt file that can be loaded during training and inference.

Usage:
    python scripts/compute_normalization.py \
        --config run_config.yaml \
        --output normalization.pt

    # Restrict to specific datasets:
    python scripts/compute_normalization.py \
        --config run_config.yaml \
        --datasets billboard jaah rwc \
        --output normalization_BiJaRw.pt
"""

import os
import sys
import argparse
import glob
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.hparams import HParams


def find_all_pt_files(data_root, dataset_names, config):
    """Find every preprocessed .pt sample across the given datasets."""
    mp3_cfg = config.mp3
    feat_cfg = config.feature
    mp3_str = "%d_%.1f_%.1f" % (mp3_cfg['song_hz'], mp3_cfg['inst_len'], mp3_cfg['skip_interval'])
    feat_str = "%s_%d_%d_%d" % ('cqt', feat_cfg['n_bins'], feat_cfg['bins_per_octave'], feat_cfg['hop_length'])

    all_files = []
    for name in dataset_names:
        candidates = [
            os.path.join(data_root, "result_decomposed", name + "_voca", mp3_str, feat_str),
            os.path.join(data_root, "result", name + "_voca", mp3_str, feat_str),
        ]
        dataset_path = None
        for c in candidates:
            if os.path.isdir(c):
                dataset_path = c
                break

        if dataset_path is None:
            print(f"  WARNING: no data found for '{name}', tried:")
            for c in candidates:
                print(f"    {c}")
            continue

        pt_files = sorted(glob.glob(os.path.join(dataset_path, "**", "*.pt"), recursive=True))
        print(f"  {name}: {len(pt_files)} files  ({dataset_path})")
        all_files.extend(pt_files)

    return all_files


def compute_normalization(pt_files):
    """Welford's online algorithm for numerically stable mean/std."""
    n = 0
    mean = 0.0
    m2 = 0.0

    total = len(pt_files)
    for i, path in enumerate(pt_files):
        data = torch.load(path, weights_only=False)
        raw = data['feature']
        if isinstance(raw, torch.Tensor):
            raw = raw.numpy()
        feat = np.log(np.abs(raw) + 1e-6).astype(np.float64).ravel()

        for val in feat:
            n += 1
            delta = val - mean
            mean += delta / n
            delta2 = val - mean
            m2 += delta * delta2

        if (i + 1) % 500 == 0 or i == total - 1:
            std = np.sqrt(m2 / n) if n > 1 else 1.0
            print(f"  [{i + 1}/{total}]  mean={mean:.6f}  std={std:.6f}  (n={n:,})")

    std = np.sqrt(m2 / n) if n > 1 else 1.0
    return float(mean), float(std)


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean/std normalization for log-CQT features (all data, no k-fold split)"
    )
    parser.add_argument("--config", type=str, default="run_config.yaml")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Override dataset names (default: from config)")
    parser.add_argument("--output", type=str, default="normalization.pt")
    args = parser.parse_args()

    config = HParams.load(args.config)
    data_root = config.experiment.get(
        'data_root', config.path.get('root_path', '')
    )
    dataset_names = args.datasets or config.experiment.get('dataset_names', ['billboard'])

    print(f"Config     : {args.config}")
    print(f"Data root  : {data_root}")
    print(f"Datasets   : {dataset_names}")
    print()

    pt_files = find_all_pt_files(data_root, dataset_names, config)
    print(f"\nTotal files: {len(pt_files)}\n")

    if not pt_files:
        print("ERROR: No .pt files found.")
        return

    mean, std = compute_normalization(pt_files)

    payload = {
        'mean': mean,
        'std': std,
        'datasets': dataset_names,
        'n_files': len(pt_files),
    }
    torch.save(payload, args.output)

    print(f"\nSaved to: {args.output}")
    print(f"  mean = {mean:.6f}")
    print(f"  std  = {std:.6f}")


if __name__ == "__main__":
    main()
