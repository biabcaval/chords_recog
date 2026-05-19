#!/usr/bin/env python
"""
Compute global mean and std of **dB-ref-max log-CQT** features across ALL
data of the specified datasets (no k-fold split — uses 100% of the data).

The transform applied here is *exactly* the same one used by the loaders
and inference paths via :func:`utils.preprocess.cqt_to_log_db` — namely
``20 * log10(|cqt| / max|cqt|)`` clipped to ``-80`` dB.  This guarantees
that the statistics saved by this script are compatible with the runtime
loader; mixing the previous ``ln(|x| + 1e-6)`` stats with the new dB
loader (or vice-versa) would shift features by tens of dB and destroy
training.

Mean/std are accumulated with a **vectorized parallel Welford** update
(Chan/Golub/LeVeque 1979): each file's ~252k values are reduced to three
scalars in numpy (microseconds per file), and only those scalars are
folded into the running aggregate.  This is mathematically equivalent
to the original element-wise Welford loop but ~1000x faster — without
this optimisation, the script took ~12-25h to process 8 datasets; with
it, ~30-60 min, dominated by ``torch.load`` IO.

Live progress (mean/std/n, ETA) is shown via a tqdm bar.

Saves the result as a .pt file that can be loaded during training and
inference.

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
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.hparams import HParams
from utils.preprocess import cqt_to_log_db


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
    """Vectorized parallel Welford for mean/std over all .pt features.

    For each file, we reduce the (252, ~1001) feature array to three
    summary scalars (chunk_n, chunk_mean, chunk_m2) using vectorized
    numpy ops, then fold them into the running aggregate via Chan/Welford's
    parallel update formula (Chan, Golub & LeVeque, 1979):

        delta    = chunk_mean - running_mean
        new_n    = n + chunk_n
        new_mean = running_mean + delta * chunk_n / new_n
        new_M2   = running_M2 + chunk_M2 + delta^2 * n * chunk_n / new_n

    This is mathematically equivalent to the previous element-wise
    Welford loop but ~1000x faster, because the per-file reduction stays
    inside numpy and avoids the Python-level boxing of every value.

    Progress (mean/std/n) is exposed live through a tqdm progress bar.
    """
    n = 0
    mean = 0.0
    m2 = 0.0

    pbar = tqdm(
        pt_files,
        desc="compute_normalization",
        unit="file",
        smoothing=0.05,
        dynamic_ncols=True,
    )
    for i, path in enumerate(pbar):
        data = torch.load(path, weights_only=False)
        raw = data['feature']
        if isinstance(raw, torch.Tensor):
            raw = raw.numpy()
        feat = cqt_to_log_db(raw).astype(np.float64).ravel()

        chunk_n = int(feat.size)
        chunk_mean = float(feat.mean())
        # Numerically stable: subtract chunk mean before squaring.
        chunk_m2 = float(((feat - chunk_mean) ** 2).sum())

        if n == 0:
            n, mean, m2 = chunk_n, chunk_mean, chunk_m2
        else:
            delta = chunk_mean - mean
            new_n = n + chunk_n
            new_mean = mean + delta * chunk_n / new_n
            new_m2 = m2 + chunk_m2 + delta * delta * n * chunk_n / new_n
            n, mean, m2 = new_n, new_mean, new_m2

        # Refresh live stats every 100 files (cheap; avoids tqdm spam).
        if (i + 1) % 100 == 0 or i == len(pt_files) - 1:
            std = (m2 / n) ** 0.5 if n > 1 else 1.0
            pbar.set_postfix(
                mean=f"{mean:+.4f}",
                std=f"{std:.4f}",
                n=f"{n:,}",
            )

    std = (m2 / n) ** 0.5 if n > 1 else 1.0
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

    # Force line-buffered stdout so the header/footer prints reach the
    # terminal (and any `| tee log`) immediately, even when stdout is a
    # pipe.  Without this, Python defaults to 4-8 KB block buffering on
    # non-TTY stdout, which would delay these short messages for minutes.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # py>=3.7
    except AttributeError:
        pass

    config = HParams.load(args.config)
    data_root = config.experiment.get(
        'data_root', config.path.get('root_path', '')
    )
    dataset_names = args.datasets or config.experiment.get('dataset_names', ['billboard'])

    print(f"Config     : {args.config}", flush=True)
    print(f"Data root  : {data_root}", flush=True)
    print(f"Datasets   : {dataset_names}", flush=True)
    print(flush=True)

    pt_files = find_all_pt_files(data_root, dataset_names, config)
    print(f"\nTotal files: {len(pt_files)}\n", flush=True)

    if not pt_files:
        print("ERROR: No .pt files found.", flush=True)
        return

    mean, std = compute_normalization(pt_files)

    payload = {
        'mean': mean,
        'std': std,
        'datasets': dataset_names,
        'n_files': len(pt_files),
    }
    torch.save(payload, args.output)

    print(f"\nSaved to: {args.output}", flush=True)
    print(f"  mean = {mean:.6f}", flush=True)
    print(f"  std  = {std:.6f}", flush=True)


if __name__ == "__main__":
    main()
