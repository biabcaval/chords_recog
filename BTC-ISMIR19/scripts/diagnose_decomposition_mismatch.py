#!/usr/bin/env python
# encoding: utf-8
"""
Diagnose train/validation mismatch for decomposed chord training.

This script inspects:
1) Label source usage (original_chord_labels/original_chords/chord)
2) Per-component class distributions
3) Train-vs-validation divergence (Jensen-Shannon) per component

It uses the same dataset pipeline as training (AudioDatasetStructured), so the
analysis reflects what the model actually sees.
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from data.audio_dataset_structured import AudioDatasetStructured
from utils.chord_decomposition import COMPONENT_NAMES, CHORD_VOCAB
from utils.hparams import HParams


def _js_divergence_from_counts(train_counts: np.ndarray, val_counts: np.ndarray) -> float:
    """Jensen-Shannon divergence for two count vectors."""
    train_total = float(np.sum(train_counts))
    val_total = float(np.sum(val_counts))
    if train_total == 0 or val_total == 0:
        return float("nan")

    p = train_counts.astype(np.float64) / train_total
    q = val_counts.astype(np.float64) / val_total
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _top_classes(counts: np.ndarray, vocab: list[str], k: int = 5) -> list[dict]:
    """Return top-k classes with counts and ratios."""
    total = int(np.sum(counts))
    if total == 0:
        return []
    order = np.argsort(counts)[::-1][:k]
    out = []
    for idx in order:
        c = int(counts[idx])
        out.append(
            {
                "idx": int(idx),
                "label": vocab[int(idx)],
                "count": c,
                "ratio": c / total,
            }
        )
    return out


def analyze_split(dataset: AudioDatasetStructured, split_name: str, sample_limit: int | None) -> dict:
    """Analyze one split (train or validation)."""
    n_available = len(dataset)
    n_to_process = n_available if sample_limit is None else min(n_available, sample_limit)

    source_counter: Counter = Counter()
    component_counts = {
        comp: np.zeros(len(CHORD_VOCAB[comp]), dtype=np.int64) for comp in COMPONENT_NAMES
    }

    for idx in range(n_to_process):
        # Inspect raw source to identify which label field is being used.
        path = dataset.paths[idx]
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if "original_chord_labels" in raw:
            source_counter["original_chord_labels"] += 1
        elif "original_chords" in raw:
            source_counter["original_chords"] += 1
        elif "chord" in raw:
            source_counter["chord"] += 1
        else:
            source_counter["missing"] += 1

        # Use dataset pipeline output (same decomposition used for training).
        sample = dataset[idx]
        comps = sample.get("components", {})
        for comp in COMPONENT_NAMES:
            arr = comps.get(comp, None)
            if arr is None:
                continue
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            arr = np.asarray(arr).reshape(-1)
            binc = np.bincount(arr, minlength=len(CHORD_VOCAB[comp]))
            component_counts[comp] += binc

    summary = {
        "split": split_name,
        "samples_available": n_available,
        "samples_processed": n_to_process,
        "label_source_counts": dict(source_counter),
        "components": {},
    }

    for comp in COMPONENT_NAMES:
        counts = component_counts[comp]
        total = int(np.sum(counts))
        n_count = int(counts[0]) if len(counts) > 0 else 0
        summary["components"][comp] = {
            "total_frames": total,
            "n_class_count": n_count,
            "n_class_ratio": (n_count / total) if total > 0 else None,
            "top_classes": _top_classes(counts, CHORD_VOCAB[comp], k=5),
            "counts": counts.tolist(),
        }

    return summary


def print_fold_report(fold_result: dict) -> None:
    fold = fold_result["kfold"]
    train = fold_result["train"]
    val = fold_result["val"]
    print("\n" + "=" * 72)
    print(f"K-FOLD {fold}")
    print("=" * 72)
    print(
        f"Samples processed -> train: {train['samples_processed']}/{train['samples_available']} | "
        f"val: {val['samples_processed']}/{val['samples_available']}"
    )
    print(f"Label sources (train): {train['label_source_counts']}")
    print(f"Label sources (val):   {val['label_source_counts']}")
    print("\nComponent divergence (JS, higher means larger mismatch):")
    for comp in COMPONENT_NAMES:
        js = fold_result["divergence"][comp]["js_divergence"]
        n_tr = fold_result["divergence"][comp]["train_n_ratio"]
        n_va = fold_result["divergence"][comp]["val_n_ratio"]
        js_txt = "nan" if np.isnan(js) else f"{js:.4f}"
        print(f"  - {comp:>4}: JS={js_txt} | N-ratio train={n_tr:.3f} val={n_va:.3f}")


def run_fold(config, data_root: str, dataset_names: tuple[str, ...], kfold: int, sample_limit: int | None) -> dict:
    train_dataset = AudioDatasetStructured(
        config,
        root_dir=data_root,
        dataset_names=dataset_names,
        train=True,
        decompose=True,
        kfold=kfold,
    )
    val_dataset = AudioDatasetStructured(
        config,
        root_dir=data_root,
        dataset_names=dataset_names,
        train=False,
        decompose=True,
        kfold=kfold,
    )

    train_summary = analyze_split(train_dataset, "train", sample_limit)
    val_summary = analyze_split(val_dataset, "val", sample_limit)

    divergence = {}
    for comp in COMPONENT_NAMES:
        tr_counts = np.array(train_summary["components"][comp]["counts"], dtype=np.int64)
        va_counts = np.array(val_summary["components"][comp]["counts"], dtype=np.int64)
        js = _js_divergence_from_counts(tr_counts, va_counts)
        divergence[comp] = {
            "js_divergence": js,
            "train_n_ratio": train_summary["components"][comp]["n_class_ratio"],
            "val_n_ratio": val_summary["components"][comp]["n_class_ratio"],
        }

    return {
        "kfold": kfold,
        "train": train_summary,
        "val": val_summary,
        "divergence": divergence,
    }


def parse_folds(text: str) -> list[int]:
    items = [x.strip() for x in text.split(",") if x.strip()]
    folds = sorted(set(int(x) for x in items))
    for f in folds:
        if f < 0 or f > 4:
            raise ValueError("kfold must be in [0,1,2,3,4]")
    return folds


def main():
    parser = argparse.ArgumentParser(description="Diagnose decomposition train/val mismatch")
    parser.add_argument("--config", type=str, default="run_config.yaml", help="Path to config file")
    parser.add_argument(
        "--kfolds",
        type=str,
        default="4",
        help="Comma-separated folds to inspect (e.g. '0,1,2,3,4')",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=2000,
        help="Max samples per split for faster diagnosis. Use -1 for full dataset.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save full diagnostic report as JSON",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Optional comma-separated dataset override (defaults to config.experiment.dataset_names)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Optional data root override (defaults to config.experiment.data_root or config.path.root_path)",
    )
    args = parser.parse_args()

    config = HParams.load(args.config)
    folds = parse_folds(args.kfolds)
    sample_limit = None if args.sample_limit is not None and args.sample_limit < 0 else args.sample_limit

    data_root = (
        args.data_root
        if args.data_root is not None
        else config.experiment.get("data_root", config.path.get("root_path", "/data/music/chord_recognition"))
    )
    dataset_names = (
        tuple([x.strip() for x in args.datasets.split(",") if x.strip()])
        if args.datasets
        else tuple(config.experiment.get("dataset_names", ["billboard"]))
    )

    print("Running decomposition mismatch diagnosis")
    print(f"  config: {args.config}")
    print(f"  data_root: {data_root}")
    print(f"  datasets: {dataset_names}")
    print(f"  kfolds: {folds}")
    print(f"  sample_limit: {sample_limit if sample_limit is not None else 'FULL'}")

    report = {
        "created_at": datetime.now().isoformat(),
        "config": args.config,
        "data_root": data_root,
        "datasets": list(dataset_names),
        "kfolds": folds,
        "sample_limit": sample_limit,
        "fold_reports": [],
    }

    for kf in folds:
        fold_result = run_fold(config, data_root, dataset_names, kf, sample_limit)
        report["fold_reports"].append(fold_result)
        print_fold_report(fold_result)

    # Aggregate mean JS across folds
    if report["fold_reports"]:
        agg = {}
        for comp in COMPONENT_NAMES:
            vals = [fr["divergence"][comp]["js_divergence"] for fr in report["fold_reports"]]
            vals = [v for v in vals if not np.isnan(v)]
            agg[comp] = float(np.mean(vals)) if vals else float("nan")
        report["mean_js_by_component"] = agg
        print("\nMean JS divergence across folds:")
        for comp in COMPONENT_NAMES:
            v = report["mean_js_by_component"][comp]
            print(f"  - {comp:>4}: {'nan' if np.isnan(v) else f'{v:.4f}'}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nSaved report: {out}")


if __name__ == "__main__":
    main()

