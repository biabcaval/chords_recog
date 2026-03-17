#!/usr/bin/env python3
"""
Diagnose .pt file format across datasets.

Checks whether preprocessed .pt files contain 'original_chord_labels'
(full extensions) or only legacy 170-class indices. Reports statistics
to help understand if the model was trained with or without extension info.

Usage:
    python scripts/diagnose_pt_format.py --data_root /path/to/datasets
    python scripts/diagnose_pt_format.py --data_root /path/to/datasets --datasets billboard jaah
    python scripts/diagnose_pt_format.py --data_root /path/to/datasets --sample 5
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch


def inspect_pt_file(pt_path: Path) -> dict:
    """Load a .pt file and report which keys are present."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    info = {
        "path": str(pt_path),
        "keys": sorted(data.keys()) if isinstance(data, dict) else ["<not-a-dict>"],
        "has_original_chord_labels": "original_chord_labels" in data if isinstance(data, dict) else False,
        "has_original_chords": "original_chords" in data if isinstance(data, dict) else False,
        "has_chord": "chord" in data if isinstance(data, dict) else False,
    }

    if isinstance(data, dict):
        if "original_chord_labels" in data:
            labels = data["original_chord_labels"]
            info["label_type"] = "original_chord_labels"
            if hasattr(labels, '__len__') and len(labels) > 0:
                info["sample_labels"] = list(labels[:5])
                info["n_labels"] = len(labels)
                has_ext = any(
                    any(ext in str(lbl) for ext in ['(9)', '(b9)', '(#9)', '(11)', '(#11)', '(13)', '(b13)'])
                    for lbl in labels
                )
                info["has_parenthetical_extensions"] = has_ext
        elif "original_chords" in data:
            info["label_type"] = "original_chords"
            chords = data["original_chords"]
            if hasattr(chords, '__len__') and len(chords) > 0:
                info["sample_labels"] = list(chords[:5])
                info["n_labels"] = len(chords)
        elif "chord" in data:
            info["label_type"] = "chord_indices"
            chord = data["chord"]
            if hasattr(chord, '__len__') and len(chord) > 0:
                info["sample_labels"] = list(chord[:5])
                info["n_labels"] = len(chord)
        else:
            info["label_type"] = "none"

    return info


def find_pt_files(data_root: Path, dataset_names=None) -> list:
    """Find all .pt files under dataset directories."""
    files = []
    if dataset_names:
        candidates = [data_root / name for name in dataset_names]
    else:
        candidates = sorted(p for p in data_root.iterdir() if p.is_dir())

    for dataset_dir in candidates:
        if not dataset_dir.exists():
            print(f"  WARNING: {dataset_dir} does not exist, skipping")
            continue
        files.extend(sorted(dataset_dir.rglob("*.pt")))

    return files


def main():
    parser = argparse.ArgumentParser(description="Diagnose .pt file format for chord labels")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing datasets")
    parser.add_argument("--datasets", type=str, nargs="*", default=None, help="Specific dataset names")
    parser.add_argument("--sample", type=int, default=3, help="Number of sample files to show detailed info (default: 3)")
    parser.add_argument("--max_files", type=int, default=None, help="Max files to scan (default: all)")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    if not data_root.exists():
        print(f"ERROR: data_root does not exist: {data_root}")
        sys.exit(1)

    pt_files = find_pt_files(data_root, args.datasets)
    if not pt_files:
        print(f"No .pt files found under {data_root}")
        sys.exit(1)

    if args.max_files:
        pt_files = pt_files[:args.max_files]

    print(f"Data root: {data_root}")
    print(f"Total .pt files found: {len(pt_files)}")
    print()

    label_type_counts = Counter()
    has_extensions_count = 0
    key_sets = Counter()
    sample_shown = 0

    for pt_path in pt_files:
        try:
            info = inspect_pt_file(pt_path)
        except Exception as e:
            print(f"  ERROR loading {pt_path}: {e}")
            continue

        label_type_counts[info.get("label_type", "unknown")] += 1
        key_sets[tuple(info["keys"])] += 1

        if info.get("has_parenthetical_extensions"):
            has_extensions_count += 1

        if sample_shown < args.sample:
            print(f"--- Sample {sample_shown + 1}: {pt_path.name} ---")
            print(f"  Keys: {info['keys']}")
            print(f"  Label type: {info.get('label_type', 'N/A')}")
            if "sample_labels" in info:
                print(f"  Sample labels: {info['sample_labels']}")
            if "n_labels" in info:
                print(f"  Total labels: {info['n_labels']}")
            if "has_parenthetical_extensions" in info:
                print(f"  Has parenthetical extensions: {info['has_parenthetical_extensions']}")
            print()
            sample_shown += 1

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files scanned: {len(pt_files)}")
    print()
    print("Label type distribution:")
    for label_type, count in label_type_counts.most_common():
        pct = 100 * count / len(pt_files)
        print(f"  {label_type:30s}: {count:5d} ({pct:5.1f}%)")
    print()
    print(f"Files with parenthetical extensions (9, b9, #11, etc.): {has_extensions_count}")
    print()
    print("Key-set patterns:")
    for keys, count in key_sets.most_common(5):
        print(f"  {count:5d} files: {list(keys)}")

    print()
    if label_type_counts.get("chord_indices", 0) > 0 or label_type_counts.get("original_chords", 0) > 0:
        missing = label_type_counts.get("chord_indices", 0) + label_type_counts.get("original_chords", 0)
        print(f"WARNING: {missing} files use legacy 170-class labels without extensions.")
        print("         Extensions (9th, 11th, 13th) will be lost for these files.")
        print("         Run add_original_labels.py to fix.")
    else:
        print("OK: All files have original_chord_labels with full extension support.")


if __name__ == "__main__":
    main()
