#!/usr/bin/env python3
"""
Extract chord component frequencies from .lab annotation files.

This script scans dataset annotation folders, parses chord labels, maps them
to plot-style component classes, and writes aggregate counts to a CSV.

Main output format:
    component,count
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


# Ensure project root is importable when run from scripts/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.chord_decomposition import ChordDecomposer as ProjectChordDecomposer
except Exception:
    ProjectChordDecomposer = None


# Plot-compatible baseline order (extras discovered in data are appended after).
PLOT_COMPONENT_ORDER = [
    "N",
    "maj",
    "min",
    "sus4",
    "sus2",
    "dim",
    "aug",
    "/2",
    "/b3",
    "/3",
    "/4",
    "/5",
    "/6",
    "/b7",
    "/7",
    "+7",
    "+b7",
    "+bb7",
    "+9",
    "+#9",
    "+b9",
    "+11",
    "+#11",
    "+13",
    "+b13",
]

# Semitone interval from root to bass -> scale degree label.
# We use flat-based naming for consistency with the user-provided plot.
BASS_INTERVAL_TO_DEGREE = {
    1: "b2",
    2: "2",
    3: "b3",
    4: "3",
    5: "4",
    6: "b5",
    7: "5",
    8: "b6",
    9: "6",
    10: "b7",
    11: "7",
}

PITCH_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}


@dataclass
class ProcessingStats:
    files_processed: int = 0
    lines_total: int = 0
    lines_valid: int = 0
    lines_skipped: int = 0
    unique_chords_seen: int = 0

    def merge(self, other: "ProcessingStats") -> None:
        self.files_processed += other.files_processed
        self.lines_total += other.lines_total
        self.lines_valid += other.lines_valid
        self.lines_skipped += other.lines_skipped
        self.unique_chords_seen += other.unique_chords_seen


class LightweightChordDecomposer:
    """
    Dependency-light decomposer fallback.

    It captures the component logic needed by this script when importing the
    full project decomposer is not possible in the current Python environment.
    """

    FLAT_TO_SHARP = {
        "Cb": "B",
        "Db": "C#",
        "Eb": "D#",
        "Fb": "E",
        "Gb": "F#",
        "Ab": "G#",
        "Bb": "A#",
    }

    PITCH_CLASSES = set(PITCH_TO_SEMITONE.keys())

    def _normalize_pitch(self, pitch: str) -> str:
        return self.FLAT_TO_SHARP.get(pitch, pitch)

    def _parse_chord(self, label: str) -> Tuple[str | None, str | None, str | None]:
        root: str | None = None
        quality: str | None = None
        bass: str | None = None

        colon_idx = label.find(":")
        slash_idx = label.find("/")

        if colon_idx == -1 and slash_idx == -1:
            root = label
        elif colon_idx == -1:
            root = label[:slash_idx]
            bass = label[slash_idx + 1 :]
        elif slash_idx == -1:
            root = label[:colon_idx]
            quality = label[colon_idx + 1 :]
        else:
            root = label[:colon_idx]
            quality = label[colon_idx + 1 : slash_idx]
            bass = label[slash_idx + 1 :]

        if root:
            root = self._normalize_pitch(root)
            if root not in self.PITCH_CLASSES:
                root = None

        if bass:
            bass = self._normalize_pitch(bass)
            if bass not in self.PITCH_CLASSES:
                bass = None

        return root, quality, bass

    def _extract_extensions(self, remaining: str, components: Dict[str, str]) -> None:
        rem = remaining.replace("(", "").replace(")", "")

        if "b13" in rem:
            components["13th"] = "b13"
            rem = rem.replace("b13", "", 1)
        elif "#13" in rem:
            components["13th"] = "13"
            rem = rem.replace("#13", "", 1)
        elif "13" in rem:
            components["13th"] = "13"
            rem = rem.replace("13", "", 1)

        if "#11" in rem:
            components["11th"] = "#11"
            rem = rem.replace("#11", "", 1)
        elif "b11" in rem:
            components["11th"] = "11"
            rem = rem.replace("b11", "", 1)
        elif "11" in rem:
            components["11th"] = "11"
            rem = rem.replace("11", "", 1)

        if "#9" in rem:
            components["9th"] = "#9"
            rem = rem.replace("#9", "", 1)
        elif "b9" in rem:
            components["9th"] = "b9"
            rem = rem.replace("b9", "", 1)
        elif "9" in rem:
            components["9th"] = "9"
            rem = rem.replace("9", "", 1)

        if components["7th"] == "N":
            if "bb7" in rem:
                components["7th"] = "bb7"
                rem = rem.replace("bb7", "", 1)
            elif "maj7" in rem:
                components["7th"] = "7"
                rem = rem.replace("maj7", "", 1)
            elif "b7" in rem:
                components["7th"] = "b7"
                rem = rem.replace("b7", "", 1)
            elif "7" in rem:
                components["7th"] = "b7"
                rem = rem.replace("7", "", 1)

        if "6" in rem:
            components["6th"] = "6"

    def _decompose_quality(self, quality: str, components: Dict[str, str]) -> None:
        q = quality.replace("(", "").replace(")", "").lower()

        if q in {"5", "pedal"} or q.startswith("(1,5)") or q == "power":
            components["misc"] = "5"
            return

        if q.startswith("hdim") or q == "hdim7":
            components["triad"] = "dim"
            components["7th"] = "b7"
            rem = q.replace("hdim", "", 1).replace("7", "", 1)
            self._extract_extensions(rem, components)
            return

        if "minmaj7" in q or "minmaj" in q:
            components["triad"] = "min"
            components["7th"] = "7"
            rem = q.replace("minmaj7", "").replace("minmaj", "")
            self._extract_extensions(rem, components)
            return

        if q == "dim7" or q.startswith("dim7"):
            components["triad"] = "dim"
            components["7th"] = "bb7"
            rem = q.replace("dim7", "")
            self._extract_extensions(rem, components)
            return

        if "maj7" in q:
            components["triad"] = "maj"
            components["7th"] = "7"
            rem = q.replace("maj7", "")
            self._extract_extensions(rem, components)
            return

        triad = None
        rem = q
        for triad_name in ("sus2", "sus4", "maj", "min", "dim", "aug"):
            if triad_name in q:
                triad = triad_name
                idx = q.find(triad_name)
                rem = q[:idx] + q[idx + len(triad_name) :]
                break

        if triad is not None:
            components["triad"] = triad
        elif any(token in q for token in ("6", "7", "9", "11", "13")):
            components["triad"] = "maj"
            rem = q

        self._extract_extensions(rem, components)

    def decompose(self, chord_label: str) -> Dict[str, str]:
        components = {
            "root": "N",
            "bass": "N",
            "triad": "N",
            "misc": "N",
            "6th": "N",
            "7th": "N",
            "9th": "N",
            "11th": "N",
            "13th": "N",
        }

        if chord_label in {"N", "X"}:
            return components

        root, quality, bass = self._parse_chord(chord_label)
        if root is not None:
            components["root"] = root
        if bass is not None and bass != root:
            components["bass"] = bass

        if quality:
            self._decompose_quality(quality, components)
        elif root is not None:
            components["triad"] = "maj"

        return components


def make_decomposer():
    if ProjectChordDecomposer is not None:
        return ProjectChordDecomposer()
    return LightweightChordDecomposer()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract chord component frequency counts from datasets/*/annotations/*.lab"
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        default="/home/daniel.melo/datasets",
        help="Root directory containing datasets (default: /home/daniel.melo/datasets)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Optional dataset names (e.g., billboard jaah rwc). If omitted, all subdirs with annotations are used.",
    )
    parser.add_argument(
        "--annotations_dirname",
        type=str,
        default="annotations",
        help="Annotation directory name inside each dataset (default: annotations)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="chord_component_frequencies.csv",
        help="Output CSV path (default: chord_component_frequencies.csv)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Parallel workers for file processing (default: 1 = sequential)",
    )
    parser.add_argument(
        "--include_zero_known",
        action="store_true",
        help="Include known plot classes even when count is zero.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=200,
        help="Print progress every N files in sequential mode (default: 200)",
    )
    return parser.parse_args()


def find_lab_files(
    datasets_root: Path,
    annotations_dirname: str,
    dataset_names: Sequence[str] | None,
) -> List[Path]:
    files: List[Path] = []

    if dataset_names:
        candidates = [datasets_root / name for name in dataset_names]
    else:
        candidates = [p for p in sorted(datasets_root.iterdir()) if p.is_dir()]

    for dataset_dir in candidates:
        annotations_dir = dataset_dir / annotations_dirname
        if not annotations_dir.exists() or not annotations_dir.is_dir():
            continue
        files.extend(sorted(annotations_dir.rglob("*.lab")))

    return files


def chord_to_plot_classes(
    chord_label: str,
    decomposer: ChordDecomposer,
    cache: Dict[str, Tuple[str, ...]],
) -> Tuple[str, ...]:
    cached = cache.get(chord_label)
    if cached is not None:
        return cached

    label = chord_label.strip()
    if not label:
        classes = ("N",)
        cache[chord_label] = classes
        return classes

    if label in {"N", "X"}:
        classes = ("N",)
        cache[chord_label] = classes
        return classes

    components = decomposer.decompose(label)
    classes_list: List[str] = []

    # Triad / power-chord class.
    triad = components.get("triad", "N")
    misc = components.get("misc", "N")
    if triad != "N":
        classes_list.append(triad)
    if misc != "N":
        classes_list.append(misc)

    # Slash-bass class as interval degree (/2, /b3, /5, ...).
    root = components.get("root", "N")
    bass = components.get("bass", "N")
    if root != "N" and bass != "N" and root in PITCH_TO_SEMITONE and bass in PITCH_TO_SEMITONE:
        interval = (PITCH_TO_SEMITONE[bass] - PITCH_TO_SEMITONE[root]) % 12
        degree = BASS_INTERVAL_TO_DEGREE.get(interval)
        if degree is not None:
            classes_list.append(f"/{degree}")
        elif interval != 0:
            # Keep unseen interval naming explicit instead of dropping it.
            classes_list.append(f"/interval_{interval}")

    # Extensions in plot-style notation.
    for extension_component in ("6th", "7th", "9th", "11th", "13th"):
        value = components.get(extension_component, "N")
        if value != "N":
            classes_list.append(f"+{value}")

    # Fallback: keep unknown chord visible in outputs for auditability.
    if not classes_list:
        classes_list.append(f"RAW:{label}")

    # Deduplicate while preserving order.
    deduped = tuple(dict.fromkeys(classes_list))
    cache[chord_label] = deduped
    return deduped


def parse_lab_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    parts = stripped.split()
    if len(parts) < 3:
        return None

    # Typical format: <start> <end> <label...>
    return " ".join(parts[2:])


def process_lab_file(
    lab_path: Path,
    decomposer: ChordDecomposer,
    cache: Dict[str, Tuple[str, ...]],
) -> Tuple[Counter, ProcessingStats]:
    counts: Counter = Counter()
    stats = ProcessingStats(files_processed=1)

    with lab_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stats.lines_total += 1
            chord_label = parse_lab_line(line)
            if chord_label is None:
                stats.lines_skipped += 1
                continue

            for comp_class in chord_to_plot_classes(chord_label, decomposer, cache):
                counts[comp_class] += 1

            stats.lines_valid += 1

    return counts, stats


def process_lab_file_worker(lab_path_str: str) -> Tuple[Dict[str, int], ProcessingStats]:
    lab_path = Path(lab_path_str)
    decomposer = make_decomposer()
    local_cache: Dict[str, Tuple[str, ...]] = {}
    counts, stats = process_lab_file(lab_path, decomposer, local_cache)
    stats.unique_chords_seen = len(local_cache)
    return dict(counts), stats


def write_counts_csv(
    output_csv: Path,
    total_counts: Counter,
    include_zero_known: bool,
) -> None:
    known_set = set(PLOT_COMPONENT_ORDER)
    ordered_classes: List[str] = []

    for comp_class in PLOT_COMPONENT_ORDER:
        if include_zero_known or comp_class in total_counts:
            ordered_classes.append(comp_class)

    extras = sorted(cls for cls in total_counts if cls not in known_set)
    ordered_classes.extend(extras)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", "count"])
        for comp_class in ordered_classes:
            writer.writerow([comp_class, int(total_counts.get(comp_class, 0))])


def run_sequential(
    lab_files: Sequence[Path],
    progress_every: int,
) -> Tuple[Counter, ProcessingStats]:
    decomposer = make_decomposer()
    cache: Dict[str, Tuple[str, ...]] = {}
    total_counts: Counter = Counter()
    total_stats = ProcessingStats()

    for idx, lab_path in enumerate(lab_files, start=1):
        file_counts, file_stats = process_lab_file(lab_path, decomposer, cache)
        total_counts.update(file_counts)
        total_stats.merge(file_stats)

        if progress_every > 0 and idx % progress_every == 0:
            print(f"Processed {idx}/{len(lab_files)} files...")

    total_stats.unique_chords_seen = len(cache)
    return total_counts, total_stats


def run_parallel(lab_files: Sequence[Path], num_workers: int) -> Tuple[Counter, ProcessingStats]:
    total_counts: Counter = Counter()
    total_stats = ProcessingStats()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for counts_dict, stats in executor.map(process_lab_file_worker, (str(p) for p in lab_files)):
            total_counts.update(counts_dict)
            total_stats.merge(stats)

    # Parallel workers have local caches, so this is an upper bound sum.
    return total_counts, total_stats


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    datasets_root = Path(args.datasets_root).expanduser()
    if not datasets_root.exists() or not datasets_root.is_dir():
        raise FileNotFoundError(f"datasets_root does not exist or is not a directory: {datasets_root}")

    lab_files = find_lab_files(
        datasets_root=datasets_root,
        annotations_dirname=args.annotations_dirname,
        dataset_names=args.datasets,
    )
    if not lab_files:
        raise RuntimeError(
            f"No .lab files found under {datasets_root} with annotations dir '{args.annotations_dirname}'."
        )

    print(f"Datasets root: {datasets_root}")
    print(f"Annotation files found: {len(lab_files)}")
    if args.datasets:
        print(f"Datasets selected: {args.datasets}")

    if args.num_workers > 1:
        print(f"Processing mode: parallel ({args.num_workers} workers)")
        total_counts, stats = run_parallel(lab_files, args.num_workers)
    else:
        print("Processing mode: sequential (shared cache enabled)")
        total_counts, stats = run_sequential(lab_files, args.progress_every)

    output_csv = Path(args.output_csv).expanduser()
    write_counts_csv(output_csv, total_counts, args.include_zero_known)

    elapsed_s = time.perf_counter() - start_time
    print("\nDone.")
    print(f"Output CSV: {output_csv.resolve()}")
    print(f"Unique component classes: {len(total_counts)}")
    print(f"Files processed: {stats.files_processed}")
    print(f"Lines total: {stats.lines_total}")
    print(f"Lines valid: {stats.lines_valid}")
    print(f"Lines skipped: {stats.lines_skipped}")
    print(f"Unique chord labels seen (cache): {stats.unique_chords_seen}")
    print(f"Elapsed time: {elapsed_s:.2f}s")


if __name__ == "__main__":
    main()
