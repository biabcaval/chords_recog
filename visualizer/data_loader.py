"""
Data loading and chord decomposition utilities for the visualizer.

Parses .lab files, decomposes chords into 9 components, and computes
diffs between ground truth and prediction annotations.
"""

import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'BTC-ISMIR19'))
from utils.chord_decomposition import ChordDecomposer, COMPONENT_NAMES, CHORD_VOCAB


_decomposer = ChordDecomposer()


def parse_lab_file(filepath: str) -> List[Dict]:
    """Parse a .lab file into a list of segment dicts."""
    segments = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                segments.append({
                    'start': float(parts[0]),
                    'end': float(parts[1]),
                    'chord': ' '.join(parts[2:]),
                })
    return segments


def decompose_segments(segments: List[Dict]) -> Dict[str, List[Dict]]:
    """Decompose chord segments into per-component timelines.

    Returns a dict mapping component name -> list of
    ``{"start", "end", "value"}`` dicts.
    """
    result: Dict[str, List[Dict]] = {comp: [] for comp in COMPONENT_NAMES}
    for seg in segments:
        components = _decomposer.decompose(seg['chord'])
        for comp in COMPONENT_NAMES:
            result[comp].append({
                'start': seg['start'],
                'end': seg['end'],
                'value': components[comp],
            })
    return result


def _merge_adjacent(segments: List[Dict], key: str = 'value') -> List[Dict]:
    """Merge adjacent segments that share the same value."""
    if not segments:
        return []
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        if seg[key] == merged[-1][key]:
            merged[-1]['end'] = seg['end']
        else:
            merged.append(dict(seg))
    return merged


def compute_diff(gt_segments: List[Dict], pred_segments: List[Dict]) -> List[Dict]:
    """Compute a time-aligned diff between GT and predicted chord sequences.

    Uses mir_eval-style interval overlap to align segments and mark matches.
    Returns a flat list of dicts with keys:
        start, end, gt_chord, pred_chord, match
    """
    if not gt_segments or not pred_segments:
        return []

    all_boundaries = sorted(set(
        [s['start'] for s in gt_segments] +
        [s['end'] for s in gt_segments] +
        [s['start'] for s in pred_segments] +
        [s['end'] for s in pred_segments]
    ))

    gt_idx = 0
    pred_idx = 0
    diff = []

    for i in range(len(all_boundaries) - 1):
        t_start = all_boundaries[i]
        t_end = all_boundaries[i + 1]
        if t_end <= t_start:
            continue

        mid = (t_start + t_end) / 2

        while gt_idx < len(gt_segments) - 1 and gt_segments[gt_idx]['end'] <= mid:
            gt_idx += 1
        while pred_idx < len(pred_segments) - 1 and pred_segments[pred_idx]['end'] <= mid:
            pred_idx += 1

        gt_chord = gt_segments[gt_idx]['chord'] if gt_idx < len(gt_segments) else 'N'
        pred_chord = pred_segments[pred_idx]['chord'] if pred_idx < len(pred_segments) else 'N'

        diff.append({
            'start': round(t_start, 4),
            'end': round(t_end, 4),
            'gt_chord': gt_chord,
            'pred_chord': pred_chord,
            'match': gt_chord == pred_chord,
        })

    return _merge_adjacent_diff(diff)


def _merge_adjacent_diff(diff: List[Dict]) -> List[Dict]:
    """Merge adjacent diff entries with same gt/pred/match."""
    if not diff:
        return []
    merged = [dict(diff[0])]
    for d in diff[1:]:
        prev = merged[-1]
        if (d['gt_chord'] == prev['gt_chord'] and
                d['pred_chord'] == prev['pred_chord'] and
                d['match'] == prev['match']):
            prev['end'] = d['end']
        else:
            merged.append(dict(d))
    return merged


def compute_component_diff(
    gt_segments: List[Dict],
    pred_segments: List[Dict],
) -> Dict[str, List[Dict]]:
    """Compute per-component diffs between GT and pred.

    Returns a dict mapping component name -> list of
    ``{"start", "end", "gt_value", "pred_value", "match"}`` dicts.
    """
    gt_decomposed = decompose_segments(gt_segments)
    pred_decomposed = decompose_segments(pred_segments)

    all_boundaries = sorted(set(
        [s['start'] for s in gt_segments] +
        [s['end'] for s in gt_segments] +
        [s['start'] for s in pred_segments] +
        [s['end'] for s in pred_segments]
    ))

    result: Dict[str, List[Dict]] = {}

    for comp in COMPONENT_NAMES:
        gt_comp = gt_decomposed[comp]
        pred_comp = pred_decomposed[comp]
        gt_i, pred_i = 0, 0
        comp_diff = []

        for i in range(len(all_boundaries) - 1):
            t_start = all_boundaries[i]
            t_end = all_boundaries[i + 1]
            if t_end <= t_start:
                continue

            mid = (t_start + t_end) / 2
            while gt_i < len(gt_comp) - 1 and gt_comp[gt_i]['end'] <= mid:
                gt_i += 1
            while pred_i < len(pred_comp) - 1 and pred_comp[pred_i]['end'] <= mid:
                pred_i += 1

            gt_val = gt_comp[gt_i]['value'] if gt_i < len(gt_comp) else 'N'
            pred_val = pred_comp[pred_i]['value'] if pred_i < len(pred_comp) else 'N'

            comp_diff.append({
                'start': round(t_start, 4),
                'end': round(t_end, 4),
                'gt_value': gt_val,
                'pred_value': pred_val,
                'match': gt_val == pred_val,
            })

        result[comp] = _merge_adjacent_component_diff(comp_diff)

    return result


def _merge_adjacent_component_diff(diff: List[Dict]) -> List[Dict]:
    if not diff:
        return []
    merged = [dict(diff[0])]
    for d in diff[1:]:
        prev = merged[-1]
        if (d['gt_value'] == prev['gt_value'] and
                d['pred_value'] == prev['pred_value']):
            prev['end'] = d['end']
        else:
            merged.append(dict(d))
    return merged


def scan_lab_directory(directory: str) -> List[str]:
    """Return sorted list of track names (without .lab extension) in a directory."""
    if not os.path.isdir(directory):
        return []
    files = glob.glob(os.path.join(directory, '*.lab'))
    return sorted(Path(f).stem for f in files)


def scan_inference_dirs(base_dir: str) -> List[Dict]:
    """Scan for inference output directories under base_dir."""
    if not os.path.isdir(base_dir):
        return []
    dirs = []
    for entry in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, entry)
        if os.path.isdir(full):
            lab_count = len(glob.glob(os.path.join(full, '*.lab')))
            if lab_count > 0:
                dirs.append({'name': entry, 'path': full, 'track_count': lab_count})
    return dirs


def get_track_data(
    track_id: str,
    gt_dir: Optional[str] = None,
    pred_dir: Optional[str] = None,
) -> Dict:
    """Load full track data including segments, decomposition, and diff."""
    data: Dict = {'track_id': track_id}

    gt_path = os.path.join(gt_dir, f'{track_id}.lab') if gt_dir else None
    pred_path = os.path.join(pred_dir, f'{track_id}.lab') if pred_dir else None

    gt_segments = None
    pred_segments = None

    if gt_path and os.path.isfile(gt_path):
        gt_segments = parse_lab_file(gt_path)
        data['gt'] = {
            'segments': gt_segments,
            'decomposed': decompose_segments(gt_segments),
        }

    if pred_path and os.path.isfile(pred_path):
        pred_segments = parse_lab_file(pred_path)
        data['pred'] = {
            'segments': pred_segments,
            'decomposed': decompose_segments(pred_segments),
        }

    if gt_segments and pred_segments:
        data['diff'] = compute_diff(gt_segments, pred_segments)
        data['component_diff'] = compute_component_diff(gt_segments, pred_segments)

        total_dur = sum(d['end'] - d['start'] for d in data['diff'])
        match_dur = sum(d['end'] - d['start'] for d in data['diff'] if d['match'])
        data['match_ratio'] = round(match_dur / total_dur, 4) if total_dur > 0 else 0.0

    return data
