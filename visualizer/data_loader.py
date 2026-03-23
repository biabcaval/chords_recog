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
from utils.chord_decomposition import (
    ChordDecomposer, ChordReassembler, COMPONENT_NAMES, CHORD_VOCAB, CHORD_VOCAB_IDX,
)
import re as _re
import copy as _copy


_decomposer = ChordDecomposer()
_reassembler = ChordReassembler()


def debug_chord_parsing(chord_label: str) -> Dict:
    """Trace every phase of chord decomposition and reassembly.

    Returns a dict with the output of each pipeline stage so the
    visualizer can display them interactively.
    """
    d = _decomposer
    result: Dict = {'input': chord_label, 'phases': []}

    # Phase 0: input validation
    if chord_label in ('N', 'X', '', None):
        components = {c: 'N' for c in COMPONENT_NAMES}
        result['phases'].append({
            'name': 'Input Validation',
            'description': f'Special token "{chord_label}" — all components set to N',
            'output': dict(components),
        })
        result['final_components'] = components
        result['indices'] = d.to_indices(components)
        result['reassembled'] = _reassembler.reassemble(components)
        result['round_trip_match'] = (result['reassembled'] == chord_label)
        return result

    result['phases'].append({
        'name': 'Input Validation',
        'description': f'Not a special token, proceeding to parse',
        'output': {'chord_label': chord_label},
    })

    # Phase 1: _parse_chord
    try:
        root, quality, bass = d._parse_chord(chord_label)
    except Exception as e:
        result['phases'].append({
            'name': 'Parse Chord',
            'description': f'ERROR: {e}',
            'output': {'error': str(e)},
        })
        result['final_components'] = {c: 'N' for c in COMPONENT_NAMES}
        return result

    result['phases'].append({
        'name': 'Parse Chord (_parse_chord)',
        'description': 'Split label into root, quality, bass. Normalize flats → sharps. Resolve degree-bass (e.g. /5 → note).',
        'output': {'root': root, 'quality': quality, 'bass': bass},
    })

    components = {c: 'N' for c in COMPONENT_NAMES}
    if root is not None:
        components['root'] = root
    if bass is not None and bass != root:
        components['bass'] = bass

    if not quality and root is not None:
        components['triad'] = 'maj'
        result['phases'].append({
            'name': 'Default Quality',
            'description': 'No explicit quality → default to major triad',
            'output': _copy.deepcopy(components),
        })
    elif quality:
        # Phase 2a: extract paren content
        paren_extensions, omit_notes = d._extract_paren_content(quality)
        shorthand = _re.sub(r'\([^)]*\)', '', quality).strip()

        result['phases'].append({
            'name': 'Phase 1: Extract Parenthetical Content',
            'description': 'Separate (...) groups into extensions and * omit notes',
            'output': {
                'quality': quality,
                'shorthand_without_parens': shorthand,
                'paren_extensions': paren_extensions,
                'omit_notes': omit_notes,
            },
        })

        # Phase 2b: process shorthand
        before = _copy.deepcopy(components)
        d._process_shorthand(shorthand.lower(), components)
        result['phases'].append({
            'name': 'Phase 2: Process Shorthand',
            'description': f'Process "{shorthand}" — extract triad type and inline extensions',
            'output': _copy.deepcopy(components),
            'changes': _diff_components(before, components),
        })

        # Phase 3: implied tones
        before = _copy.deepcopy(components)
        d._add_implied_tones(shorthand, components)
        result['phases'].append({
            'name': 'Phase 3: Add Implied Tones',
            'description': 'Shorthand 9→implies 7th, 11→implies 7th+9th, 13→implies 7th+9th+11th',
            'output': _copy.deepcopy(components),
            'changes': _diff_components(before, components),
        })

        # Phase 4: paren extensions
        before = _copy.deepcopy(components)
        d._apply_paren_extensions(paren_extensions, components)
        result['phases'].append({
            'name': 'Phase 4: Apply Paren Extensions',
            'description': f'Apply parenthetical extensions: {paren_extensions}',
            'output': _copy.deepcopy(components),
            'changes': _diff_components(before, components),
        })

        # Phase 5: omit rules
        before = _copy.deepcopy(components)
        d._apply_omit_rules(omit_notes, components)
        result['phases'].append({
            'name': 'Phase 5: Apply Omit Rules',
            'description': f'Handle * omit notation: {omit_notes}' if omit_notes else 'No omit rules to apply',
            'output': _copy.deepcopy(components),
            'changes': _diff_components(before, components),
        })

    # Indices
    indices = d.to_indices(components)
    result['phases'].append({
        'name': 'Convert to Indices',
        'description': 'Map component strings to vocabulary indices',
        'output': indices,
    })

    # Reassemble
    reassembled = _reassembler.reassemble(components)
    result['phases'].append({
        'name': 'Reassemble (Round-trip)',
        'description': 'Reconstruct chord label from components using ChordReassembler',
        'output': {
            'reassembled': reassembled,
            'matches_input': reassembled == chord_label,
        },
    })

    result['final_components'] = components
    result['indices'] = indices
    result['reassembled'] = reassembled
    result['round_trip_match'] = (reassembled == chord_label)

    return result


def _diff_components(before: Dict[str, str], after: Dict[str, str]) -> Dict[str, Dict]:
    """Return which components changed between two snapshots."""
    changes = {}
    for comp in COMPONENT_NAMES:
        if before.get(comp) != after.get(comp):
            changes[comp] = {'from': before.get(comp, 'N'), 'to': after.get(comp, 'N')}
    return changes

KNOWN_DATASETS = [
    'billboard', 'dj_avan_songbook1', 'dj_avan_songbook2',
    'jaah', 'queen', 'robbiewilliams', 'rwc',
]

ANNOTATION_SUBDIRS = ['annotations', 'lab']


def scan_datasets(data_root: str) -> List[Dict]:
    """Discover all datasets under data_root with their annotation directories.

    Looks for ``{data_root}/{dataset}/{annotations|lab}/*.lab`` for each
    known dataset name, and also discovers any other subdirectory that
    contains a recognised annotation folder.
    """
    if not data_root or not os.path.isdir(data_root):
        return []

    results = []
    seen = set()

    for entry in sorted(os.listdir(data_root)):
        ds_path = os.path.join(data_root, entry)
        if not os.path.isdir(ds_path):
            continue

        ann_dir = _find_annotation_dir(ds_path)
        if ann_dir is None:
            continue

        lab_files = glob.glob(os.path.join(ann_dir, '*.lab'))
        if not lab_files:
            continue

        total_duration = 0.0
        chord_counts: Dict[str, int] = {}
        for lf in lab_files:
            segs = parse_lab_file(lf)
            for seg in segs:
                total_duration += seg['end'] - seg['start']
                chord_counts[seg['chord']] = chord_counts.get(seg['chord'], 0) + 1

        results.append({
            'name': entry,
            'annotation_dir': ann_dir,
            'track_count': len(lab_files),
            'total_duration_s': round(total_duration, 1),
            'unique_chords': len(chord_counts),
        })
        seen.add(entry)

    return results


def _find_annotation_dir(dataset_path: str) -> Optional[str]:
    """Find the annotation subdirectory within a dataset folder."""
    for sub in ANNOTATION_SUBDIRS:
        candidate = os.path.join(dataset_path, sub)
        if os.path.isdir(candidate):
            return candidate
    if glob.glob(os.path.join(dataset_path, '*.lab')):
        return dataset_path
    return None


def search_chord_in_datasets(data_root: str, query: str, exact: bool = False) -> Dict:
    """Search for a chord string across all GT datasets.

    Args:
        data_root: Root directory containing dataset folders.
        query: Chord string to search for.
        exact: If True, match exactly. Otherwise substring (case-insensitive).

    Returns:
        Dict with 'results' list and 'summary' aggregate info.
    """
    datasets = scan_datasets(data_root)
    results: List[Dict] = []
    dataset_counts: Dict[str, int] = {}
    track_set: set = set()

    for ds in datasets:
        ann_dir = ds['annotation_dir']
        lab_files = sorted(glob.glob(os.path.join(ann_dir, '*.lab')))

        for lab_path in lab_files:
            track_name = Path(lab_path).stem
            segments = parse_lab_file(lab_path)

            for i, seg in enumerate(segments):
                chord = seg['chord']
                matched = (chord == query) if exact else (query.lower() in chord.lower())
                if not matched:
                    continue

                results.append({
                    'dataset': ds['name'],
                    'track': track_name,
                    'chord': chord,
                    'start': round(seg['start'], 3),
                    'end': round(seg['end'], 3),
                    'duration': round(seg['end'] - seg['start'], 3),
                    'context_before': segments[i - 1]['chord'] if i > 0 else None,
                    'context_after': segments[i + 1]['chord'] if i < len(segments) - 1 else None,
                })

                dataset_counts[ds['name']] = dataset_counts.get(ds['name'], 0) + 1
                track_set.add(f"{ds['name']}/{track_name}")

    return {
        'query': query,
        'exact': exact,
        'total_occurrences': len(results),
        'datasets_matched': len(dataset_counts),
        'tracks_matched': len(track_set),
        'per_dataset': [{'dataset': k, 'count': v}
                        for k, v in sorted(dataset_counts.items(), key=lambda x: -x[1])],
        'results': results,
    }


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


def compute_segment_stats(segments: List[Dict]) -> Dict:
    """Compute statistics for a set of chord segments."""
    if not segments:
        return {}

    total_dur = sum(s['end'] - s['start'] for s in segments)
    chords = [s['chord'] for s in segments]

    chord_durations: Dict[str, float] = {}
    for s in segments:
        c = s['chord']
        chord_durations[c] = chord_durations.get(c, 0) + (s['end'] - s['start'])

    top_chords = sorted(chord_durations.items(), key=lambda x: -x[1])[:15]

    decomposed = decompose_segments(segments)
    comp_distributions: Dict[str, Dict[str, float]] = {}
    for comp in COMPONENT_NAMES:
        val_dur: Dict[str, float] = {}
        for entry in decomposed[comp]:
            v = entry['value']
            d = entry['end'] - entry['start']
            val_dur[v] = val_dur.get(v, 0) + d
        dist = {v: round(d / total_dur * 100, 2) for v, d in
                sorted(val_dur.items(), key=lambda x: -x[1]) if d > 0}
        comp_distributions[comp] = dist

    return {
        'total_duration_s': round(total_dur, 2),
        'num_segments': len(segments),
        'unique_chords': len(set(chords)),
        'changes_per_min': round(len(segments) / (total_dur / 60), 2) if total_dur > 0 else 0,
        'top_chords': [{'chord': c, 'duration_s': round(d, 2),
                        'pct': round(d / total_dur * 100, 1)}
                       for c, d in top_chords],
        'component_distributions': comp_distributions,
    }


def compute_head_accuracy(component_diff: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Compute per-head accuracy weighted by duration."""
    result = {}
    for comp in COMPONENT_NAMES:
        segs = component_diff.get(comp, [])
        if not segs:
            continue
        total_dur = sum(s['end'] - s['start'] for s in segs)
        match_dur = sum(s['end'] - s['start'] for s in segs if s['match'])
        accuracy = round(match_dur / total_dur * 100, 2) if total_dur > 0 else 0.0

        error_pairs: Dict[str, float] = {}
        for s in segs:
            if not s['match']:
                key = f"{s['gt_value']}\u2192{s['pred_value']}"
                error_pairs[key] = error_pairs.get(key, 0) + (s['end'] - s['start'])
        top_errors = sorted(error_pairs.items(), key=lambda x: -x[1])[:5]

        result[comp] = {
            'accuracy_pct': accuracy,
            'match_duration_s': round(match_dur, 2),
            'total_duration_s': round(total_dur, 2),
            'top_errors': [{'pair': p, 'duration_s': round(d, 2),
                            'pct': round(d / total_dur * 100, 1)}
                           for p, d in top_errors],
        }
    return result


def get_track_data(
    track_id: str,
    gt_dir: Optional[str] = None,
    pred_dir: Optional[str] = None,
) -> Dict:
    """Load full track data including segments, decomposition, diff, and stats."""
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
        data['gt_stats'] = compute_segment_stats(gt_segments)

    if pred_path and os.path.isfile(pred_path):
        pred_segments = parse_lab_file(pred_path)
        data['pred'] = {
            'segments': pred_segments,
            'decomposed': decompose_segments(pred_segments),
        }
        data['pred_stats'] = compute_segment_stats(pred_segments)

    if gt_segments and pred_segments:
        data['diff'] = compute_diff(gt_segments, pred_segments)
        data['component_diff'] = compute_component_diff(gt_segments, pred_segments)

        total_dur = sum(d['end'] - d['start'] for d in data['diff'])
        match_dur = sum(d['end'] - d['start'] for d in data['diff'] if d['match'])
        data['match_ratio'] = round(match_dur / total_dur, 4) if total_dur > 0 else 0.0

        data['head_accuracy'] = compute_head_accuracy(data['component_diff'])

    return data
