#!/usr/bin/env python3
"""Analyze chord quality distribution across the balanced selection for split strategy."""

import os
import sys
from collections import Counter

DATA_ROOT = '/home/daniel.melo/datasets'
DATASETS = ['billboard', 'dj_avan_songbook1', 'dj_avan_songbook2',
            'jaah', 'queen', 'robbiewilliams', 'rwc']

DIR_MAP = {
    'chords_billboard_verified': 'billboard',
    'chords_djavan_songbook1_verified': 'dj_avan_songbook1',
    'chords_djavan_songbook2_verified': 'dj_avan_songbook2',
    'chords_jaah_verified': 'jaah',
    'chords_queen_verified': 'queen',
    'chords_robbie_verified': 'robbiewilliams',
    'chords_rwc_verified': 'rwc',
}


def get_quality_family(chord):
    if chord == 'N':
        return 'N'
    parts = chord.split(':')
    if len(parts) < 2:
        return 'maj'
    q = parts[1].split('/')[0].split('(')[0]
    families = {
        'maj': 'maj', '1': 'maj',
        'min': 'min',
        '7': 'dom7',
        'maj7': 'maj7',
        'min7': 'min7',
        'dim': 'dim', 'dim7': 'dim',
        'aug': 'aug',
        'sus2': 'sus', 'sus4': 'sus',
        'hdim7': 'hdim7',
    }
    if q in families:
        return families[q]
    if 'min' in q:
        return 'min_ext'
    if 'maj' in q:
        return 'maj_ext'
    if '7' in q:
        return 'dom7_ext'
    return 'other'


def get_genre_group(dataset):
    """Group datasets by broad genre for stratification."""
    groups = {
        'billboard': 'pop_rock',
        'queen': 'pop_rock',
        'robbiewilliams': 'pop_rock',
        'rwc': 'pop_rock',
        'dj_avan_songbook1': 'mpb',
        'dj_avan_songbook2': 'mpb',
        'jaah': 'jazz',
    }
    return groups.get(dataset, 'unknown')


def find_lab_on_vm(fname, ds_hint):
    norm_fname = fname.lower().replace(' ', '_')
    search = [ds_hint] + DATASETS if ds_hint else DATASETS
    for ds in search:
        adir = os.path.join(DATA_ROOT, ds, 'annotations')
        if not os.path.isdir(adir):
            continue
        for f in os.listdir(adir):
            if f.lower().replace(' ', '_') == norm_fname:
                return ds, os.path.join(adir, f)
    return None, None


def main():
    manifest_path = sys.argv[1] if len(sys.argv) > 1 else '/home/daniel.melo/balanced_selection.txt'

    with open(manifest_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    songs = []
    for line in lines:
        norm = line.replace('\\', '/')
        parts = norm.split('/')
        fname = parts[-1]
        parent = parts[-2].lower() if len(parts) >= 2 else ''
        ds_hint = DIR_MAP.get(parent)

        found_ds, lab_path = find_lab_on_vm(fname, ds_hint)
        if not lab_path:
            continue

        quality_raw = Counter()
        total_dur = 0
        with open(lab_path) as lf:
            for ll in lf:
                pp = ll.strip().split()
                if len(pp) >= 3:
                    dur = float(pp[1]) - float(pp[0])
                    if dur > 0:
                        qf = get_quality_family(pp[2])
                        quality_raw[qf] += dur
                        total_dur += dur

        songs.append({
            'stem': fname.replace('.lab', ''),
            'dataset': found_ds,
            'genre': get_genre_group(found_ds),
            'quality_raw': quality_raw,
            'total_dur': total_dur,
        })

    # --- Summary ---
    print(f"Loaded {len(songs)} songs\n")

    print("=== Songs per dataset ===")
    ds_counts = Counter(s['dataset'] for s in songs)
    for ds, c in sorted(ds_counts.items()):
        print(f"  {ds}: {c}")

    print("\n=== Songs per genre group ===")
    genre_counts = Counter(s['genre'] for s in songs)
    for g, c in sorted(genre_counts.items()):
        print(f"  {g}: {c}")

    print("\n=== Overall quality family distribution (duration-weighted) ===")
    total_qf = Counter()
    for s in songs:
        total_qf.update(s['quality_raw'])
    grand_total = sum(total_qf.values())
    for qf, dur in total_qf.most_common():
        print(f"  {qf:>10s}: {100*dur/grand_total:5.1f}%  ({dur:.0f}s)")

    print("\n=== Quality profile per dataset ===")
    for ds in sorted(ds_counts.keys()):
        ds_qf = Counter()
        ds_dur = 0
        for s in songs:
            if s['dataset'] == ds:
                ds_qf.update(s['quality_raw'])
                ds_dur += s['total_dur']
        top = ds_qf.most_common(8)
        profile = ', '.join(f'{q}:{100*d/ds_dur:.0f}%' for q, d in top)
        print(f"  {ds}: {profile}")

    print("\n=== Quality profile per genre ===")
    for genre in sorted(genre_counts.keys()):
        g_qf = Counter()
        g_dur = 0
        for s in songs:
            if s['genre'] == genre:
                g_qf.update(s['quality_raw'])
                g_dur += s['total_dur']
        top = g_qf.most_common(8)
        profile = ', '.join(f'{q}:{100*d/g_dur:.0f}%' for q, d in top)
        print(f"  {genre}: {profile}")

    # --- Harmonic complexity per song ---
    print("\n=== Harmonic complexity stats ===")
    complexities = []
    for s in songs:
        n_families = len([k for k in s['quality_raw'] if k != 'N'])
        ext_ratio = sum(v for k, v in s['quality_raw'].items()
                        if k in ('dom7_ext', 'min_ext', 'maj_ext', 'hdim7', 'dim', 'aug'))
        ext_pct = ext_ratio / s['total_dur'] if s['total_dur'] > 0 else 0
        complexities.append((s['stem'], s['genre'], n_families, ext_pct))

    for genre in sorted(genre_counts.keys()):
        g_songs = [c for c in complexities if c[1] == genre]
        avg_families = sum(c[2] for c in g_songs) / len(g_songs)
        avg_ext = sum(c[3] for c in g_songs) / len(g_songs)
        print(f"  {genre}: avg {avg_families:.1f} quality families/song, "
              f"avg {100*avg_ext:.1f}% extended chords")


if __name__ == '__main__':
    main()
