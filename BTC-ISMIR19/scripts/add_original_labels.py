#!/usr/bin/env python
"""
Add original chord labels from .lab files to existing .pt files.

This script:
1. Scans all .pt files in result_decomposed/
2. Finds the corresponding .lab file for each song
3. Aligns chord labels to the time frames in the .pt file
4. Adds 'original_chord_labels' field with the full chord strings

Usage:
    python scripts/add_original_labels.py --data_root /path/to/datasets
    python scripts/add_original_labels.py --data_root /path/to/datasets --dry_run
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_lab_file(lab_path):
    """Load chord annotations from a .lab file."""
    annotations = []
    with open(lab_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    chord = parts[2]
                    annotations.append((start, end, chord))
                except ValueError:
                    continue
    return annotations


def get_chord_at_time(annotations, time):
    """Get the chord label at a specific time."""
    for start, end, chord in annotations:
        if start <= time < end:
            return chord
    return 'N'


def get_chords_for_segment(annotations, start_time, n_frames, time_interval=0.1):
    """
    Get chord labels for each frame in a segment.
    
    Args:
        annotations: list of (start, end, chord) from .lab file
        start_time: start time of the segment in seconds
        n_frames: number of frames in the segment
        time_interval: time per frame in seconds
        
    Returns:
        list of chord label strings
    """
    chords = []
    for i in range(n_frames):
        t = start_time + i * time_interval
        chord = get_chord_at_time(annotations, t)
        chords.append(chord)
    return chords


def scan_lab_files(data_root):
    """Scan for all .lab files and create song_name -> path mapping."""
    lab_files = {}
    data_root = Path(data_root)
    
    for dataset_dir in data_root.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name.startswith('result'):
            continue
        
        # Look in common annotation folders
        for ann_folder in ['annotations', 'lab', '']:
            ann_path = dataset_dir / ann_folder if ann_folder else dataset_dir
            if ann_path.exists():
                for lab_file in ann_path.glob('*.lab'):
                    song_name = lab_file.stem
                    # Normalize name for matching
                    normalized = song_name.lower().replace(' ', '_').replace('-', '_')
                    lab_files[song_name] = lab_file
                    lab_files[normalized] = lab_file
    
    return lab_files


def extract_song_name_from_pt_path(pt_path):
    """Extract song name from .pt file path."""
    # Path structure: result_decomposed/dataset_voca/config/song_folder/file.pt
    pt_path = Path(pt_path)
    song_folder = pt_path.parent.name
    return song_folder


def extract_segment_info_from_filename(filename):
    """
    Extract segment info from .pt filename.
    
    Format: stretch_shift_idx.pt
    Example: 1.00_0_42.pt -> stretch=1.00, shift=0, idx=42
    """
    name = Path(filename).stem
    parts = name.split('_')
    if len(parts) >= 3:
        try:
            stretch = float(parts[0])
            shift = int(parts[1])
            idx = int(parts[2])
            return stretch, shift, idx
        except ValueError:
            pass
    return 1.0, 0, 0


def calculate_segment_start_time(idx, inst_len=10.0, hop_len=5.0):
    """
    Calculate the start time of a segment based on its index.
    
    Args:
        idx: segment index from filename
        inst_len: length of each segment in seconds
        hop_len: hop between segments (usually inst_len/2 for 50% overlap)
        
    Returns:
        start time in seconds
    """
    # Segments typically have 50% overlap
    return idx * hop_len


def main():
    parser = argparse.ArgumentParser(description='Add original chord labels to .pt files')
    parser.add_argument('--data_root', required=True, help='Root directory with datasets')
    parser.add_argument('--dry_run', action='store_true', help='Just analyze, don\'t modify files')
    parser.add_argument('--time_interval', type=float, default=0.1, help='Time per frame (seconds)')
    parser.add_argument('--inst_len', type=float, default=10.0, help='Segment length (seconds)')
    parser.add_argument('--hop_len', type=float, default=5.0, help='Hop between segments (seconds)')
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    result_dir = data_root / 'result_decomposed'
    
    if not result_dir.exists():
        print(f"Error: result_decomposed directory not found at {result_dir}")
        return 1
    
    # Scan for .lab files
    print("Scanning for .lab files...")
    lab_files = scan_lab_files(data_root)
    print(f"Found {len(lab_files)} .lab file mappings")
    
    # Scan for .pt files
    print("\nScanning for .pt files...")
    pt_files = list(result_dir.glob('**/*.pt'))
    print(f"Found {len(pt_files)} .pt files")
    
    # Group .pt files by song
    songs = defaultdict(list)
    for pt_path in pt_files:
        song_name = extract_song_name_from_pt_path(pt_path)
        songs[song_name].append(pt_path)
    
    print(f"Found {len(songs)} unique songs")
    
    # Match songs to .lab files
    matched = 0
    unmatched = []
    
    for song_name in songs:
        # Try different name variations
        variants = [
            song_name,
            song_name.lower(),
            song_name.lower().replace(' ', '_'),
            song_name.lower().replace('-', '_'),
            song_name.replace('_', ' '),
        ]
        
        found = False
        for variant in variants:
            if variant in lab_files:
                matched += 1
                found = True
                break
        
        if not found:
            unmatched.append(song_name)
    
    print(f"\nMatched {matched}/{len(songs)} songs to .lab files")
    if unmatched and len(unmatched) <= 10:
        print(f"Unmatched songs: {unmatched}")
    elif unmatched:
        print(f"First 10 unmatched: {unmatched[:10]}...")
    
    if args.dry_run:
        print("\n[DRY RUN] Would process files to add original chord labels")
        
        # Show sample processing
        print("\n=== Sample Processing ===")
        for song_name, pt_paths in list(songs.items())[:3]:
            # Find lab file
            lab_path = None
            for variant in [song_name, song_name.lower(), song_name.lower().replace(' ', '_')]:
                if variant in lab_files:
                    lab_path = lab_files[variant]
                    break
            
            if not lab_path:
                continue
            
            print(f"\nSong: {song_name}")
            print(f"  Lab file: {lab_path}")
            print(f"  .pt files: {len(pt_paths)}")
            
            # Load lab annotations
            annotations = load_lab_file(lab_path)
            print(f"  Lab annotations: {len(annotations)}")
            print(f"  First 3 chords: {[a[2] for a in annotations[:3]]}")
            
            # Check one .pt file
            pt_path = pt_paths[0]
            stretch, shift, idx = extract_segment_info_from_filename(pt_path.name)
            print(f"  Sample .pt: {pt_path.name}")
            print(f"    stretch={stretch}, shift={shift}, idx={idx}")
            
            # Load .pt data
            data = torch.load(pt_path, weights_only=False)
            feature = data.get('feature')
            if feature is not None:
                if isinstance(feature, np.ndarray):
                    n_frames = feature.shape[1] if feature.ndim > 1 else feature.shape[0]
                else:
                    n_frames = feature.shape[1] if feature.dim() > 1 else feature.shape[0]
                
                # Swap dimensions if needed (feature is often n_bins x n_frames)
                if n_frames > 200:  # Likely n_bins, not n_frames
                    n_frames = feature.shape[0] if isinstance(feature, np.ndarray) else feature.shape[0]
                
                print(f"    n_frames={n_frames}")
                
                # Calculate start time and get chords
                start_time = calculate_segment_start_time(idx, args.inst_len, args.hop_len)
                # Apply stretch factor
                start_time = start_time / stretch
                
                chords = get_chords_for_segment(annotations, start_time, n_frames, args.time_interval)
                
                # Count extensions
                extensions = sum(1 for c in chords if '(' in c or '9' in c or '11' in c or '13' in c)
                print(f"    start_time={start_time:.2f}s")
                print(f"    First 5 chords: {chords[:5]}")
                print(f"    Chords with extensions: {extensions}/{len(chords)}")
        
        return 0
    
    # Actually process files
    print("\n=== Processing .pt files ===")
    
    updated = 0
    skipped = 0
    errors = 0
    
    for song_name, pt_paths in tqdm(songs.items(), desc="Processing songs"):
        # Find lab file
        lab_path = None
        for variant in [song_name, song_name.lower(), song_name.lower().replace(' ', '_'), 
                        song_name.lower().replace('-', '_')]:
            if variant in lab_files:
                lab_path = lab_files[variant]
                break
        
        if not lab_path:
            skipped += len(pt_paths)
            continue
        
        # Load lab annotations
        try:
            annotations = load_lab_file(lab_path)
        except Exception as e:
            print(f"Error loading {lab_path}: {e}")
            errors += len(pt_paths)
            continue
        
        # Process each .pt file for this song
        for pt_path in pt_paths:
            try:
                stretch, shift, idx = extract_segment_info_from_filename(pt_path.name)
                
                # Load .pt data
                data = torch.load(pt_path, weights_only=False)
                
                # Skip if already processed
                if 'original_chord_labels' in data:
                    skipped += 1
                    continue
                
                # Get frame count
                feature = data.get('feature')
                if feature is None:
                    skipped += 1
                    continue
                
                if isinstance(feature, np.ndarray):
                    # Feature shape is typically (n_bins, n_frames) = (144, 108)
                    if feature.ndim > 1:
                        n_frames = min(feature.shape)  # Take smaller dimension as n_frames
                        if feature.shape[0] < feature.shape[1]:
                            n_frames = feature.shape[0]
                        else:
                            n_frames = feature.shape[1]
                    else:
                        n_frames = feature.shape[0]
                else:
                    if feature.dim() > 1:
                        n_frames = min(feature.shape)
                        if feature.shape[0] < feature.shape[1]:
                            n_frames = feature.shape[0]
                        else:
                            n_frames = feature.shape[1]
                    else:
                        n_frames = feature.shape[0]
                
                # Typical n_frames should be around 100-110
                if n_frames > 150:
                    n_frames = 108  # Default
                
                # Calculate start time
                start_time = calculate_segment_start_time(idx, args.inst_len, args.hop_len)
                start_time = start_time / stretch  # Apply stretch factor
                
                # Get chord labels for this segment
                chords = get_chords_for_segment(annotations, start_time, n_frames, args.time_interval)
                
                # Add to data
                data['original_chord_labels'] = chords
                
                # Save
                torch.save(data, pt_path)
                updated += 1
                
            except Exception as e:
                print(f"Error processing {pt_path}: {e}")
                errors += 1
    
    print(f"\nDone!")
    print(f"  Updated: {updated}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
