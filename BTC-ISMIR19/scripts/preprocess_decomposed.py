#!/usr/bin/env python
# encoding: utf-8
"""
Preprocess datasets for chord recognition with decomposed structure (9 components).

This script preprocesses audio datasets and converts chord labels to the
9-component decomposed format (root, bass, triad, misc, 6th, 7th, 9th, 11th, 13th).
"""

import os
import sys
import argparse
import math
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import torch
import numpy as np

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocess import Preprocess, FeatureTypes
from utils.hparams import HParams
from utils.chord_decomposition import ChordDecomposer
from utils.mir_eval_modules import idx2voca_chord


def decompose_preprocessed_data(data_dir, output_dir, force=False, log_file=None):
    """
    Convert preprocessed data from Step 1 to 9-component decomposed format.

    Prefers the ``original_chord_labels`` field (full chord strings with
    extensions) when available.  Falls back to converting 170-class integer
    indices via ``idx2voca_chord`` for legacy .pt files.
    
    Args:
        data_dir: Directory with preprocessed .pt files
        output_dir: Directory to save decomposed files
        force: Overwrite existing files
        log_file: Optional path to write a CSV mapping each unique chord label
                  to its 9 decomposed components.
    """
    from collections import Counter

    COMP_NAMES = ['root', 'bass', 'triad', 'misc', '6th', '7th', '9th', '11th', '13th']

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    decomposer = ChordDecomposer()
    idx2chord = idx2voca_chord()
    
    pt_files = list(data_path.rglob('*.pt'))
    
    if not pt_files:
        print(f"No .pt files found in {data_dir}")
        return False
    
    print(f"Found {len(pt_files)} files to convert")
    
    successful = 0
    failed = []
    label_source_counts = Counter()
    extension_counts = Counter()
    unique_decompositions = {}

    for pt_file in tqdm(pt_files, desc="Converting to decomposed format"):
        try:
            data = torch.load(pt_file, map_location='cpu', weights_only=False)
            
            if isinstance(data, dict):
                features = data.get('feature', data.get('cqt'))
            else:
                features = data[0] if len(data) > 0 else None
            
            if features is None:
                failed.append((pt_file.name, "No features found"))
                continue

            # --- Resolve chord labels (prefer full strings) ----------------
            if isinstance(data, dict) and 'original_chord_labels' in data:
                chord_list = list(data['original_chord_labels'])
                label_source_counts['original_chord_labels'] += 1
            elif isinstance(data, dict) and 'chord_str' in data:
                chord_list = list(data['chord_str'])
                label_source_counts['chord_str'] += 1
            else:
                chords = data.get('chord') if isinstance(data, dict) else data[1]
                if isinstance(chords, torch.Tensor):
                    if chords.dtype in [torch.long, torch.int]:
                        chord_list = [idx2chord.get(int(c), 'N') for c in chords]
                    else:
                        chord_list = [str(c) for c in chords]
                elif isinstance(chords, np.ndarray):
                    if chords.dtype in (np.int64, np.int32, np.int16):
                        chord_list = [idx2chord.get(int(c), 'N') for c in chords]
                    else:
                        chord_list = [str(c) for c in chords]
                elif isinstance(chords, list) and chords and isinstance(chords[0], int):
                    chord_list = [idx2chord.get(c, 'N') for c in chords]
                else:
                    chord_list = list(chords)
                label_source_counts['index_fallback'] += 1

            # --- Decompose -------------------------------------------------
            decomposed_list = []
            for chord_str in chord_list:
                try:
                    decomposed = decomposer.decompose(chord_str)
                    decomposed_list.append(decomposed)
                except Exception:
                    decomposed_list.append({name: 'N' for name in COMP_NAMES})

                if chord_str not in unique_decompositions:
                    unique_decompositions[chord_str] = decomposed_list[-1]

            for d in decomposed_list:
                for ext in ('9th', '11th', '13th'):
                    if d.get(ext, 'N') != 'N':
                        extension_counts[ext] += 1
                if d.get('bass', 'N') != 'N':
                    extension_counts['bass_inversion'] += 1
            
            # --- Write output -----------------------------------------------
            rel_path = pt_file.relative_to(data_path)
            output_file = output_path / rel_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if output_file.exists() and not force:
                successful += 1
                continue
            
            decomposed_data = {
                'feature': features,
                'decomposed_chord': decomposed_list,
                'original_chord_labels': chord_list,
                'original_chords': chord_list,
            }
            
            torch.save(decomposed_data, output_file)
            successful += 1
            
        except Exception as e:
            failed.append((pt_file.name, str(e)))
    
    # --- Summary -----------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Conversion Complete!")
    print(f"Successfully converted: {successful}/{len(pt_files)}")
    print(f"Output directory: {output_dir}")

    print(f"\nLabel source breakdown:")
    for source, count in label_source_counts.most_common():
        marker = " (full strings)" if source == 'original_chord_labels' else \
                 " (WARNING: 170-idx fallback, extensions lost)" if source == 'index_fallback' else ""
        print(f"  - {source}: {count} files{marker}")

    if extension_counts:
        print(f"\nExtension / inversion frames detected:")
        for ext, count in extension_counts.most_common():
            print(f"  - {ext}: {count} frames")
    else:
        print(f"\nWARNING: No extensions (9th/11th/13th) or bass inversions detected.")
        print(f"  If the .lab files contain extensions, the preprocessing may")
        print(f"  still be using the 170-index fallback path.")

    print(f"\nUnique chord labels decomposed: {len(unique_decompositions)}")

    # --- Write decomposition log -------------------------------------------
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as f:
            header = 'original_label,' + ','.join(COMP_NAMES)
            f.write(header + '\n')
            for label in sorted(unique_decompositions.keys()):
                comps = unique_decompositions[label]
                vals = [comps.get(c, 'N') for c in COMP_NAMES]
                f.write(f'{label},' + ','.join(vals) + '\n')
        print(f"Decomposition log written to: {log_path}")

        non_trivial = [l for l, d in unique_decompositions.items()
                       if any(d.get(c, 'N') != 'N'
                              for c in ('9th', '11th', '13th'))
                       or (d.get('bass', 'N') != 'N')]
        if non_trivial:
            print(f"\nSample labels with extensions / inversions ({min(len(non_trivial), 20)} shown):")
            for label in non_trivial[:20]:
                d = unique_decompositions[label]
                parts = [f"{c}={d.get(c,'N')}" for c in COMP_NAMES if d.get(c, 'N') != 'N']
                print(f"  {label:30s} -> {', '.join(parts)}")

    if failed:
        reason_counts = Counter(reason for _, reason in failed)
        print(f"\nFailed: {len(failed)}")
        print("Failure breakdown:")
        for reason, count in reason_counts.most_common():
            print(f"  - {reason}: {count} files")
        if len(failed) <= 10:
            print("\nFailed files:")
            for filename, reason in failed:
                print(f"  - {filename}: {reason}")
    
    return successful > 0


def preprocess_datasets_decomposed(config_path, root_dir, dataset_names, 
                                   num_workers=1, force=False, log_file=None):
    """
    Preprocess datasets to decomposed format.
    
    Args:
        config_path: Path to config YAML file
        root_dir: Root directory containing datasets
        dataset_names: List of dataset names to preprocess
        num_workers: Number of parallel workers
        force: Force preprocessing even if exists
    """
    
    # Step 1: Preprocess with standard pipeline (170 chords)
    print("\n" + "="*70)
    print("STEP 1: Standard Preprocessing (170 chords)")
    print("="*70)
    
    # Load config
    config = HParams.load(config_path)
    config.feature['large_voca'] = True
    config.model['num_chords'] = 170
    
    # Create preprocessor
    preprocessor = Preprocess(
        config=config,
        feature_to_use=FeatureTypes.cqt,
        dataset_names=dataset_names,
        root_dir=root_dir
    )
    
    # Get all files
    print(f"\nScanning for annotation and audio files...")
    all_files = preprocessor.get_all_files()
    
    if len(all_files) == 0:
        print("ERROR: No files found!")
        return False
    
    print(f"Found {len(all_files)} songs to process")
    
    # Group by dataset
    dataset_counts = {}
    for song_name, lab_path, mp3_path, save_path in all_files:
        dataset_name = save_path.split('/')[-1]
        if dataset_name not in dataset_counts:
            dataset_counts[dataset_name] = []
        dataset_counts[dataset_name].append((song_name, lab_path, mp3_path))
    
    print("\nDataset breakdown:")
    for dataset_name, songs in dataset_counts.items():
        print(f"  {dataset_name}: {len(songs)} songs")
    
    # Check if already preprocessed
    mp3_config = config.mp3
    feature_config = config.feature
    mp3_string = "%d_%.1f_%.1f" % (mp3_config['song_hz'], mp3_config['inst_len'], mp3_config['skip_interval'])
    feature_string = "%s_%d_%d_%d" % (FeatureTypes.cqt.value, feature_config['n_bins'], 
                                      feature_config['bins_per_octave'], feature_config['hop_length'])
    
    # Store intermediate paths for conversion
    intermediate_dirs = []
    
    if not force:
        print("\nChecking for existing preprocessed data...")
        needs_processing = []
        for dataset_name in dataset_names:
            result_path = os.path.join(root_dir, 'result', dataset_name + '_voca', mp3_string, feature_string)
            intermediate_dirs.append((dataset_name, result_path))
            
            if os.path.exists(result_path):
                print(f"  {dataset_name}: Preprocessed data exists")
            else:
                print(f"  {dataset_name}: Need to preprocess")
                needs_processing.append(dataset_name)
        
        if not needs_processing:
            print("\nAll datasets already preprocessed!")
        else:
            # Process only needed datasets
            all_files = [f for f in all_files if any(
                dataset in f[3] for dataset in needs_processing
            )]
    else:
        for dataset_name in dataset_names:
            result_path = os.path.join(root_dir, 'result', dataset_name + '_voca', mp3_string, feature_string)
            intermediate_dirs.append((dataset_name, result_path))
    
    if all_files:
        # Start preprocessing
        print("\n" + "="*60)
        print("Starting standard preprocessing...")
        print("="*60)
        
        if num_workers > 1:
            print(f"Using {num_workers} parallel workers")
            num_path_per_process = math.ceil(len(all_files) / num_workers)
            args = [all_files[i * num_path_per_process:(i + 1) * num_path_per_process] 
                    for i in range(num_workers)]
            
            p = Pool(processes=num_workers)
            p.map(preprocessor.generate_labels_features_voca, args)
            p.close()
        else:
            print("Using single worker (sequential processing)")
            preprocessor.generate_labels_features_voca(all_files)
    
    # Step 2: Convert to decomposed format
    print("\n" + "="*70)
    print("STEP 2: Converting to Decomposed Format (9 components)")
    print("="*70)
    
    for dataset_name, intermediate_dir in intermediate_dirs:
        if os.path.exists(intermediate_dir):
            output_dir = os.path.join(
                root_dir, 'result_decomposed', dataset_name + '_voca', 
                mp3_string, feature_string
            )
            
            print(f"\nConverting {dataset_name}...")
            print(f"  Input:  {intermediate_dir}")
            print(f"  Output: {output_dir}")
            
            ds_log = None
            if log_file:
                ds_log = str(Path(log_file).with_suffix('')) + f'_{dataset_name}.csv'
            decompose_preprocessed_data(
                intermediate_dir, 
                output_dir, 
                force=force,
                log_file=ds_log,
            )
        else:
            print(f"\nSkipping {dataset_name} (not found at {intermediate_dir})")
    
    print("\n" + "="*70)
    print("Complete! Preprocessed decomposed data ready for training.")
    print("="*70)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess datasets for chord recognition with decomposed structure'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='run_config.yaml',
        help='Path to config YAML file (default: run_config.yaml)'
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        default=None,
        help='Root directory containing datasets (default: from config)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['billboard', 'dj_avan_songbook1', 'dj_avan_songbook2'],
        help='Dataset names to preprocess (default: billboard dj_avan_songbook1 dj_avan_songbook2)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of parallel workers (default: 8)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force preprocessing even if data already exists'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='Write a CSV log mapping each unique chord label to its '
             '9 decomposed components (one file per dataset)'
    )
    
    args = parser.parse_args()
    
    # Load config to get default root_dir if not provided
    config = HParams.load(args.config)
    root_dir = args.root_dir if args.root_dir else config.path['root_path']
    
    success = preprocess_datasets_decomposed(
        config_path=args.config,
        root_dir=root_dir,
        dataset_names=args.datasets,
        num_workers=args.num_workers,
        force=args.force,
        log_file=args.log_file,
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
