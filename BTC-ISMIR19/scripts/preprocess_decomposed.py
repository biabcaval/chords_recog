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


def decompose_preprocessed_data(data_dir, output_dir, force=False):
    """
    Convert preprocessed data from 170-chord to 9-component decomposed format.
    
    Args:
        data_dir: Directory with preprocessed .pt files
        output_dir: Directory to save decomposed files
        force: Overwrite existing files
    """
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize decomposer
    decomposer = ChordDecomposer()
    
    # Find all .pt files recursively
    pt_files = list(data_path.rglob('*.pt'))
    
    if not pt_files:
        print(f"No .pt files found in {data_dir}")
        return False
    
    print(f"Found {len(pt_files)} files to convert")
    
    successful = 0
    failed = []
    
    for pt_file in tqdm(pt_files, desc="Converting to decomposed format"):
        try:
            # Load original data
            data = torch.load(pt_file, map_location='cpu')
            
            # Extract components
            if isinstance(data, dict):
                features = data.get('feature', data.get('cqt'))
                chords = data.get('chord_str', data.get('chord'))
            else:
                # Assume it's tuple/list
                features, chords = data[:2]
            
            if features is None:
                failed.append((pt_file.name, "No features found"))
                continue
            
            # Convert chords to strings if needed
            if isinstance(chords, torch.Tensor):
                if chords.dtype in [torch.long, torch.int]:
                    # Index-based, skip
                    failed.append((pt_file.name, "Index-based labels"))
                    continue
                chord_list = [str(c) for c in chords]
            elif isinstance(chords, np.ndarray):
                chord_list = [str(c) for c in chords]
            else:
                chord_list = list(chords)
            
            # Decompose each chord
            decomposed_list = []
            for chord_str in chord_list:
                try:
                    decomposed = decomposer.decompose(chord_str)
                    decomposed_list.append(decomposed)
                except Exception as e:
                    # Fallback: use chord as root
                    decomposed_list.append({
                        'root': chord_str,
                        'bass': 'N',
                        'triad': 'N',
                        'misc': 'N',
                        '7': 'N',
                        '9': 'N',
                        '11': 'N',
                        '13': 'N'
                    })
            
            # Create output file preserving directory structure
            rel_path = pt_file.relative_to(data_path)
            output_file = output_path / rel_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if exists and not forcing
            if output_file.exists() and not force:
                successful += 1
                continue
            
            # Save decomposed format
            decomposed_data = {
                'feature': features,
                'decomposed_chord': decomposed_list,
                'original_chords': chord_list,
            }
            
            torch.save(decomposed_data, output_file)
            successful += 1
            
        except Exception as e:
            failed.append((pt_file.name, str(e)))
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Conversion Complete!")
    print(f"Successfully converted: {successful}/{len(pt_files)}")
    print(f"Output directory: {output_dir}")
    
    if failed and len(failed) <= 10:
        print(f"Failed: {len(failed)}")
        print("\nFailed files:")
        for filename, reason in failed:
            print(f"  - {filename}: {reason}")
    
    return successful > 0


def preprocess_datasets_decomposed(config_path, root_dir, dataset_names, 
                                   num_workers=1, force=False):
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
            
            decompose_preprocessed_data(
                intermediate_dir, 
                output_dir, 
                force=force
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
    
    args = parser.parse_args()
    
    # Load config to get default root_dir if not provided
    config = HParams.load(args.config)
    root_dir = args.root_dir if args.root_dir else config.path['root_path']
    
    success = preprocess_datasets_decomposed(
        config_path=args.config,
        root_dir=root_dir,
        dataset_names=args.datasets,
        num_workers=args.num_workers,
        force=args.force
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
