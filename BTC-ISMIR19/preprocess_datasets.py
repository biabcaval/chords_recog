#!/usr/bin/env python
"""
Standalone script to preprocess datasets for chord recognition.
This script will extract features from audio files and generate training data.
"""
import os
import sys
import argparse
import math
from multiprocessing import Pool

# Add the parent directory to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocess import Preprocess, FeatureTypes
from utils.hparams import HParams


def preprocess_datasets(config_path, root_dir, dataset_names, large_voca=True, num_workers=1, force=False):
    """
    Preprocess datasets for training.
    
    Args:
        config_path: Path to config YAML file
        root_dir: Root directory containing datasets
        dataset_names: List of dataset names to preprocess (e.g., ['billboard', 'jaah', 'rwc'])
        large_voca: Whether to use large vocabulary (170 chords) or majmin (25 chords)
        num_workers: Number of parallel workers for preprocessing
        force: Force preprocessing even if data already exists
    """
    # Load config
    print(f"Loading config from: {config_path}")
    config = HParams.load(config_path)
    
    # Override large_voca setting
    if large_voca:
        config.feature['large_voca'] = True
        config.model['num_chords'] = 170
        print("Using large vocabulary (170 chords)")
    else:
        config.feature['large_voca'] = False
        config.model['num_chords'] = 25
        print("Using majmin vocabulary (25 chords)")
    
    # Create preprocessor
    print(f"\nInitializing preprocessor...")
    print(f"Root directory: {root_dir}")
    print(f"Datasets: {dataset_names}")
    
    preprocessor = Preprocess(
        config=config,
        feature_to_use=FeatureTypes.cqt,
        dataset_names=dataset_names,
        root_dir=root_dir
    )
    
    # Get all files
    print("\nScanning for annotation and audio files...")
    all_files = preprocessor.get_all_files()
    
    if len(all_files) == 0:
        print("ERROR: No files found! Please check:")
        print(f"  1. Root directory exists: {root_dir}")
        print(f"  2. Dataset directories exist: {[os.path.join(root_dir, d) for d in dataset_names]}")
        print(f"  3. Annotation files exist in annotations/ directories")
        print(f"  4. Audio files exist in audio/ directories")
        return
    
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
    
    if not force:
        print("\nChecking for existing preprocessed data...")
        all_exist = True
        for dataset_name in dataset_names:
            if large_voca:
                result_path = os.path.join(root_dir, 'result', dataset_name + '_voca', mp3_string, feature_string)
            else:
                result_path = os.path.join(root_dir, 'result', dataset_name, mp3_string, feature_string)
            
            if os.path.exists(result_path):
                print(f"  {dataset_name}: Preprocessed data exists at {result_path}")
            else:
                print(f"  {dataset_name}: No preprocessed data found")
                all_exist = False
        
        if all_exist:
            print("\nAll datasets are already preprocessed!")
            print("Use --force to reprocess anyway.")
            return
    
    # Start preprocessing
    print("\n" + "="*60)
    print("Starting preprocessing...")
    print("="*60)
    
    if num_workers > 1:
        print(f"Using {num_workers} parallel workers")
        num_path_per_process = math.ceil(len(all_files) / num_workers)
        args = [all_files[i * num_path_per_process:(i + 1) * num_path_per_process] 
                for i in range(num_workers)]
        
        if large_voca:
            p = Pool(processes=num_workers)
            p.map(preprocessor.generate_labels_features_voca, args)
            p.close()
        else:
            p = Pool(processes=num_workers)
            p.map(preprocessor.generate_labels_features_new, args)
            p.close()
    else:
        print("Using single worker (sequential processing)")
        if large_voca:
            preprocessor.generate_labels_features_voca(all_files)
        else:
            preprocessor.generate_labels_features_new(all_files)
    
    print("\n" + "="*60)
    print("Preprocessing completed!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Preprocess datasets for chord recognition')
    parser.add_argument('--config', type=str, default='run_config.yaml',
                        help='Path to config YAML file (default: run_config.yaml)')
    parser.add_argument('--root_dir', type=str, default=None,
                        help='Root directory containing datasets (default: from config)')
    parser.add_argument('--datasets', type=str, nargs='+', 
                        default=['billboard', 'jaah', 'rwc'],
                        help='Dataset names to preprocess (default: billboard jaah rwc)')
    parser.add_argument('--large_voca', action='store_true',
                        help='Use large vocabulary (170 chords) instead of majmin (25 chords)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--force', action='store_true',
                        help='Force preprocessing even if data already exists')
    
    args = parser.parse_args()
    
    # Load config to get default root_dir if not provided
    config = HParams.load(args.config)
    root_dir = args.root_dir if args.root_dir else config.path['root_path']
    
    preprocess_datasets(
        config_path=args.config,
        root_dir=root_dir,
        dataset_names=args.datasets,
        large_voca=True,
        num_workers=args.num_workers,
        force=args.force
    )


if __name__ == '__main__':
    main()