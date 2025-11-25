#!/usr/bin/env python
"""
Demonstration script showing how the preprocessing now skips existing files.

This script demonstrates:
1. How to check what files already exist
2. How the preprocessing will skip existing files
3. Expected time savings
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.hparams import HParams
from utils.preprocess import FeatureTypes


def analyze_preprocessed_data(root_dir, datasets, large_voca=True):
    """
    Analyze what preprocessed data already exists.
    
    Args:
        root_dir: Root directory containing datasets
        datasets: List of dataset names
        large_voca: Whether using large vocabulary (170 chords) or majmin (25 chords)
    """
    print("="*60)
    print("ANALYZING PREPROCESSED DATA")
    print("="*60)
    
    # Load config
    config = HParams.load('run_config.yaml')
    mp3_config = config.mp3
    feature_config = config.feature
    
    mp3_string = "%d_%.1f_%.1f" % (mp3_config['song_hz'], mp3_config['inst_len'], 
                                    mp3_config['skip_interval'])
    feature_string = "%s_%d_%d_%d" % (FeatureTypes.cqt.value, feature_config['n_bins'], 
                                      feature_config['bins_per_octave'], feature_config['hop_length'])
    
    total_songs = 0
    total_files = 0
    total_size_mb = 0
    
    for dataset_name in datasets:
        if large_voca:
            result_path = os.path.join(root_dir, 'result', dataset_name + '_voca', 
                                      mp3_string, feature_string)
        else:
            result_path = os.path.join(root_dir, 'result', dataset_name, 
                                      mp3_string, feature_string)
        
        if os.path.exists(result_path):
            print(f"\n{dataset_name}:")
            print(f"  Path: {result_path}")
            
            # Count songs (directories)
            song_dirs = [d for d in os.listdir(result_path) 
                        if os.path.isdir(os.path.join(result_path, d))]
            
            dataset_files = 0
            dataset_size = 0
            
            for song_dir in song_dirs:
                song_path = os.path.join(result_path, song_dir)
                pt_files = [f for f in os.listdir(song_path) if f.endswith('.pt')]
                dataset_files += len(pt_files)
                
                # Calculate size
                for pt_file in pt_files:
                    file_path = os.path.join(song_path, pt_file)
                    dataset_size += os.path.getsize(file_path)
            
            dataset_size_mb = dataset_size / (1024 * 1024)
            
            print(f"  Songs: {len(song_dirs)}")
            print(f"  Files: {dataset_files}")
            print(f"  Size: {dataset_size_mb:.2f} MB")
            
            total_songs += len(song_dirs)
            total_files += dataset_files
            total_size_mb += dataset_size_mb
        else:
            print(f"\n{dataset_name}:")
            print(f"  ✗ No preprocessed data found at: {result_path}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total songs preprocessed: {total_songs}")
    print(f"Total .pt files: {total_files}")
    print(f"Total size: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
    
    if total_files > 0:
        print("\n✓ Preprocessed data exists!")
        print("  When you run preprocessing again, these files will be skipped.")
        print("  Only new or missing files will be processed.")
        
        # Estimate time savings
        # Rough estimate: ~0.5 seconds per file for CQT computation
        estimated_time_saved = total_files * 0.5
        print(f"\n  Estimated time saved by skipping: ~{estimated_time_saved/60:.1f} minutes")
    else:
        print("\n✗ No preprocessed data found.")
        print("  All files will be processed on the next run.")


def check_specific_song(root_dir, dataset_name, song_name, large_voca=True):
    """
    Check preprocessed files for a specific song.
    
    Args:
        root_dir: Root directory containing datasets
        dataset_name: Name of the dataset
        song_name: Name of the song
        large_voca: Whether using large vocabulary
    """
    print("="*60)
    print(f"CHECKING SPECIFIC SONG: {song_name}")
    print("="*60)
    
    # Load config
    config = HParams.load('run_config.yaml')
    mp3_config = config.mp3
    feature_config = config.feature
    
    mp3_string = "%d_%.1f_%.1f" % (mp3_config['song_hz'], mp3_config['inst_len'], 
                                    mp3_config['skip_interval'])
    feature_string = "%s_%d_%d_%d" % (FeatureTypes.cqt.value, feature_config['n_bins'], 
                                      feature_config['bins_per_octave'], feature_config['hop_length'])
    
    if large_voca:
        result_path = os.path.join(root_dir, 'result', dataset_name + '_voca', 
                                  mp3_string, feature_string, song_name)
    else:
        result_path = os.path.join(root_dir, 'result', dataset_name, 
                                  mp3_string, feature_string, song_name)
    
    print(f"\nPath: {result_path}")
    
    if os.path.exists(result_path):
        pt_files = [f for f in os.listdir(result_path) if f.endswith('.pt')]
        pt_files.sort()
        
        print(f"\n✓ Found {len(pt_files)} preprocessed files")
        
        if len(pt_files) > 0:
            print(f"\nSample files:")
            for f in pt_files[:5]:
                file_path = os.path.join(result_path, f)
                size_kb = os.path.getsize(file_path) / 1024
                print(f"  - {f} ({size_kb:.1f} KB)")
            
            if len(pt_files) > 5:
                print(f"  ... and {len(pt_files) - 5} more files")
        
        print(f"\nAll these files will be skipped during preprocessing!")
    else:
        print(f"\n✗ No preprocessed data found for this song")
        print(f"  All files will be generated on the next run")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze preprocessed data and demonstrate skip functionality'
    )
    parser.add_argument('--root_dir', type=str, 
                       default='/home/daniel.melo/datasets',
                       help='Root directory containing datasets')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['jaah', 'rwc'],
                       help='Dataset names to check')
    parser.add_argument('--song', type=str, default=None,
                       help='Check specific song (requires --dataset)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name for specific song check')
    
    args = parser.parse_args()
    
    if args.song and args.dataset:
        check_specific_song(args.root_dir, args.dataset, args.song, large_voca=True)
    else:
        analyze_preprocessed_data(args.root_dir, args.datasets, large_voca=True)


if __name__ == '__main__':
    main()

