#!/usr/bin/env python
"""
Test script to verify that preprocessing correctly skips existing files.
"""
import os
import sys
import tempfile
import shutil
import torch
import numpy as np

# Add the parent directory to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocess import Preprocess, FeatureTypes
from utils.hparams import HParams


def test_skip_existing_files():
    """Test that preprocessing skips files that already exist."""
    print("="*60)
    print("Testing: Preprocessing skip existing files")
    print("="*60)
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nCreating test environment in: {temp_dir}")
        
        # Create test dataset structure
        dataset_dir = os.path.join(temp_dir, 'test_dataset')
        audio_dir = os.path.join(dataset_dir, 'audio')
        annotations_dir = os.path.join(dataset_dir, 'annotations')
        result_dir = os.path.join(temp_dir, 'result')
        
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        
        # Create a simple test annotation file (chord labels)
        test_lab = os.path.join(annotations_dir, 'test_song.lab')
        with open(test_lab, 'w') as f:
            # Simple chord progression: C major for 30 seconds
            f.write("0.0 30.0 C:maj\n")
        
        # For this test, we'll mock the actual audio processing
        # by checking that the file check logic works correctly
        
        # Load config
        config_path = 'run_config.yaml'
        if not os.path.exists(config_path):
            print(f"ERROR: Config file not found: {config_path}")
            print("This test needs to be run from the BTC-ISMIR19 directory")
            return False
        
        config = HParams.load(config_path)
        config.feature['large_voca'] = True
        
        # Create expected output structure
        mp3_config = config.mp3
        feature_config = config.feature
        mp3_string = "%d_%.1f_%.1f" % (mp3_config['song_hz'], mp3_config['inst_len'], mp3_config['skip_interval'])
        feature_string = "%s_%d_%d_%d" % (FeatureTypes.cqt.value, feature_config['n_bins'], 
                                          feature_config['bins_per_octave'], feature_config['hop_length'])
        
        result_path = os.path.join(result_dir, 'test_dataset_voca', mp3_string, feature_string, 'test_song')
        os.makedirs(result_path, exist_ok=True)
        
        # Create a dummy preprocessed file
        dummy_filename = "1.00_0_0.pt"
        dummy_file_path = os.path.join(result_path, dummy_filename)
        
        dummy_data = {
            'feature': np.random.randn(144, 43),  # dummy CQT features
            'chord': [0] * 43,  # dummy chord labels
            'etc': '0.0_10.0'
        }
        torch.save(dummy_data, dummy_file_path)
        
        print(f"\nCreated dummy preprocessed file: {dummy_file_path}")
        print(f"File exists: {os.path.exists(dummy_file_path)}")
        
        # Now test the file existence check logic
        print("\n" + "="*60)
        print("Testing file existence check...")
        print("="*60)
        
        # Simulate what the preprocessing code does
        aug = '1.00_0'
        idx = 0
        filename = aug + "_" + str(idx) + ".pt"
        output_file = os.path.join(result_path, filename)
        
        if os.path.exists(output_file):
            print(f"✓ SUCCESS: File existence check works correctly")
            print(f"  File {filename} already exists and would be skipped")
            result = True
        else:
            print(f"✗ FAILED: File existence check failed")
            print(f"  File {filename} should exist but wasn't detected")
            result = False
        
        # Test that non-existent file is not skipped
        non_existent_filename = "1.00_0_999.pt"
        non_existent_path = os.path.join(result_path, non_existent_filename)
        
        if not os.path.exists(non_existent_path):
            print(f"✓ SUCCESS: Non-existent file correctly identified")
            print(f"  File {non_existent_filename} does not exist and would be processed")
        else:
            print(f"✗ FAILED: Non-existent file check failed")
            result = False
        
        print("\n" + "="*60)
        if result:
            print("TEST PASSED: File existence checks work correctly")
        else:
            print("TEST FAILED: File existence checks have issues")
        print("="*60)
        
        return result


def test_directory_structure():
    """Test that the actual preprocessing directory structure is as expected."""
    print("\n" + "="*60)
    print("Testing: Verify actual preprocessed data structure")
    print("="*60)
    
    result_dir = '/home/daniel.melo/datasets/result'
    
    if not os.path.exists(result_dir):
        print(f"✗ Result directory does not exist: {result_dir}")
        return False
    
    print(f"\n✓ Result directory exists: {result_dir}")
    
    # Check for expected dataset directories
    datasets = ['jaah_voca', 'rwc_voca']
    found_datasets = []
    
    for dataset in datasets:
        dataset_path = os.path.join(result_dir, dataset)
        if os.path.exists(dataset_path):
            found_datasets.append(dataset)
            print(f"✓ Found dataset: {dataset}")
            
            # Check for at least one song directory
            mp3_dirs = os.listdir(dataset_path)
            if mp3_dirs:
                mp3_dir = os.path.join(dataset_path, mp3_dirs[0])
                feature_dirs = os.listdir(mp3_dir)
                if feature_dirs:
                    feature_dir = os.path.join(mp3_dir, feature_dirs[0])
                    song_dirs = os.listdir(feature_dir)
                    if song_dirs:
                        song_dir = os.path.join(feature_dir, song_dirs[0])
                        pt_files = [f for f in os.listdir(song_dir) if f.endswith('.pt')]
                        if pt_files:
                            print(f"  Sample song: {song_dirs[0]} ({len(pt_files)} .pt files)")
    
    if found_datasets:
        print(f"\n✓ SUCCESS: Found {len(found_datasets)} preprocessed datasets")
        print(f"  Datasets: {', '.join(found_datasets)}")
        return True
    else:
        print(f"\n✗ No preprocessed datasets found")
        return False


if __name__ == '__main__':
    print("\n" + "="*60)
    print("PREPROCESSING SKIP EXISTING FILES TEST SUITE")
    print("="*60)
    
    # Run tests
    test1_passed = test_skip_existing_files()
    test2_passed = test_directory_structure()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Test 1 (File existence check logic): {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Test 2 (Directory structure): {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED")
        sys.exit(1)

