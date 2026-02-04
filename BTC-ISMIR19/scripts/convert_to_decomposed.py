#!/usr/bin/env python
# encoding: utf-8
"""
Convert preprocessed data from 170-chord vocabulary to 8-component decomposed format.

Takes existing .pt files with large vocabulary chords and converts them to 
the new decomposed format (8 components, 49 total classes).
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.chord_decomposition import ChordDecomposer
from utils.chords import Chord


def convert_preprocessed_data(input_dir, output_dir, force=False):
    """
    Convert preprocessed data from 170-chord to decomposed format.
    
    Args:
        input_dir: Directory containing .pt files with old format
        output_dir: Directory to save converted files
        force: Overwrite existing files
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize decomposer
    decomposer = ChordDecomposer()
    
    # Find all .pt files
    pt_files = list(input_path.glob('*.pt'))
    
    if not pt_files:
        print(f"No .pt files found in {input_dir}")
        return False
    
    print(f"Found {len(pt_files)} files to convert")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    successful = 0
    failed = []
    
    for pt_file in tqdm(pt_files, desc="Converting"):
        try:
            # Load old format
            data = torch.load(pt_file, map_location='cpu')
            
            # Get features and chords
            if isinstance(data, dict):
                features = data.get('feature', data.get('cqt'))
                chord_labels = data.get('label', data.get('chord'))
            else:
                # Assume it's a tuple (feature, chord)
                features, chord_labels = data
            
            if features is None:
                failed.append((pt_file.name, "No features found"))
                continue
            
            # Convert chord labels to decomposed format
            # chord_labels can be indices or strings
            
            if torch.is_tensor(chord_labels):
                # It's indices - need to convert back to strings first
                # This is tricky without the original chord list
                # For now, skip or use placeholder
                print(f"Warning: {pt_file.name} has index-based labels, skipping")
                failed.append((pt_file.name, "Index-based labels (need original chord list)"))
                continue
            
            # Convert string labels to decomposed format
            decomposed_labels = []
            
            if isinstance(chord_labels, torch.Tensor):
                # Try to convert tensor to strings
                if chord_labels.dtype == torch.long or chord_labels.dtype == torch.int:
                    # These are indices, we can't convert without the mapping
                    failed.append((pt_file.name, "Tensor indices without mapping"))
                    continue
                else:
                    chord_list = [str(c) for c in chord_labels]
            else:
                chord_list = chord_labels if isinstance(chord_labels, list) else list(chord_labels)
            
            # Decompose each chord
            for chord_str in chord_list:
                try:
                    decomposed = decomposer.decompose(chord_str)
                    decomposed_labels.append(decomposed)
                except Exception as e:
                    # If decomposition fails, use the original chord as root
                    decomposed_labels.append({
                        'root': chord_str,
                        'bass': 'N',
                        'triad': 'N',
                        'misc': 'N',
                        '7': 'N',
                        '9': 'N',
                        '11': 'N',
                        '13': 'N'
                    })
            
            # Save in new format
            output_file = output_path / pt_file.name
            
            new_data = {
                'feature': features,
                'decomposed_chord': decomposed_labels,
                'original_chords': chord_list,
            }
            
            torch.save(new_data, output_file)
            successful += 1
            
        except Exception as e:
            failed.append((pt_file.name, str(e)))
            print(f"Error processing {pt_file.name}: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Conversion complete!")
    print(f"Successfully converted: {successful}/{len(pt_files)}")
    print(f"Failed: {len(failed)}")
    
    if failed and len(failed) <= 10:
        print("\nFailed files:")
        for filename, reason in failed:
            print(f"  - {filename}: {reason}")
    
    return successful > 0


def main():
    parser = argparse.ArgumentParser(
        description='Convert preprocessed data from 170-chord to decomposed format'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='/home/daniel.melo/datasets/result/billboard_voca/22050_10.0_5.0/cqt_144_24_2048',
        help='Input directory with preprocessed .pt files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/data/music/chord_recognition/result/isophonic_voca/22050_10.0_5.0/cqt_144_24_2048',
        help='Output directory for converted files'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing files'
    )
    
    args = parser.parse_args()
    
    success = convert_preprocessed_data(
        args.input_dir,
        args.output_dir,
        force=args.force
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
