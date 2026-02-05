# encoding: utf-8
"""
Debug script to verify that chord indices are now properly converted to labels.
Run this on the VM to check the fix is working.
"""
import sys
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, '.')

from utils.hparams import HParams
from data.audio_dataset_structured import AudioDatasetStructured, get_idx2chord_mapping
from utils.chord_decomposition import CHORD_VOCAB, COMPONENT_NAMES

def main():
    print("=" * 60)
    print("DEBUG: Checking chord index to label conversion")
    print("=" * 60)
    
    # Load config
    config = HParams.load("run_config.yaml")
    
    # Get paths
    data_root = config.experiment.get('data_root', '/home/daniel.melo/datasets')
    dataset_names = config.experiment.get('dataset_names', ['billboard'])
    
    print(f"\nData root: {data_root}")
    print(f"Datasets: {dataset_names}")
    
    # Check idx2chord mapping first
    print("\n--- Checking idx2chord mapping ---")
    idx2chord = get_idx2chord_mapping()
    print(f"Mapping size: {len(idx2chord)} entries")
    print("Sample mappings:")
    for idx in [0, 1, 130, 169]:
        if idx in idx2chord:
            print(f"  {idx} -> '{idx2chord[idx]}'")
    
    # Create dataset
    print("\n--- Loading dataset ---")
    dataset = AudioDatasetStructured(
        config=config,
        data_root=data_root,
        dataset_names=dataset_names,
        kfold=0,
        mode='train',
        decompose=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("ERROR: Dataset is empty!")
        return
    
    # Get first sample
    print("\n--- Checking first sample ---")
    sample = dataset[0]
    
    print(f"Sample keys: {sample.keys()}")
    print(f"Feature shape: {sample['feature'].shape}")
    
    # Check chord data
    chord_data = sample['chord']
    print(f"\nChord type: {type(chord_data)}, len: {len(chord_data) if hasattr(chord_data, '__len__') else 'N/A'}")
    print(f"First 5 chords (raw): {chord_data[:5]}")
    
    # Check components
    print("\n--- Checking components ---")
    components = sample['components']
    
    for comp_name in COMPONENT_NAMES:
        comp_data = components[comp_name]
        unique_vals = torch.unique(comp_data).tolist()
        print(f"  {comp_name}: shape={comp_data.shape}, unique={unique_vals[:10]}...")
        
        # Show what the indices map to
        vocab = CHORD_VOCAB[comp_name]
        print(f"    -> Labels: {[vocab[i] for i in unique_vals[:10]]}")
    
    # Count non-N components
    print("\n--- Component distribution (first sample) ---")
    for comp_name in COMPONENT_NAMES:
        comp_data = components[comp_name]
        n_count = (comp_data == 0).sum().item()
        non_n_count = (comp_data != 0).sum().item()
        print(f"  {comp_name}: N={n_count}, non-N={non_n_count}")
    
    # Check multiple samples
    print("\n--- Checking multiple samples ---")
    total_non_n = {comp: 0 for comp in COMPONENT_NAMES}
    total_frames = 0
    
    n_samples = min(100, len(dataset))
    for i in range(n_samples):
        sample = dataset[i]
        components = sample['components']
        for comp_name in COMPONENT_NAMES:
            total_non_n[comp_name] += (components[comp_name] != 0).sum().item()
        total_frames += components['root'].shape[0]
    
    print(f"Analyzed {n_samples} samples, {total_frames} total frames:")
    for comp_name in COMPONENT_NAMES:
        pct = 100 * total_non_n[comp_name] / total_frames if total_frames > 0 else 0
        print(f"  {comp_name}: {total_non_n[comp_name]} non-N frames ({pct:.1f}%)")
    
    # Final verdict
    print("\n" + "=" * 60)
    if total_non_n['root'] == 0:
        print("ERROR: All root components are 'N' - something is still wrong!")
    else:
        print("SUCCESS: Chord decomposition appears to be working correctly!")
    print("=" * 60)

if __name__ == '__main__':
    main()
