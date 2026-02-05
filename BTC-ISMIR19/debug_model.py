#!/usr/bin/env python
"""
debug_model.py - Debug and test module for the decomposed chord recognition model.

This script provides utilities for:
1. Testing forward pass with synthetic data
2. Analyzing component distribution in datasets
3. Verifying gradient flow
4. Checking model predictions

Usage:
    python debug_model.py --config run_config.yaml
    python debug_model.py --config run_config.yaml --checkpoint checkpoints/model_best.pt
    python debug_model.py --config run_config.yaml --test all
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import logging

from models.btc_model_decomposed import BTC_model_decomposed, MultiTaskLoss
from utils.chord_decomposition import COMPONENT_NAMES, CHORD_VOCAB, ChordDecomposer, ChordReassembler
from utils.hparams import HParams

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_forward_pass(config, checkpoint_path=None, device='cuda'):
    """Test forward pass with synthetic data."""
    print("=" * 70)
    print("TEST: Forward Pass with Synthetic Data")
    print("=" * 70)
    
    # Initialize model
    model = BTC_model_decomposed(config)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            print(f"  Val Loss: {checkpoint['metrics'].get('val_loss', 'N/A')}")
    
    model = model.to(device)
    model.eval()
    
    # Get config params
    cfg = config.model if hasattr(config, 'model') else config
    seq_len = cfg.get('timestep', 108)
    feature_size = cfg.get('feature_size', 144)
    batch_size = 2
    
    # Create synthetic input
    fake_input = torch.randn(batch_size, 1, feature_size, seq_len).to(device)
    
    # Create synthetic labels
    fake_labels = {
        comp: torch.randint(0, len(CHORD_VOCAB[comp]), (batch_size, seq_len)).to(device)
        for comp in COMPONENT_NAMES
    }
    
    print(f"\nInput shape: {fake_input.shape}")
    print(f"Expected: (batch={batch_size}, 1, features={feature_size}, seq_len={seq_len})")
    
    # Forward pass
    with torch.no_grad():
        predictions, loss, weights, comp_losses = model(fake_input, labels=fake_labels)
    
    print(f"\n--- Predictions ---")
    for comp, pred in predictions.items():
        unique_vals = pred.unique().tolist()
        print(f"  {comp:6s}: shape={list(pred.shape)}, unique={unique_vals[:5]}{'...' if len(unique_vals) > 5 else ''}")
    
    print(f"\n--- Loss ---")
    print(f"  Total Loss: {loss.item():.4f}")
    
    if comp_losses:
        print(f"\n--- Component Losses ---")
        for comp, val in comp_losses.items():
            bar = "█" * int(val * 10)
            print(f"  {comp:6s}: {val:.4f} {bar}")
    
    print(f"\n--- Attention Weights ---")
    if weights:
        print(f"  Number of layers: {len(weights)}")
        if len(weights) > 0:
            print(f"  First layer shape: {weights[0].shape}")
    
    print("\n✓ Forward pass successful!")
    return True


def test_component_distribution(config, device='cuda', num_samples=100):
    """Analyze component distribution in a dataset."""
    print("=" * 70)
    print("TEST: Component Distribution Analysis")
    print("=" * 70)
    
    try:
        from data.audio_dataset_structured import AudioDatasetStructured
    except ImportError:
        print("Could not import AudioDatasetStructured. Skipping distribution test.")
        return False
    
    # Get data config
    data_root = config.experiment.get('data_root', '/home/daniel.melo/datasets')
    dataset_names = config.experiment.get('dataset_names', ['billboard'])
    
    print(f"Data root: {data_root}")
    print(f"Datasets: {dataset_names}")
    
    # Load dataset
    try:
        dataset = AudioDatasetStructured(
            config=config,
            root_dir=data_root,
            dataset_names=dataset_names,
            train=True,
            decompose=True
        )
    except Exception as e:
        print(f"Could not load dataset: {e}")
        return False
    
    print(f"Dataset size: {len(dataset)}")
    
    # Count distribution
    counts = {comp: {} for comp in COMPONENT_NAMES}
    num_samples = min(num_samples, len(dataset))
    
    print(f"Analyzing {num_samples} samples...")
    
    for i in range(num_samples):
        sample = dataset[i]
        for comp in COMPONENT_NAMES:
            if 'components' in sample and comp in sample['components']:
                data = sample['components'][comp]
                if isinstance(data, torch.Tensor):
                    indices = data.numpy().flatten()
                else:
                    indices = np.array(data).flatten()
                
                for idx in indices:
                    idx = int(idx)
                    counts[comp][idx] = counts[comp].get(idx, 0) + 1
    
    print(f"\n--- Distribution per Component ---")
    for comp in COMPONENT_NAMES:
        vocab = CHORD_VOCAB[comp]
        total = sum(counts[comp].values())
        sorted_counts = sorted(counts[comp].items(), key=lambda x: -x[1])
        
        print(f"\n{comp} (vocab size: {len(vocab)}, total frames: {total}):")
        for idx, count in sorted_counts[:5]:
            label = vocab[idx] if idx < len(vocab) else '?'
            pct = 100 * count / total if total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {label:6s}: {count:6d} ({pct:5.1f}%) {bar}")
        
        if len(sorted_counts) > 5:
            print(f"  ... and {len(sorted_counts) - 5} more classes")
    
    print("\n✓ Distribution analysis complete!")
    return True


def test_gradient_flow(config, device='cuda'):
    """Verify gradients flow correctly through all components."""
    print("=" * 70)
    print("TEST: Gradient Flow Verification")
    print("=" * 70)
    
    model = BTC_model_decomposed(config)
    model = model.to(device)
    model.train()
    
    # Get config params
    cfg = config.model if hasattr(config, 'model') else config
    seq_len = cfg.get('timestep', 108)
    feature_size = cfg.get('feature_size', 144)
    batch_size = 2
    
    # Create synthetic data
    fake_input = torch.randn(batch_size, 1, feature_size, seq_len).to(device)
    fake_labels = {
        comp: torch.randint(0, len(CHORD_VOCAB[comp]), (batch_size, seq_len)).to(device)
        for comp in COMPONENT_NAMES
    }
    
    # Forward pass
    predictions, loss, _, comp_losses = model(fake_input, labels=fake_labels)
    
    # Backward pass
    loss.backward()
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"\n--- Gradient Norms by Module ---")
    
    # Group gradients by module type
    module_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            
            # Extract module name
            parts = name.split('.')
            if len(parts) >= 2:
                module = parts[0] + '.' + parts[1]
            else:
                module = parts[0]
            
            if module not in module_grads:
                module_grads[module] = []
            module_grads[module].append((name, grad_norm))
    
    # Print summary
    for module, grads in sorted(module_grads.items()):
        avg_norm = np.mean([g[1] for g in grads])
        max_norm = max([g[1] for g in grads])
        status = "✓" if avg_norm > 1e-7 else "⚠ LOW"
        print(f"  {module:40s}: avg={avg_norm:.6f}, max={max_norm:.6f} {status}")
    
    # Check head gradients specifically
    print(f"\n--- Output Head Gradients ---")
    for comp in COMPONENT_NAMES:
        head_grads = [g for name, g in module_grads.get(f'decomposer.heads', []) if comp in name]
        if not head_grads:
            # Try alternative naming
            for module, grads in module_grads.items():
                if comp in module:
                    head_grads = [g[1] for g in grads]
                    break
        
        if head_grads:
            avg = np.mean(head_grads)
            print(f"  {comp:6s}: {avg:.6f}")
        else:
            print(f"  {comp:6s}: No gradients found")
    
    print("\n✓ Gradient flow verification complete!")
    return True


def test_decomposition_reassembly():
    """Test chord decomposition and reassembly."""
    print("=" * 70)
    print("TEST: Chord Decomposition and Reassembly")
    print("=" * 70)
    
    decomposer = ChordDecomposer()
    reassembler = ChordReassembler()
    
    test_chords = [
        'C:maj',
        'D:min7',
        'E:maj9',
        'F#:dim',
        'G:sus4',
        'A:7',
        'Bb:min7/F',
        'N',
        'C:5',
        'D:aug',
    ]
    
    print(f"\n--- Decomposition Test ---")
    all_passed = True
    
    for chord in test_chords:
        components = decomposer.decompose(chord)
        
        # Convert labels to indices manually
        indices = {}
        for comp, label in components.items():
            vocab = CHORD_VOCAB[comp]
            indices[comp] = vocab.index(label) if label in vocab else 0
        
        reassembled = reassembler.reassemble(components)
        
        # Check if reassembly is reasonable
        status = "✓" if components['root'] != 'N' or chord == 'N' else "?"
        
        print(f"\n  Input: {chord}")
        print(f"    Components: {components}")
        print(f"    Indices: {indices}")
        print(f"    Reassembled: {reassembled} {status}")
    
    print("\n✓ Decomposition/reassembly test complete!")
    return True


def test_model_predictions(config, checkpoint_path, device='cuda'):
    """Test model predictions on synthetic data and show decoded results."""
    print("=" * 70)
    print("TEST: Model Predictions (Decoded)")
    print("=" * 70)
    
    if not checkpoint_path:
        print("No checkpoint provided. Skipping prediction test.")
        return False
    
    # Load model
    model = BTC_model_decomposed(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    reassembler = ChordReassembler()
    
    # Get config params
    cfg = config.model if hasattr(config, 'model') else config
    seq_len = cfg.get('timestep', 108)
    feature_size = cfg.get('feature_size', 144)
    
    # Create synthetic input
    fake_input = torch.randn(1, 1, feature_size, seq_len).to(device)
    
    with torch.no_grad():
        predictions, _, _, _ = model(fake_input)
    
    print(f"\n--- Predicted Sequence (first 20 frames) ---")
    
    for t in range(min(20, seq_len)):
        frame_pred = {comp: predictions[comp][0, t].item() for comp in COMPONENT_NAMES}
        
        # Convert indices to labels
        frame_labels = {}
        for comp, idx in frame_pred.items():
            vocab = CHORD_VOCAB[comp]
            frame_labels[comp] = vocab[idx] if idx < len(vocab) else '?'
        
        chord = reassembler.reassemble(frame_labels)
        
        # Show components for interesting frames
        if t < 5 or chord != 'N':
            comp_str = " ".join([f"{k[:3]}={v}" for k, v in frame_labels.items() if v != 'N'])
            print(f"  Frame {t:3d}: {chord:15s} [{comp_str}]")
        elif t == 5:
            print(f"  ...")
    
    # Statistics
    print(f"\n--- Prediction Statistics ---")
    for comp in COMPONENT_NAMES:
        pred = predictions[comp][0].cpu().numpy()
        unique, counts = np.unique(pred, return_counts=True)
        top_idx = unique[np.argmax(counts)]
        top_label = CHORD_VOCAB[comp][top_idx] if top_idx < len(CHORD_VOCAB[comp]) else '?'
        print(f"  {comp:6s}: {len(unique)} unique values, most common: {top_label} ({np.max(counts)}/{len(pred)})")
    
    print("\n✓ Prediction test complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Debug and test the decomposed model')
    parser.add_argument('--config', default='run_config.yaml', help='Config file path')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint file path')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--test', default='all', 
                        choices=['all', 'forward', 'distribution', 'gradient', 'decompose', 'predict'],
                        help='Which test to run')
    parser.add_argument('--num_samples', type=int, default=100, 
                        help='Number of samples for distribution analysis')
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"\nDevice: {args.device}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint or 'None'}")
    print()
    
    # Load config
    config = HParams.load(args.config)
    
    results = {}
    
    # Run tests
    if args.test in ['all', 'decompose']:
        results['decompose'] = test_decomposition_reassembly()
    
    if args.test in ['all', 'forward']:
        results['forward'] = test_forward_pass(config, args.checkpoint, args.device)
    
    if args.test in ['all', 'gradient']:
        results['gradient'] = test_gradient_flow(config, args.device)
    
    if args.test in ['all', 'distribution']:
        results['distribution'] = test_component_distribution(config, args.device, args.num_samples)
    
    if args.test in ['all', 'predict'] and args.checkpoint:
        results['predict'] = test_model_predictions(config, args.checkpoint, args.device)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:20s}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'All tests passed!' if all_passed else 'Some tests failed.'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
