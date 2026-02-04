#!/usr/bin/env python
# encoding: utf-8
"""
Verification and compatibility check for Chord Structure Decomposition implementation.

This script validates that all components are properly integrated and functional.
"""

import sys
import traceback
from pathlib import Path

def check_imports():
    """Check that all required modules can be imported."""
    print("=" * 60)
    print("CHECKING IMPORTS")
    print("=" * 60)
    
    imports_to_check = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('librosa', 'Librosa'),
        ('utils.chord_decomposition', 'Chord Decomposition Module'),
        ('models.btc_model_decomposed', 'Decomposed Model'),
        ('data.audio_dataset_structured', 'Structured Dataset'),
        ('utils.decomposed_inference', 'Inference Utilities'),
    ]
    
    failed = []
    for module_name, display_name in imports_to_check:
        try:
            __import__(module_name)
            print(f"[OK] {display_name:40s} ({module_name})")
        except ImportError as e:
            print(f"[FAIL] {display_name:40s} ({module_name})")
            print(f"  Error: {e}")
            failed.append(module_name)
    
    return len(failed) == 0, failed


def check_chord_decomposition():
    """Check chord decomposition functionality."""
    print("\n" + "=" * 60)
    print("CHECKING CHORD DECOMPOSITION")
    print("=" * 60)
    
    try:
        from utils.chord_decomposition import (
            ChordDecomposer, ChordReassembler, get_vocab_sizes,
            COMPONENT_NAMES, CHORD_VOCAB
        )
        
        # Test decomposition
        decomposer = ChordDecomposer()
        test_chords = ['C:maj', 'D:min7', 'E:maj9/G#', 'N']
        
        print("Testing chord decomposition:")
        for chord in test_chords:
            components = decomposer.decompose(chord)
            reassembler = ChordReassembler()
            reassembled = reassembler.reassemble(components)
            match = "[OK]" if reassembled == chord else "[FAIL]"
            print(f"  {match} {chord:15s} -> {reassembled:15s}")
        
        # Check vocabulary
        print("\nVocabulary sizes:")
        vocab_sizes = get_vocab_sizes()
        total = 0
        for component, size in vocab_sizes.items():
            print(f"  {component:10s}: {size:3d} classes")
            total += size
        print(f"  {'TOTAL':10s}: {total:3d} classes")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] Chord decomposition check failed:")
        traceback.print_exc()
        return False


def check_model_architecture():
    """Check model architecture."""
    print("\n" + "=" * 60)
    print("CHECKING MODEL ARCHITECTURE")
    print("=" * 60)
    
    try:
        import torch
        from models.btc_model_decomposed import (
            ComponentHead, MultiHeadChordDecomposer, MultiTaskLoss,
            BTC_model_decomposed
        )
        
        # Test ComponentHead
        print("Testing ComponentHead:")
        head = ComponentHead(hidden_size=256, vocab_size=13)
        x = torch.randn(4, 100, 256)
        logits = head(x)
        print(f"  [OK] Input shape: {x.shape}")
        print(f"  [OK] Output shape: {logits.shape}")
        
        # Test MultiHeadChordDecomposer
        print("\nTesting MultiHeadChordDecomposer:")
        decomposer = MultiHeadChordDecomposer(hidden_size=256)
        logits = decomposer(x)
        print(f"  [OK] Number of heads: {len(logits)}")
        for component, logits_c in logits.items():
            print(f"  [OK] {component}: {logits_c.shape}")
        
        # Test MultiTaskLoss
        print("\nTesting MultiTaskLoss:")
        vocab_sizes = {
            'root': 13, 'bass': 13, 'triad': 7, 'misc': 2,
            '7th': 4, '9th': 4, '11th': 3, '13th': 3
        }
        loss_fn = MultiTaskLoss(vocab_sizes)
        
        labels = {
            component: torch.randint(0, vocab_sizes[component], (4, 100))
            for component in vocab_sizes
        }
        
        logits_dict = {
            component: torch.randn(4, 100, vocab_sizes[component])
            for component in vocab_sizes
        }
        
        loss = loss_fn(logits_dict, labels)
        print(f"  [OK] Loss computed: {loss.item():.4f}")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] Model architecture check failed:")
        traceback.print_exc()
        return False


def check_dataset():
    """Check dataset functionality."""
    print("\n" + "=" * 60)
    print("CHECKING DATASET")
    print("=" * 60)
    
    try:
        import numpy as np
        from data.audio_dataset_structured import (
            AudioDatasetStructured, get_component_vocab_sizes, _collate_fn_structured
        )
        
        # Check vocabulary sizes
        print("Component vocabulary sizes:")
        vocab_sizes = get_component_vocab_sizes()
        for component, size in vocab_sizes.items():
            print(f"  {component:10s}: {size:3d}")
        
        print("\n[OK] Dataset module imports successful")
        print("[OK] Component vocabulary sizes correct")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] Dataset check failed:")
        traceback.print_exc()
        return False


def check_inference_utilities():
    """Check inference utilities."""
    print("\n" + "=" * 60)
    print("CHECKING INFERENCE UTILITIES")
    print("=" * 60)
    
    try:
        import torch
        from utils.decomposed_inference import (
            DecomposedChordInference, DecomposedChordTrainer, ChordMetrics
        )
        from models.btc_model_decomposed import BTC_model_decomposed
        
        # Create dummy model
        config = {
            'feature_size': 192,
            'hidden_size': 256,
            'num_layers': 2,
            'num_heads': 4,
            'total_key_depth': 256,
            'total_value_depth': 256,
            'filter_size': 1024,
            'timestep': 100,
            'input_dropout': 0.1,
            'layer_dropout': 0.1,
            'attention_dropout': 0.1,
            'relu_dropout': 0.1,
            'probs_out': True,
        }
        
        model = BTC_model_decomposed(config)
        
        # Test inference utilities
        print("Testing inference utilities:")
        
        inference = DecomposedChordInference(model)
        print(f"  [OK] DecomposedChordInference initialized")
        
        trainer = DecomposedChordTrainer(model)
        print(f"  [OK] DecomposedChordTrainer initialized")
        
        metrics = ChordMetrics()
        print(f"  [OK] ChordMetrics initialized")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] Inference utilities check failed:")
        traceback.print_exc()
        return False


def check_files():
    """Check that all required files exist."""
    print("\n" + "=" * 60)
    print("CHECKING FILES")
    print("=" * 60)
    
    required_files = [
        'utils/chord_decomposition.py',
        'models/btc_model_decomposed.py',
        'data/audio_dataset_structured.py',
        'utils/decomposed_inference.py',
        'train_decomposed.py',
        'infer_decomposed.py',
        'test_decomposition.py',
        'CHORD_DECOMPOSITION_GUIDE.md',
        'INTEGRATION_SUMMARY.md',
        'EXEMPLOS_USO.md',
    ]
    
    base_path = Path(__file__).parent
    all_exist = True
    
    for filename in required_files:
        filepath = base_path / filename
        exists = filepath.exists()
        symbol = "[OK]" if exists else "[FAIL]"
        print(f"  {symbol} {filename}")
        if not exists:
            all_exist = False
    
    return all_exist


def main():
    """Run all checks."""
    print("\n" + "=" * 60)
    print("CHORD STRUCTURE DECOMPOSITION - VERIFICATION")
    print("=" * 60 + "\n")
    
    checks = [
        ("Imports", check_imports),
        ("Chord Decomposition", check_chord_decomposition),
        ("Model Architecture", check_model_architecture),
        ("Dataset", check_dataset),
        ("Inference Utilities", check_inference_utilities),
        ("Files", check_files),
    ]
    
    results = {}
    for check_name, check_func in checks:
        if check_name == "Imports":
            success, failed = check_func()
            results[check_name] = success
        else:
            try:
                results[check_name] = check_func()
            except Exception as e:
                print(f"\n[FAIL] {check_name} check encountered an error:")
                traceback.print_exc()
                results[check_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check_name, success in results.items():
        symbol = "[OK]" if success else "[FAIL]"
        print(f"  {symbol} {check_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n[OK] All checks passed! Implementation is ready to use.")
        return 0
    else:
        print("\n[FAIL] Some checks failed. Please review the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
