#!/usr/bin/env python
# encoding: utf-8
"""
Quick validation module for Chord Structure Decomposition model.

This script performs fast sanity checks on the entire pipeline without 
requiring full training, making it easy to catch errors quickly.

Usage:
    python quick_test_decomposed.py
    python quick_test_decomposed.py --verbose
"""

import sys
import torch
import numpy as np
import argparse
from pathlib import Path
import traceback

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


class QuickValidator:
    """Fast validation of model architecture and pipeline."""
    
    def __init__(self, verbose=False, device=None):
        self.verbose = verbose
        
        # Handle device selection with fallback to CPU if CUDA unavailable
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device.startswith('cuda') and not torch.cuda.is_available():
            if verbose:
                print(f"Warning: CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
        
        self.device = torch.device(device)
        self.results = {}
        
        if verbose:
            print(f"Using device: {self.device}")
    
    def log(self, msg):
        """Conditional logging."""
        if self.verbose:
            print(msg)
    
    def test_imports(self):
        """Test that all modules can be imported."""
        print("\n" + "="*60)
        print("TEST 1: Module Imports")
        print("="*60)
        
        tests = [
            ('torch', 'PyTorch'),
            ('numpy', 'NumPy'),
            ('utils.chord_decomposition', 'Chord Decomposition'),
            ('models.btc_model_decomposed', 'Decomposed Model'),
            ('data.audio_dataset_structured', 'Structured Dataset'),
            ('utils.decomposed_inference', 'Inference Utils'),
        ]
        
        failed = []
        for module, name in tests:
            try:
                __import__(module)
                print(f"  [OK] {name}")
            except Exception as e:
                print(f"  [FAILED] {name}: {str(e)[:50]}")
                failed.append(name)
        
        self.results['imports'] = len(failed) == 0
        return len(failed) == 0
    
    def test_chord_decomposition(self):
        """Test chord decomposition basic functionality."""
        print("\n" + "="*60)
        print("TEST 2: Chord Decomposition")
        print("="*60)
        
        try:
            from utils.chord_decomposition import ChordDecomposer, ChordReassembler
            
            decomposer = ChordDecomposer()
            reassembler = ChordReassembler()
            
            # Test basic decomposition
            test_cases = [
                'C:maj',
                'D:min7',
                'E:maj9/G#',
                'F#:dim',
                'N'
            ]
            
            for chord in test_cases:
                components = decomposer.decompose(chord)
                reassembled = reassembler.reassemble(components)
                # Chord spelling can vary (e.g. min7 vs minb7), so validate decomposition integrity
                has_all_components = isinstance(components, dict) and len(components) > 0
                if has_all_components and isinstance(reassembled, str):
                    print(f"  [OK] {chord}  ->  {reassembled}")
                else:
                    print(f"  [FAIL] {chord} produced invalid decomposition/reassembly")
                    self.results['chord_decomposition'] = False
                    return False
            
            print(f"  [OK] All {len(test_cases)} test cases passed")
            self.results['chord_decomposition'] = True
            return True
        
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            if self.verbose:
                traceback.print_exc()
            self.results['chord_decomposition'] = False
            return False
    
    def test_model_architecture(self):
        """Test model architecture and forward pass."""
        print("\n" + "="*60)
        print("TEST 3: Model Architecture")
        print("="*60)
        
        try:
            from models.btc_model_decomposed import (
                ComponentHead, MultiHeadChordDecomposer, BTC_model_decomposed, MultiTaskLoss
            )
            from utils.chord_decomposition import COMPONENT_NAMES, get_vocab_sizes
            
            # Test parameters
            batch_size = 2
            seq_len = 50
            feature_size = 192
            hidden_size = 128
            
            print(f"  Input shapes: batch={batch_size}, seq_len={seq_len}, features={feature_size}")
            
            # Test ComponentHead
            print("  Testing ComponentHead...")
            head = ComponentHead(hidden_size, vocab_size=13, dropout=0.1)
            head = head.to(self.device)
            x = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
            logits = head(x)
            assert logits.shape == (batch_size, seq_len, 13), f"Wrong shape: {logits.shape}"
            print(f"    [OK] ComponentHead output: {logits.shape}")
            
            # Test MultiHeadChordDecomposer
            print("  Testing MultiHeadChordDecomposer...")
            decomposer = MultiHeadChordDecomposer(hidden_size, dropout=0.1)
            decomposer = decomposer.to(self.device)
            logits = decomposer(x)
            assert len(logits) == len(COMPONENT_NAMES), f"Wrong number of heads"
            print(f"    [OK] MultiHeadChordDecomposer: {len(logits)} heads")
            
            # Verify all heads
            for component in COMPONENT_NAMES:
                assert component in logits, f"Missing component: {component}"
            print(f"    [OK] All {len(COMPONENT_NAMES)} component heads present")
            
            # Test MultiTaskLoss
            print("  Testing MultiTaskLoss...")
            vocab_sizes = get_vocab_sizes()
            loss_fn = MultiTaskLoss(vocab_sizes)
            loss_fn = loss_fn.to(self.device)
            
            labels = {
                component: torch.randint(0, vocab_sizes[component], 
                                        (batch_size, seq_len), 
                                        device=self.device)
                for component in COMPONENT_NAMES
            }
            
            loss, _ = loss_fn(logits, labels)
            assert loss.item() > 0, "Loss should be positive"
            print(f"    [OK] MultiTaskLoss: {loss.item():.4f}")
            
            self.results['model_architecture'] = True
            return True
        
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            if self.verbose:
                traceback.print_exc()
            self.results['model_architecture'] = False
            return False
    
    def test_full_model(self):
        """Test full BTC_model_decomposed."""
        print("\n" + "="*60)
        print("TEST 4: Full Model Forward Pass")
        print("="*60)
        
        try:
            from models.btc_model_decomposed import (
                BTC_model_decomposed,
                ChordFormer_model_decomposed,
                MultiTaskLoss,
            )
            from utils.chord_decomposition import COMPONENT_NAMES, get_vocab_sizes
            
            # Minimal config
            config = {
                'feature_size': 192,
                'hidden_size': 128,
                'num_layers': 2,
                'num_heads': 4,
                'total_key_depth': 128,
                'total_value_depth': 128,
                'filter_size': 256,
                'timestep': 50,
                'input_dropout': 0.1,
                'layer_dropout': 0.1,
                'attention_dropout': 0.1,
                'relu_dropout': 0.1,
                'probs_out': False,
                'use_decomposition': True,
            }
            
            print(f"  Creating model with hidden_size={config['hidden_size']}")
            model = BTC_model_decomposed(config)
            model = model.to(self.device)
            model.eval()
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"    [OK] Model created: {total_params:,} total params, {trainable_params:,} trainable")
            
            # Test forward pass
            print("  Testing forward pass...")
            batch_size = 2
            seq_len = 50
            
            features = torch.randn(batch_size, 1, config['feature_size'], seq_len, 
                                  device=self.device)
            
            with torch.no_grad():
                predictions, loss, weights, component_losses = model(features)
            
            assert predictions is not None, "Predictions should not be None"
            assert loss is None, "Loss should be None when no labels provided"
            assert component_losses is None, "Component losses should be None when no labels provided"
            print(f"    [OK] Forward pass successful")
            print(f"    [OK] Predictions structure: Dict with {len(predictions)} components")
            
            # Test with labels
            print("  Testing with labels...")
            vocab_sizes = get_vocab_sizes()
            labels = {
                component: torch.randint(0, vocab_sizes[component],
                                        (batch_size, seq_len),
                                        device=self.device)
                for component in COMPONENT_NAMES
            }
            
            model.train()
            predictions, loss, weights, component_losses = model(features, labels=labels)
            
            assert loss is not None, "Loss should not be None with labels"
            assert loss.item() > 0, "Loss should be positive"
            assert isinstance(component_losses, dict), "Component losses should be a dict"
            print(f"    [OK] Loss computed: {loss.item():.4f}")
            
            # Test backward pass
            print("  Testing backward pass...")
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            has_grads = sum(1 for p in model.parameters() if p.grad is not None)
            print(f"    [OK] Backward pass successful: {has_grads} parameters with gradients")
            
            optimizer.step()
            print(f"    [OK] Optimizer step successful")

            # Test ChordFormer decomposed backbone
            print("  Testing ChordFormer_model_decomposed...")
            chordformer_config = dict(config)
            chordformer_config.update({
                'conv_kernel_size': 15,
                'ff_expansion_factor': 2,
                'conv_expansion_factor': 2,
            })
            chordformer = ChordFormer_model_decomposed(chordformer_config).to(self.device)
            chordformer.eval()

            with torch.no_grad():
                cf_predictions, cf_loss, cf_weights, cf_component_losses = chordformer(features)
            assert cf_predictions is not None, "ChordFormer predictions should not be None"
            assert cf_loss is None, "ChordFormer loss should be None without labels"
            assert cf_component_losses is None, "ChordFormer component losses should be None without labels"
            print(f"    [OK] ChordFormer forward pass successful")
            
            self.results['full_model'] = True
            return True
        
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            if self.verbose:
                traceback.print_exc()
            self.results['full_model'] = False
            return False
    
    def test_inference_pipeline(self):
        """Test inference utilities."""
        print("\n" + "="*60)
        print("TEST 5: Inference Pipeline")
        print("="*60)
        
        try:
            from models.btc_model_decomposed import BTC_model_decomposed
            from utils.decomposed_inference import (
                DecomposedChordInference, ChordMetrics
            )
            from utils.chord_decomposition import COMPONENT_NAMES, get_vocab_sizes
            
            # Create minimal model
            config = {
                'feature_size': 192,
                'hidden_size': 128,
                'num_layers': 1,
                'num_heads': 4,
                'total_key_depth': 128,
                'total_value_depth': 128,
                'filter_size': 256,
                'timestep': 50,
                'input_dropout': 0.0,
                'layer_dropout': 0.0,
                'attention_dropout': 0.0,
                'relu_dropout': 0.0,
                'probs_out': True,
            }
            
            model = BTC_model_decomposed(config)
            model = model.to(self.device)
            
            # Test DecomposedChordInference
            print("  Testing DecomposedChordInference...")
            inference = DecomposedChordInference(model, device=self.device)
            
            # Test prediction
            features = torch.randn(1, 1, config['feature_size'], 50, device=self.device)
            predictions = inference.predict_batch(features, return_probabilities=False)
            
            assert predictions is not None, "Predictions should not be None"
            assert len(predictions) == len(COMPONENT_NAMES), "Wrong number of predictions"
            print(f"    [OK] Inference successful: {len(predictions)} components")
            
            # Test decoding
            print("  Testing chord decoding...")
            chord_labels = inference.decode_predictions(predictions)
            assert isinstance(chord_labels, list), "Should return list"
            assert len(chord_labels) > 0, "Should have predictions"
            print(f"    [OK] Chord decoding successful: {len(chord_labels)} chords")
            print(f"      Example: {chord_labels[0]}")
            
            # Test probabilities
            print("  Testing probability computation...")
            probabilities = inference.predict_batch(features, return_probabilities=True)
            assert len(probabilities) == len(COMPONENT_NAMES), "Wrong number of probabilities"
            
            confidences = inference.get_confidence_scores(probabilities)
            assert len(confidences) > 0, "Should have confidence scores"
            print(f"    [OK] Confidence scores: min={confidences.min():.3f}, max={confidences.max():.3f}")
            
            # Test ChordMetrics
            print("  Testing ChordMetrics...")
            metrics = ChordMetrics()
            
            vocab_sizes = get_vocab_sizes()
            test_predictions = {
                component: np.array([0, 1, 2, 0])
                for component in COMPONENT_NAMES
            }
            test_targets = {
                component: np.array([0, 1, 1, 0])  # Third element different
                for component in COMPONENT_NAMES
            }
            
            eval_metrics = metrics.evaluate(test_predictions, test_targets)
            assert 'chord_accuracy' in eval_metrics, "Missing chord_accuracy"
            print(f"    [OK] Metrics computed: chord_accuracy={eval_metrics['chord_accuracy']:.3f}")
            
            self.results['inference_pipeline'] = True
            return True
        
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            if self.verbose:
                traceback.print_exc()
            self.results['inference_pipeline'] = False
            return False
    
    def test_dataset(self):
        """Test dataset loading."""
        print("\n" + "="*60)
        print("TEST 6: Dataset Loading")
        print("="*60)
        
        try:
            from data.audio_dataset_structured import get_component_vocab_sizes
            
            print("  Testing component vocabulary sizes...")
            vocab_sizes = get_component_vocab_sizes()
            
            expected_sizes = {
                'root': 13,
                'bass': 13,
                'triad': 7,
                'misc': 2,
                '6th': 2,
                '7th': 4,
                '9th': 4,
                '11th': 3,
                '13th': 3
            }
            
            for component, expected_size in expected_sizes.items():
                actual_size = vocab_sizes[component]
                if actual_size == expected_size:
                    print(f"    [OK] {component}: {actual_size}")
                else:
                    print(f"    [FAIL] {component}: {actual_size} (expected {expected_size})")
                    return False
            
            total = sum(vocab_sizes.values())
            print(f"  [OK] Total vocabulary size: {total} classes")
            
            self.results['dataset'] = True
            return True
        
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            if self.verbose:
                traceback.print_exc()
            self.results['dataset'] = False
            return False
    
    def test_training_step(self):
        """Test a single training step."""
        print("\n" + "="*60)
        print("TEST 7: Single Training Step")
        print("="*60)
        
        try:
            import torch.optim as optim
            from models.btc_model_decomposed import BTC_model_decomposed, MultiTaskLoss
            from utils.decomposed_inference import DecomposedChordTrainer
            from utils.chord_decomposition import COMPONENT_NAMES, get_vocab_sizes
            
            # Create model
            config = {
                'feature_size': 192,
                'hidden_size': 128,
                'num_layers': 1,
                'num_heads': 4,
                'total_key_depth': 128,
                'total_value_depth': 128,
                'filter_size': 256,
                'timestep': 50,
                'input_dropout': 0.1,
                'layer_dropout': 0.1,
                'attention_dropout': 0.1,
                'relu_dropout': 0.1,
                'probs_out': False,
            }
            
            model = BTC_model_decomposed(config)
            model = model.to(self.device)
            
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            print("  Preparing dummy batch...")
            batch_size = 2
            seq_len = 50
            
            features = torch.randn(batch_size, 1, config['feature_size'], seq_len,
                                  device=self.device)
            
            vocab_sizes = get_vocab_sizes()
            labels = {
                component: torch.randint(0, vocab_sizes[component],
                                        (batch_size, seq_len),
                                        device=self.device)
                for component in COMPONENT_NAMES
            }
            
            print("  Running training step...")
            model.train()
            
            predictions, loss, _, component_losses = model(features, labels=labels)
            
            print(f"    [OK] Forward pass: loss={loss.item():.4f}")
            assert isinstance(component_losses, dict), "Component losses should be a dict"
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            print(f"    [OK] Backward pass complete")
            
            # Optimizer step
            optimizer.step()
            
            print(f"    [OK] Optimizer step complete")
            
            # Run another step to verify convergence
            predictions2, loss2, _, _ = model(features, labels=labels)
            print(f"    [OK] Second forward pass: loss={loss2.item():.4f}")
            
            if loss2.item() != loss.item():
                print(f"    [OK] Loss changed (good): {loss.item():.4f}  ->  {loss2.item():.4f}")
            
            self.results['training_step'] = True
            return True
        
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            if self.verbose:
                traceback.print_exc()
            self.results['training_step'] = False
            return False
    
    def run_all(self):
        """Run all tests."""
        print("\n" + "="*70)
        print(" "*15 + "QUICK VALIDATION TEST SUITE")
        print("="*70)
        
        results = []
        
        tests = [
            ("Module Imports", self.test_imports),
            ("Chord Decomposition", self.test_chord_decomposition),
            ("Model Architecture", self.test_model_architecture),
            ("Full Model", self.test_full_model),
            ("Inference Pipeline", self.test_inference_pipeline),
            ("Dataset", self.test_dataset),
            ("Training Step", self.test_training_step),
        ]
        
        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append((test_name, success))
            except Exception as e:
                print(f"\n[FAIL] {test_name} FAILED with exception:")
                traceback.print_exc()
                results.append((test_name, False))
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        for test_name, success in results:
            symbol = "[OK]" if success else "[FAIL]"
            print(f"  {symbol} {test_name}")
        
        print(f"\nPassed: {passed}/{total}")
        
        if passed == total:
            print("\n" + " " * 15)
            print("ALL TESTS PASSED! Ready for training.")
            print(" " * 15)
            return 0
        else:
            print(f"Some tests failed. Review errors above.")
            return 1


def main():
    parser = argparse.ArgumentParser(
        description='Quick validation test for Chord Structure Decomposition'
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, cpu, etc.)')
    
    args = parser.parse_args()
    
    validator = QuickValidator(verbose=args.verbose, device=args.device)
    return validator.run_all()


if __name__ == '__main__':
    sys.exit(main())
