"""
Test script for curriculum learning implementation.

This script verifies that curriculum learning is working correctly
without running full training.
"""

import torch
import numpy as np
from utils.hparams import HParams
from audio_dataset import AudioDataset
from curriculum_learning import CurriculumLearning
import matplotlib.pyplot as plt
import os

def test_curriculum_learning():
    """Test curriculum learning functionality."""
    
    print("="*60)
    print("Testing Curriculum Learning Implementation")
    print("="*60)
    
    # Load configuration
    config = HParams.load("run_config.yaml")
    
    # Force enable curriculum learning for testing
    config.curriculum['enabled'] = True
    
    print("\n1. Testing with different strategies...")
    strategies = ['chord_complexity', 'change_frequency', 'unique_chords', 'mixed']
    
    # Load a small dataset for testing
    print("\n2. Loading dataset...")
    try:
        dataset = AudioDataset(
            config, 
            root_dir=config.path['root_path'],
            dataset_names=('billboard',),
            num_workers=1,
            preprocessing=False,
            train=True,
            kfold=0
        )
        print(f"   Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"   Error loading dataset: {e}")
        print("   Make sure the dataset is preprocessed first!")
        return
    
    # Test each strategy
    for strategy in strategies:
        print(f"\n3. Testing strategy: {strategy}")
        config.curriculum['strategy'] = strategy
        
        try:
            # Initialize curriculum learning
            curriculum = CurriculumLearning(config, dataset)
            
            # Get statistics for various epochs
            print(f"   Difficulty scores computed successfully")
            print(f"   Score range: [{curriculum.difficulty_scores.min():.3f}, {curriculum.difficulty_scores.max():.3f}]")
            
            # Test progression through epochs
            print(f"\n   Epoch progression:")
            test_epochs = [0, 5, 10, 20, 30, 40, 50]
            for epoch in test_epochs:
                stats = curriculum.get_stats(epoch)
                print(f"     Epoch {epoch:3d}: {stats['ratio']:5.1%} of data, "
                      f"mean difficulty: {stats['mean_difficulty']:.3f}")
            
        except Exception as e:
            print(f"   ERROR testing {strategy}: {e}")
            import traceback
            traceback.print_exc()
    
    # Test pacing functions
    print("\n4. Testing pacing functions...")
    pacing_functions = ['linear', 'quadratic', 'exponential', 'step']
    
    config.curriculum['strategy'] = 'mixed'
    curriculum = CurriculumLearning(config, dataset)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, pacing in enumerate(pacing_functions):
        config.curriculum['pacing'] = pacing
        curriculum.pacing = pacing
        
        epochs = np.arange(0, 50)
        ratios = [curriculum.get_curriculum_ratio(e) for e in epochs]
        
        axes[idx].plot(epochs, ratios, linewidth=2)
        axes[idx].set_title(f'Pacing: {pacing}', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Epoch', fontsize=12)
        axes[idx].set_ylabel('Data Ratio', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([0, 1.05])
        
        # Add markers at key points
        warmup = config.curriculum['warmup_epochs']
        pace_end = warmup + config.curriculum['pace_epochs']
        axes[idx].axvline(x=warmup, color='r', linestyle='--', alpha=0.5, label='Warmup End')
        axes[idx].axvline(x=pace_end, color='g', linestyle='--', alpha=0.5, label='Full Data')
        axes[idx].legend()
    
    plt.tight_layout()
    
    # Save plot
    output_dir = 'assets/curriculum_learning'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'pacing_functions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Pacing functions plot saved to: {output_path}")
    plt.close()
    
    # Test difficulty distribution
    print("\n5. Testing difficulty distribution...")
    config.curriculum['strategy'] = 'mixed'
    curriculum = CurriculumLearning(config, dataset)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of difficulties
    axes[0].hist(curriculum.difficulty_scores, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title('Distribution of Sample Difficulties', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Difficulty Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_scores = np.sort(curriculum.difficulty_scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    axes[1].plot(sorted_scores, cumulative, linewidth=2)
    axes[1].set_title('Cumulative Distribution of Difficulties', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Difficulty Score', fontsize=12)
    axes[1].set_ylabel('Cumulative Proportion', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Mark percentiles
    for pct in [25, 50, 75]:
        idx = int(len(sorted_scores) * pct / 100)
        axes[1].axvline(x=sorted_scores[idx], color='r', linestyle='--', alpha=0.5)
        axes[1].text(sorted_scores[idx], 0.5, f'{pct}th', rotation=90, va='center')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'difficulty_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Difficulty distribution plot saved to: {output_path}")
    plt.close()
    
    # Test sample selection for different epochs
    print("\n6. Testing sample selection across epochs...")
    test_epochs = [0, 10, 20, 30, 40]
    
    fig, axes = plt.subplots(1, len(test_epochs), figsize=(18, 4))
    
    for idx, epoch in enumerate(test_epochs):
        indices = curriculum.get_sample_indices(epoch)
        selected_difficulties = curriculum.difficulty_scores[indices]
        
        axes[idx].hist(selected_difficulties, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[idx].set_title(f'Epoch {epoch}\n({len(indices)} samples)', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Difficulty', fontsize=10)
        axes[idx].set_ylabel('Count', fontsize=10)
        axes[idx].set_xlim([0, 1])
        
        # Add statistics
        mean_diff = selected_difficulties.mean()
        axes[idx].axvline(x=mean_diff, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_diff:.2f}')
        axes[idx].legend(fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'epoch_progression.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Epoch progression plot saved to: {output_path}")
    plt.close()
    
    print("\n" + "="*60)
    print("Testing completed successfully!")
    print("="*60)
    print(f"\nVisualization saved to: {output_dir}/")
    print("\nTo use curriculum learning in training:")
    print("  python train_curriculum.py --curriculum True --index 1 --kfold 0")
    print("\nTo modify curriculum settings, edit 'run_config.yaml'")


if __name__ == "__main__":
    test_curriculum_learning()

