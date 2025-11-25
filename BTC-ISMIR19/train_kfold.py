#!/usr/bin/env python
"""
K-Fold Training Script for BTC Chord Recognition Model.

This script runs training for all k-folds (0-4) sequentially, with each fold:
- Getting its own wandb run
- Saving checkpoints in organized folder structure: assets/experiment_{params}/kfold_{num}/
- Preventing checkpoint overwrites between folds
"""

import os
import sys
import argparse
import subprocess

def create_experiment_folder_name(args):
    """Create a folder name based on experiment parameters."""
    parts = [
        f"exp{args.index}",
        args.model,
        args.dataset1,
        args.dataset2,
        args.test_dataset,
        "voca" if args.voca else "majmin",
        "curriculum" if args.curriculum else "standard"
    ]
    return "_".join(parts)

def main():
    parser = argparse.ArgumentParser(description='Run k-fold training for all folds')
    parser.add_argument('--index', type=int, help='Experiment Number', required=True)
    parser.add_argument('--kfold_start', type=int, help='Starting k-fold (default: 0)', default=0)
    parser.add_argument('--kfold_end', type=int, help='Ending k-fold (default: 4)', default=4)
    parser.add_argument('--voca', type=bool, help='large voca is True', default=False)
    parser.add_argument('--model', type=str, help='btc, cnn, crnn', default='btc')
    parser.add_argument('--dataset1', type=str, help='Dataset', default='billboard')
    parser.add_argument('--dataset2', type=str, help='Dataset', default='jaah')
    parser.add_argument('--test_dataset', type=str, help='Test Dataset', default='rwc')
    parser.add_argument('--restore_epoch', type=int, default=1000)
    parser.add_argument('--early_stop', type=bool, help='no improvement during 10 epoch -> stop', default=True)
    parser.add_argument('--curriculum', type=bool, help='Enable curriculum learning', default=False)
    
    args = parser.parse_args()
    
    # Create experiment folder name
    experiment_folder = create_experiment_folder_name(args)
    print("="*70)
    print(f"K-FOLD TRAINING: {experiment_folder}")
    print("="*70)
    print(f"Will train k-folds: {args.kfold_start} to {args.kfold_end}")
    print(f"Experiment folder: {experiment_folder}")
    print("="*70)
    
    # Run training for each k-fold
    for kfold in range(args.kfold_start, args.kfold_end + 1):
        print("\n" + "="*70)
        print(f"Starting K-Fold {kfold}/{args.kfold_end}")
        print("="*70)
        
        # Build command
        script_path = os.path.join(os.path.dirname(__file__), 'train_curriculum.py')
        cmd = [
            sys.executable,
            script_path,
            '--index', str(args.index),
            '--kfold', str(kfold),
            '--model', args.model,
            '--dataset1', args.dataset1,
            '--dataset2', args.dataset2,
            '--test_dataset', args.test_dataset,
            '--restore_epoch', str(args.restore_epoch),
            '--early_stop', str(args.early_stop),
            '--curriculum', str(args.curriculum),
        ]
        
        if args.voca:
            cmd.extend(['--voca', 'True'])
        
        # Add experiment folder as environment variable so train_curriculum.py can use it
        env = os.environ.copy()
        env['EXPERIMENT_FOLDER'] = experiment_folder
        env['KFOLD_NUM'] = str(kfold)
        
        # Change to script directory to ensure relative imports work
        script_dir = os.path.dirname(os.path.abspath(script_path))
        
        # Run training
        try:
            result = subprocess.run(cmd, env=env, check=True, cwd=script_dir)
            print(f"\n✓ K-Fold {kfold} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ K-Fold {kfold} failed with exit code {e.returncode}")
            print("Continuing with next k-fold...")
            continue
    
    print("\n" + "="*70)
    print("ALL K-FOLDS COMPLETED")
    print("="*70)

if __name__ == '__main__':
    main()

