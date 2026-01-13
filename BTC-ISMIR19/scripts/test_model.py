#!/usr/bin/env python
"""
Standalone script to test a trained model on a dataset.
"""
import os
import sys
import argparse
import torch
import numpy as np

# Add the parent directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.hparams import HParams
from utils.mir_eval_modules import large_voca_score_calculation, root_majmin_score_calculation
from data.audio_dataset import AudioDataset
from models.btc_model import BTC_model, BTC_model_structured


def main():
    parser = argparse.ArgumentParser(description='Test a trained chord recognition model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth.tar file)')
    parser.add_argument('--config', type=str, default='run_config.yaml',
                        help='Path to config YAML file (default: run_config.yaml)')
    parser.add_argument('--test_dataset', type=str, default='rwc',
                        help='Dataset to test on (default: rwc)')
    parser.add_argument('--model', type=str, default='btc',
                        help='Model type: btc, btc_structured, cnn, crnn (default: btc)')
    parser.add_argument('--voca', action='store_true', default=True,
                        help='Use large vocabulary (170 chords)')
    parser.add_argument('--kfold', type=int, default=0,
                        help='K-fold index for test split (default: 0)')
    parser.add_argument('--normalization', type=str, default=None,
                        help='Path to normalization file (optional, will search in result folder if not provided)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-song scores')
    
    args = parser.parse_args()
    
    # Load config
    config = HParams.load(args.config)
    
    # Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    if args.model == 'btc':
        model = BTC_model(config=config.model).to(device)
    elif args.model == 'btc_structured':
        model = BTC_model_structured(config=config.model).to(device)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Use strict=False to handle checkpoints with extra keys (e.g., class_weights from reweighted loss)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load test dataset
    root_path = config.path['root_path']
    print(f"\nLoading test dataset: {args.test_dataset}")
    test_dataset = AudioDataset(config, root_dir=root_path, 
                                dataset_names=(args.test_dataset,),
                                num_workers=1, train=False, 
                                kfold=args.kfold)
    print(f"Test dataset: {len(test_dataset.song_names)} songs")
    
    # Load normalization parameters
    if args.normalization:
        z_path = args.normalization
    else:
        # Search for normalization file
        mp3_config = config.mp3
        feature_config = config.feature
        mp3_string = "%d_%.1f_%.1f" % (mp3_config['song_hz'], mp3_config['inst_len'], mp3_config['skip_interval'])
        feature_string = "cqt_%d_%d_%d" % (feature_config['n_bins'], feature_config['bins_per_octave'], feature_config['hop_length'])
        
        # Try different normalization file locations
        possible_paths = [
            os.path.join(root_path, 'result', f'{mp3_string}_{feature_string}_mix_kfold_{args.kfold}_normalization.pt'),
            os.path.join(root_path, 'result_not_structured', f'{mp3_string}_{feature_string}_mix_kfold_{args.kfold}_normalization.pt'),
        ]
        
        z_path = None
        for p in possible_paths:
            if os.path.exists(p):
                z_path = p
                break
        
        if z_path is None:
            print("ERROR: Could not find normalization file. Tried:")
            for p in possible_paths:
                print(f"  - {p}")
            print("\nPlease specify --normalization path")
            return
    
    print(f"Loading normalization from: {z_path}")
    normalization = torch.load(z_path, weights_only=False)
    mean = normalization['mean']
    std = normalization['std']
    print(f"Normalization: mean={mean:.6f}, std={std:.6f}")
    
    # Run evaluation
    print(f"\n{'='*60}")
    print(f"Starting evaluation on {args.test_dataset} dataset")
    print(f"{'='*60}")
    
    if args.voca:
        score_metrics = ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']
        score_list_dict, song_length_list, average_score_dict, wcsr_dict = large_voca_score_calculation(
            valid_dataset=test_dataset, 
            config=config, 
            model=model, 
            model_type=args.model, 
            mean=mean, 
            std=std, 
            device=device,
            verbose=args.verbose
        )
    else:
        score_metrics = ['root', 'majmin']
        score_list_dict, song_length_list, average_score_dict, wcsr_dict = root_majmin_score_calculation(
            valid_dataset=test_dataset, 
            config=config, 
            model=model, 
            model_type=args.model, 
            mean=mean, 
            std=std, 
            device=device,
            verbose=args.verbose
        )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"TEST RESULTS on {args.test_dataset}")
    print(f"{'='*60}")
    
    if len(average_score_dict) == 0:
        print("ERROR: No songs were successfully evaluated!")
        return
    
    for m in score_metrics:
        if m in average_score_dict:
            print(f"  {m:10s}: {average_score_dict[m]:.4f}  (WCSR: {wcsr_dict.get(m, 0):.2f}%)")
    
    print(f"\nTotal songs evaluated: {len(song_length_list)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

