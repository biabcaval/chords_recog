#!/usr/bin/env python
"""
Script to test a trained model checkpoint on a specific dataset.
Usage: python test_checkpoint.py --checkpoint /path/to/checkpoint.pth.tar --test_dataset dj_avan --voca
"""

import argparse
import os
import sys
import torch
import numpy as np

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.hparams import HParams
from models.btc_model import BTC_model
from utils.mir_eval_modules import large_voca_score_calculation, root_majmin_score_calculation
from utils.preprocess import Preprocess, FeatureTypes
import glob

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
parser.add_argument('--test_dataset', type=str, default='dj_avan', help='Test dataset name')
parser.add_argument('--voca', action='store_true', help='Use large vocabulary')
parser.add_argument('--model', type=str, default='btc', help='Model type')
args = parser.parse_args()

# Load config
config = HParams.load("run_config.yaml")
if args.voca:
    config.feature['large_voca'] = True
    config.model['num_chords'] = 170

print(f"Testing checkpoint: {args.checkpoint}")
print(f"Test dataset: {args.test_dataset}")
print(f"Large vocabulary: {args.voca}")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load normalization data
mp3_config = config.mp3
feature_config = config.feature
mp3_string = "%d_%.1f_%.1f" % (mp3_config['song_hz'], mp3_config['inst_len'], mp3_config['skip_interval'])
feature_string = "cqt_%d_%d_%d" % (feature_config['n_bins'], feature_config['bins_per_octave'], feature_config['hop_length'])
normalization_file = os.path.join(config.path['root_path'], 'result', f'{mp3_string}_{feature_string}_mix_kfold_0_normalization.pt')

if os.path.exists(normalization_file):
    normalization = torch.load(normalization_file, weights_only=False)
    mean = normalization['mean']
    std = normalization['std']
    print(f"Loaded normalization from: {normalization_file}")
else:
    raise FileNotFoundError(f"Normalization file not found: {normalization_file}")

# Load test data
def load_all_test_data(config, root_dir, dataset_name):
    """Load all data from a dataset without k-fold splitting"""
    mp3_config = config.mp3
    feature_config = config.feature
    mp3_string = "%d_%.1f_%.1f" % (mp3_config['song_hz'], mp3_config['inst_len'], mp3_config['skip_interval'])
    feature_string = "%s_%d_%d_%d" % ('cqt', feature_config['n_bins'], feature_config['bins_per_octave'], feature_config['hop_length'])
    
    if feature_config['large_voca'] == True:
        dataset_path = os.path.join(root_dir, "result", dataset_name+'_voca', mp3_string, feature_string)
    else:
        dataset_path = os.path.join(root_dir, "result", dataset_name, mp3_string, feature_string)
    
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    song_names = os.listdir(dataset_path)
    all_paths = []
    used_song_names = []
    
    for song_name in song_names:
        song_path = os.path.join(dataset_path, song_name)
        if os.path.isdir(song_path):
            instance_names = os.listdir(song_path)
            if len(instance_names) > 0:
                used_song_names.append(song_name)
                for instance_name in instance_names:
                    if "1.00_0" in instance_name:
                        all_paths.append(os.path.join(song_path, instance_name))
    
    print(f"Loaded {len(used_song_names)} songs with {len(all_paths)} instances from {dataset_name}")
    return used_song_names, all_paths

# Custom dataset class for test data
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, paths, song_names, config, root_dir, dataset_name):
        self.paths = paths
        self.song_names = song_names

        feature_type_str = config.feature['type']
        if feature_type_str == 'cqt':
            feature_to_use = FeatureTypes.cqt
        elif feature_type_str == 'hcqt':
            feature_to_use = FeatureTypes.hcqt
        else:
            raise ValueError(f"Unsupported feature type: {feature_type_str}")

        self.preprocessor = Preprocess(config, feature_to_use, (dataset_name,), root_dir)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        data = np.load(self.paths[idx])
        feature = data['feature']
        chord = data['chord']
        res = {'feature': feature, 'chord': chord}
        return res

# Load test dataset
test_song_names, test_paths = load_all_test_data(config, config.path['root_path'], args.test_dataset)
test_dataset = TestDataset(test_paths, test_song_names, config, config.path['root_path'], args.test_dataset)

# Initialize model
if args.model == 'btc':
    model = BTC_model(config=config.model).to(device)
else:
    raise NotImplementedError(f"Model type {args.model} not implemented")

# Load checkpoint
if os.path.isfile(args.checkpoint):
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Use strict=False to handle extra keys like class_weights
    model.load_state_dict(checkpoint['model'], strict=False)
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded checkpoint from epoch {epoch}")
else:
    raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

# Run evaluation
print(f"\n{'='*60}")
print(f"Starting evaluation on {args.test_dataset} dataset")
print(f"{'='*60}\n")

if args.voca:
    score_metrics = ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']
    score_list_dict, song_length_list, average_score_dict, wcsr_dict = large_voca_score_calculation(
        valid_dataset=test_dataset, config=config, model=model, 
        model_type=args.model, mean=mean, std=std, device=device
    )
else:
    score_metrics = ['root', 'majmin']
    score_list_dict, song_length_list, average_score_dict, wcsr_dict = root_majmin_score_calculation(
        valid_dataset=test_dataset, config=config, model=model, 
        model_type=args.model, mean=mean, std=std, device=device
    )

# Print results
print(f"\n{'='*60}")
print(f"RESULTS - Test on {args.test_dataset}")
print(f"{'='*60}")

if len(song_length_list) == 0:
    print("ERROR: No songs were processed!")
else:
    print(f"Processed {len(song_length_list)} songs\n")
    for m in score_metrics:
        score = average_score_dict.get(m, 0)
        wcsr = wcsr_dict.get(m, 0) if wcsr_dict else 0
        print(f"{m:12s}: Score = {score:.4f}, WCSR = {wcsr:.2f}%")

print(f"\n{'='*60}")
print("Testing completed!")
print(f"{'='*60}")

