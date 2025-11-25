#!/usr/bin/env python
"""Test a specific checkpoint on a test dataset"""
import os
import sys
import argparse
import torch
import numpy as np
from utils import logger
from utils.hparams import HParams
from utils.mir_eval_modules import large_voca_score_calculation, root_majmin_score_calculation
from btc_model import BTC_model
from audio_dataset import AudioDataset

logger.logging_verbosity(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='Test a specific checkpoint')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file (e.g., assets/model/idx_2_044.pth.tar)')
parser.add_argument('--config', type=str, default='run_config.yaml', help='Config file path')
parser.add_argument('--test_dataset', type=str, default='rwc', help='Test dataset name')
parser.add_argument('--kfold', type=int, default=0, help='K-fold index (for normalization file)')
parser.add_argument('--model', type=str, default='btc', help='Model type: btc, cnn, crnn')
args = parser.parse_args()

# Load config
config = HParams.load(args.config)
large_voca = config.feature.get('large_voca', False)

if large_voca:
    config.feature['large_voca'] = True
    config.model['num_chords'] = 170
    logger.info("Using large vocabulary (170 chords)")
else:
    config.feature['large_voca'] = False
    config.model['num_chords'] = 25
    logger.info("Using majmin vocabulary (25 chords)")

# Load normalization
mp3_config = config.mp3
feature_config = config.feature
mp3_string = "%d_%.1f_%.1f" % (mp3_config['song_hz'], mp3_config['inst_len'], mp3_config['skip_interval'])
feature_string = "_%s_%d_%d_%d_" % ('cqt', feature_config['n_bins'], feature_config['bins_per_octave'], feature_config['hop_length'])
z_path = os.path.join(config.path['root_path'], 'result', mp3_string + feature_string + 'mix_kfold_'+ str(args.kfold) +'_normalization.pt')

if os.path.exists(z_path):
    normalization = torch.load(z_path)
    mean = normalization['mean']
    std = normalization['std']
    logger.info("Loaded normalization from: %s" % z_path)
else:
    raise FileNotFoundError(f"Normalization file not found: {z_path}. You may need to preprocess the training data first.")

# Create model
if args.model == 'btc':
    model = BTC_model(config=config.model).to(device)
else:
    raise NotImplementedError(f"Model type {args.model} not implemented in this script")

# Load checkpoint
if os.path.isfile(args.checkpoint):
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])
    logger.info(f"Loaded checkpoint: {args.checkpoint}")
    if 'epoch' in checkpoint:
        logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
else:
    raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

# Load test dataset
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
    
    logger.info(f"Loaded {len(used_song_names)} songs with {len(all_paths)} instances from {dataset_name}")
    return used_song_names, all_paths

test_song_names, test_paths = load_all_test_data(config, config.path['root_path'], args.test_dataset)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, paths, song_names, config, root_dir, dataset_name):
        self.paths = paths
        self.song_names = song_names
        from utils.preprocess import Preprocess, FeatureTypes
        self.preprocessor = Preprocess(config, FeatureTypes.cqt, (dataset_name,), root_dir)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        instance_path = self.paths[idx]
        res = dict()
        data = torch.load(instance_path)
        res['feature'] = np.log(np.abs(data['feature']) + 1e-6)
        res['chord'] = data['chord']
        return res

test_dataset = TestDataset(test_paths, test_song_names, config, config.path['root_path'], args.test_dataset)

# Run testing
logger.info("==== Starting testing on %s dataset ====" % args.test_dataset)

if large_voca:
    score_metrics = ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']
    score_list_dict, song_length_list, average_score_dict = large_voca_score_calculation(
        valid_dataset=test_dataset, config=config, model=model, model_type=args.model, 
        mean=mean, std=std, device=device
    )
    for m in score_metrics:
        logger.info('==== TEST %s score on %s: %.4f' % (m, args.test_dataset, average_score_dict[m]))
else:
    score_metrics = ['root', 'majmin']
    score_list_dict, song_length_list, average_score_dict = root_majmin_score_calculation(
        valid_dataset=test_dataset, config=config, model=model, model_type=args.model, 
        mean=mean, std=std, device=device
    )
    for m in score_metrics:
        logger.info('==== TEST %s score on %s: %.4f' % (m, args.test_dataset, average_score_dict[m]))

logger.info("==== Testing completed ====")