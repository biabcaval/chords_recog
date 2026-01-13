#!/usr/bin/env python
"""
Batch inference script to generate .lab files from checkpoints.
Outputs are saved to inference folders named by training and test dataset.
"""
import os
import sys
import torch
import numpy as np
import librosa
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import logger
from utils.hparams import HParams
from models.btc_model import BTC_model

logger.logging_verbosity(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Chord vocabulary mappings
idx2chord_small = ['C', 'C:min', 'C#', 'C#:min', 'D', 'D:min', 'D#', 'D#:min', 'E', 'E:min', 'F', 'F:min', 'F#',
                   'F#:min', 'G', 'G:min', 'G#', 'G#:min', 'A', 'A:min', 'A#', 'A#:min', 'B', 'B:min', 'N']

root_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
quality_list = ['min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7', 'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4']

def idx2voca_chord():
    """Create mapping from index to chord label for large vocabulary (170 chords)"""
    idx2voca = {}
    idx2voca[169] = 'N'
    idx2voca[168] = 'X'
    for i in range(168):
        root = i // 14
        root = root_list[root]
        quality = i % 14
        quality = quality_list[quality]
        if i % 14 != 1:
            chord = root + ':' + quality
        else:
            chord = root
        idx2voca[i] = chord
    return idx2voca

def audio_file_to_features(audio_file, config):
    """Convert audio file to CQT features"""
    original_wav, sr = librosa.load(audio_file, sr=config.mp3['song_hz'], mono=True)
    current_sec_hz = 0
    feature = None
    
    while len(original_wav) > current_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len']:
        start_idx = int(current_sec_hz)
        end_idx = int(current_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len'])
        tmp = librosa.cqt(original_wav[start_idx:end_idx], sr=sr, 
                          n_bins=config.feature['n_bins'], 
                          bins_per_octave=config.feature['bins_per_octave'], 
                          hop_length=config.feature['hop_length'])
        if feature is None:
            feature = tmp
        else:
            feature = np.concatenate((feature, tmp), axis=1)
        current_sec_hz = end_idx
    
    tmp = librosa.cqt(original_wav[current_sec_hz:], sr=sr, 
                      n_bins=config.feature['n_bins'], 
                      bins_per_octave=config.feature['bins_per_octave'], 
                      hop_length=config.feature['hop_length'])
    feature = np.concatenate((feature, tmp), axis=1)
    feature = np.log(np.abs(feature) + 1e-6)
    feature_per_second = config.mp3['inst_len'] / config.model['timestep']
    song_length_second = len(original_wav) / config.mp3['song_hz']
    return feature, feature_per_second, song_length_second

def get_audio_paths(audio_dir):
    """Get all audio file paths (mp3 and wav)"""
    paths = []
    for root, dirs, files in os.walk(audio_dir, followlinks=True):
        for fname in files:
            if fname.lower().endswith('.wav') or fname.lower().endswith('.mp3'):
                paths.append(os.path.join(root, fname))
    return sorted(paths)

def run_inference(checkpoint_path, config_path, audio_dir, output_dir, kfold_index, large_voca=True):
    """
    Run inference on all audio files in audio_dir and save .lab files to output_dir.
    """
    # Load config
    config = HParams.load(config_path)
    
    if large_voca:
        config.feature['large_voca'] = True
        config.model['num_chords'] = 170
        idx2chord = idx2voca_chord()
        logger.info("Using large vocabulary (170 chords)")
    else:
        config.feature['large_voca'] = False
        config.model['num_chords'] = 25
        idx2chord = idx2chord_small
        logger.info("Using majmin vocabulary (25 chords)")
    
    # Load normalization
    mp3_config = config.mp3
    feature_config = config.feature
    mp3_string = "%d_%.1f_%.1f" % (mp3_config['song_hz'], mp3_config['inst_len'], mp3_config['skip_interval'])
    feature_string = "_%s_%d_%d_%d_" % ('cqt', feature_config['n_bins'], feature_config['bins_per_octave'], feature_config['hop_length'])
    z_path = os.path.join(config.path['root_path'], 'result', mp3_string + feature_string + 'mix_kfold_' + str(kfold_index) + '_normalization.pt')
    
    if os.path.exists(z_path):
        normalization = torch.load(z_path, weights_only=False)
        mean = normalization['mean']
        std = normalization['std']
        logger.info(f"Loaded normalization from: {z_path}")
    else:
        raise FileNotFoundError(f"Normalization file not found: {z_path}")
    
    # Create model and load checkpoint
    model = BTC_model(config=config.model).to(device)
    
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get audio files
    audio_paths = get_audio_paths(audio_dir)
    logger.info(f"Found {len(audio_paths)} audio files in {audio_dir}")
    
    n_timestep = config.model['timestep']
    
    for audio_path in audio_paths:
        try:
            song_name = Path(audio_path).stem
            logger.info(f"Processing: {song_name}")
            
            # Get features
            feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, config)
            feature = feature.T
            feature = (feature - mean) / std
            time_unit = feature_per_second
            
            # Pad feature
            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
            num_instance = feature.shape[0] // n_timestep
            
            # Run inference
            start_time = 0.0
            lines = []
            
            with torch.no_grad():
                feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
                
                for t in range(num_instance):
                    encoder_output, _ = model.self_attn_layers(feature_tensor[:, n_timestep * t:n_timestep * (t + 1), :])
                    prediction, _ = model.output_layer(encoder_output)
                    prediction = prediction.squeeze()
                    
                    for i in range(n_timestep):
                        if t == 0 and i == 0:
                            prev_chord = prediction[i].item()
                            continue
                        
                        if prediction[i].item() != prev_chord:
                            lines.append('%.3f %.3f %s\n' % (
                                start_time, time_unit * (n_timestep * t + i), idx2chord[prev_chord]))
                            start_time = time_unit * (n_timestep * t + i)
                            prev_chord = prediction[i].item()
                        
                        if t == num_instance - 1 and i + num_pad == n_timestep:
                            if start_time != time_unit * (n_timestep * t + i):
                                lines.append('%.3f %.3f %s\n' % (
                                    start_time, time_unit * (n_timestep * t + i), idx2chord[prev_chord]))
                            break
            
            # Save .lab file
            output_path = os.path.join(output_dir, song_name + '.lab')
            with open(output_path, 'w') as f:
                for line in lines:
                    f.write(line)
            
            logger.info(f"Saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            continue
    
    logger.info(f"Inference complete. Results saved to {output_dir}")


def get_best_checkpoint(exp_dir):
    """Get the best checkpoint (highest epoch number) from an experiment directory"""
    checkpoints = []
    
    # Check for kfold structure
    kfold_dirs = [d for d in os.listdir(exp_dir) if d.startswith('kfold_')]
    
    if kfold_dirs:
        # Use kfold_0 by default, or first available
        kfold_dir = os.path.join(exp_dir, sorted(kfold_dirs)[0])
        for f in os.listdir(kfold_dir):
            if f.endswith('.pth.tar'):
                checkpoints.append(os.path.join(kfold_dir, f))
    else:
        # Direct checkpoint files
        for f in os.listdir(exp_dir):
            if f.endswith('.pth.tar'):
                checkpoints.append(os.path.join(exp_dir, f))
    
    if not checkpoints:
        return None
    
    # Sort by epoch number and return the highest
    def get_epoch(path):
        name = os.path.basename(path)
        # Extract number from model_XXX.pth.tar or idx_X_XXX.pth.tar
        parts = name.replace('.pth.tar', '').split('_')
        return int(parts[-1])
    
    checkpoints.sort(key=get_epoch, reverse=True)
    return checkpoints[0]


def get_kfold_from_checkpoint(checkpoint_path):
    """Extract kfold index from checkpoint path"""
    path_parts = checkpoint_path.split(os.sep)
    for part in path_parts:
        if part.startswith('kfold_'):
            return int(part.split('_')[1])
    # Default to 0 if not found
    return 0


# Dataset short codes mapping
DATASET_SHORT = {
    'billboard': 'Bi',
    'jaah': 'Ja',
    'rwc': 'Rw',
    'dj_avan': 'Dj',
    'djavan': 'Dj',
    'queen': 'Qu',
    'robbiewilliams': 'Ro',
    'robbie': 'Ro',
}

DATASET_AUDIO_DIRS = {
    'billboard': '/home/daniel.melo/datasets/billboard/audio',
    'jaah': '/home/daniel.melo/datasets/jaah/audio',
    'rwc': '/home/daniel.melo/datasets/rwc/audio',
    'dj_avan': '/home/daniel.melo/datasets/dj_avan/audio',
    'queen': '/home/daniel.melo/datasets/queen/audio',
    'robbiewilliams': '/home/daniel.melo/datasets/robbiewilliams/audio',
}


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run batch inference on audio files')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--test_dataset', type=str, required=True, help='Test dataset name (rwc, dj_avan, jaah, etc.)')
    parser.add_argument('--train_name', type=str, required=True, help='Short name for training datasets (e.g., BiJaRw)')
    parser.add_argument('--exp_num', type=str, default='', help='Experiment number for folder naming')
    parser.add_argument('--output_base', type=str, default='/home/daniel.melo/BTC_ORIGINAL/BTC-ISMIR19/inferences', 
                        help='Base directory for output')
    parser.add_argument('--config', type=str, default='/home/daniel.melo/BTC_ORIGINAL/chords_recog/BTC-ISMIR19/run_config.yaml',
                        help='Path to config file')
    parser.add_argument('--kfold', type=int, default=None, help='K-fold index (auto-detected if not provided)')
    parser.add_argument('--small_voca', action='store_true', help='Use small vocabulary (25 chords) instead of large (170)')
    
    args = parser.parse_args()
    
    # Get audio directory
    audio_dir = DATASET_AUDIO_DIRS.get(args.test_dataset)
    if audio_dir is None:
        raise ValueError(f"Unknown test dataset: {args.test_dataset}")
    
    # Get test dataset short code
    test_short = DATASET_SHORT.get(args.test_dataset, args.test_dataset[:2].capitalize())
    
    # Create output folder name
    if args.exp_num:
        output_name = f"inference_{args.exp_num}_{args.train_name}_test_{test_short}"
    else:
        output_name = f"inference_{args.train_name}_test_{test_short}"
    
    output_dir = os.path.join(args.output_base, output_name)
    
    # Determine kfold
    kfold = args.kfold if args.kfold is not None else get_kfold_from_checkpoint(args.checkpoint)
    
    print(f"Running inference:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Test dataset: {args.test_dataset}")
    print(f"  Audio dir: {audio_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  K-fold: {kfold}")
    print(f"  Vocabulary: {'small (25)' if args.small_voca else 'large (170)'}")
    
    run_inference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        audio_dir=audio_dir,
        output_dir=output_dir,
        kfold_index=kfold,
        large_voca=not args.small_voca
    )

