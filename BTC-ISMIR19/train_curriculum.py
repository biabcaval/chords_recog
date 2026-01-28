"""
Training script with Curriculum Learning support for BTC model.

This script extends the original train.py with curriculum learning capabilities.
"""

import os
from torch import optim
from utils import logger
from data.audio_dataset import AudioDataset, AudioDataLoader
from utils.tf_logger import TF_Logger
from models.btc_model import *
from models.baseline_models import CNN, CRNN
from utils.hparams import HParams
import argparse
from utils.pytorch_utils import adjusting_learning_rate
from utils.mir_eval_modules import root_majmin_score_calculation, large_voca_score_calculation
from utils.transformer_modules import compute_class_weights, compute_structured_class_weights
from data.curriculum_learning import CurriculumLearning, CurriculumDataLoader
import warnings
import torch
import numpy as np
from sortedcontainers import SortedList

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logger.logging_verbosity(1)

# Try to import wandb (after logger is initialized)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Install with: pip install wandb")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, help='Experiment Number', default='e')
parser.add_argument('--kfold', type=int, help='5 fold (0,1,2,3,4)', default='e')
parser.add_argument('--voca', action='store_true', help='large voca is True', default=False)
parser.add_argument('--model', type=str, help='btc, cnn, crnn, chordformer', default='btc')
parser.add_argument('--dataset1', type=str, help='Dataset', default='billboard')
parser.add_argument('--dataset2', type=str, help='Dataset', default='jaah')
parser.add_argument('--dataset3', type=str, help='Third Dataset', default=None)
parser.add_argument('--dataset4', type=str, help='Fourth Dataset', default=None)
parser.add_argument('--dataset5', type=str, help='Fifth Dataset', default=None)
parser.add_argument('--test_dataset', type=str, help='Test Dataset', default='rwc')
parser.add_argument('--restore_epoch', type=int, default=1000)
parser.add_argument('--early_stop', action='store_true', help='no improvement during 10 epoch -> stop', default=True)
parser.add_argument('--curriculum', action='store_true', help='Enable curriculum learning', default=False)
args = parser.parse_args()

config = HParams.load("run_config.yaml")
if args.voca == True:
    config.feature['large_voca'] = True
    config.model['num_chords'] = 170

# Override curriculum setting from command line
if args.curriculum:
    config.curriculum['enabled'] = True
    logger.info("==== Curriculum Learning ENABLED ====")
    logger.info(f"Strategy: {config.curriculum['strategy']}")
    logger.info(f"Pacing: {config.curriculum['pacing']}")
    logger.info(f"Start Ratio: {config.curriculum['start_ratio']}")
    logger.info(f"Pace Epochs: {config.curriculum['pace_epochs']}")
else:
    config.curriculum['enabled'] = False
    logger.info("==== Curriculum Learning DISABLED ====")

# Result save path
asset_path = config.path['asset_path']
restore_epoch = args.restore_epoch
experiment_num = str(args.index)

# Create experiment folder structure if using k-fold training
experiment_folder = os.environ.get('EXPERIMENT_FOLDER', None)
kfold_num = os.environ.get('KFOLD_NUM', None)

if experiment_folder and kfold_num:
    # New organized structure: assets/experiment_{params}/kfold_{num}/
    experiment_dir = os.path.join(asset_path, experiment_folder)
    kfold_dir = os.path.join(experiment_dir, f'kfold_{kfold_num}')
    ckpt_path = kfold_dir
    result_path = os.path.join(experiment_dir, 'result')
    ckpt_file_name = 'model_%03d.pth.tar'
    logger.info(f"==== Using organized folder structure ====")
    logger.info(f"Experiment folder: {experiment_folder}")
    logger.info(f"K-Fold: {kfold_num}")
    logger.info(f"Checkpoint path: {ckpt_path}")
else:
    # Legacy structure for backward compatibility
    ckpt_path = config.path['ckpt_path']
    result_path = config.path['result_path']
    ckpt_file_name = 'idx_'+experiment_num+'_%03d.pth.tar'
    logger.info("==== Using legacy folder structure ====")

tf_logger = TF_Logger(os.path.join(asset_path, 'tensorboard', 'idx_'+experiment_num))
logger.info("==== Experiment Number : %d " % args.index)

# Initialize wandb
if WANDB_AVAILABLE:
    try:
        wandb.init(
            project="chordMax",
            entity="teste-time",
            name=f"exp{args.index}_kfold{args.kfold}_{args.model}",
            config={
                'experiment_index': args.index,
                'kfold': args.kfold,
                'model': args.model,
                'dataset1': args.dataset1,
                'dataset2': args.dataset2,
                'dataset3': args.dataset3 if args.dataset3 else 'None',
                'dataset4': args.dataset4 if args.dataset4 else 'None',
                'dataset5': args.dataset5 if args.dataset5 else 'None',
                'test_dataset': args.test_dataset,
                'voca': args.voca,
                'curriculum': args.curriculum,
                'restore_epoch': args.restore_epoch,
                'early_stop': args.early_stop,
                'learning_rate': config.experiment['learning_rate'],
                'batch_size': config.experiment['batch_size'],
                'max_epoch': config.experiment['max_epoch'],
                'num_chords': config.model['num_chords'],
                'large_voca': config.feature['large_voca'],
                'curriculum_strategy': config.curriculum.get('strategy', 'N/A'),
                'curriculum_pacing': config.curriculum.get('pacing', 'N/A'),
                'curriculum_start_ratio': config.curriculum.get('start_ratio', 'N/A'),
                'curriculum_pace_epochs': config.curriculum.get('pace_epochs', 'N/A'),
                'class_weights_enabled': config.class_weights.get('enabled', False),
                'class_weights_gamma': config.class_weights.get('gamma', 'N/A'),
                'class_weights_w_max': config.class_weights.get('w_max', 'N/A'),
            },
            tags=[args.model, f"kfold{args.kfold}", "curriculum" if args.curriculum else "standard"]
        )
        logger.info("==== Weights & Biases logging initialized ====")
        logger.info(f"wandb run URL: {wandb.run.url}")
    except Exception as e:
        logger.error(f"Failed to initialize wandb: {e}")
        WANDB_AVAILABLE = False
else:
    logger.warning("wandb is not available. Install with: pip install wandb")

if args.model == 'cnn':
    config.experiment['batch_size'] = 10

# Helper function to load entire dataset (for test set)
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
                # Only use non-augmented data (same as validation)
                for instance_name in instance_names:
                    if "1.00_0" in instance_name:
                        all_paths.append(os.path.join(song_path, instance_name))
    
    logger.info(f"Loaded {len(used_song_names)} songs with {len(all_paths)} instances from {dataset_name} (entire dataset)")
    return used_song_names, all_paths

# Data loader
# Training datasets: billboard, jaah, and optionally dj_avan (with k-fold split)
train_dataset1 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset1,), num_workers=20, preprocessing=False, train=True, kfold=args.kfold)
train_dataset2 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset2,), num_workers=20, preprocessing=False, train=True, kfold=args.kfold)
train_dataset = train_dataset1.__add__(train_dataset2)

# Add third dataset if provided
if args.dataset3:
    train_dataset3 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset3,), num_workers=20, preprocessing=False, train=True, kfold=args.kfold)
    train_dataset = train_dataset.__add__(train_dataset3)

# Add fourth dataset if provided
if args.dataset4:
    train_dataset4 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset4,), num_workers=20, preprocessing=False, train=True, kfold=args.kfold)
    train_dataset = train_dataset.__add__(train_dataset4)

# Add fifth dataset if provided
if args.dataset5:
    train_dataset5 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset5,), num_workers=20, preprocessing=False, train=True, kfold=args.kfold)
    train_dataset = train_dataset.__add__(train_dataset5)

# Validation datasets: billboard, jaah, and optionally dj_avan (one fold for validation)
valid_dataset1 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset1,), preprocessing=False, train=False, kfold=args.kfold)
valid_dataset2 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset2,), preprocessing=False, train=False, kfold=args.kfold)
valid_dataset = valid_dataset1.__add__(valid_dataset2)

# Add third dataset to validation if provided
if args.dataset3:
    valid_dataset3 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset3,), preprocessing=False, train=False, kfold=args.kfold)
    valid_dataset = valid_dataset.__add__(valid_dataset3)

# Add fourth dataset to validation if provided
if args.dataset4:
    valid_dataset4 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset4,), preprocessing=False, train=False, kfold=args.kfold)
    valid_dataset = valid_dataset.__add__(valid_dataset4)

# Test dataset: rwc (entire dataset, no k-fold split)
test_song_names, test_paths = load_all_test_data(config, config.path['root_path'], args.test_dataset)

# Create a custom dataset class for test data
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, paths, song_names, config, root_dir, dataset_name):
        self.paths = paths
        self.song_names = song_names
        # Create a preprocessor for the scoring function to use
        from utils.preprocess import Preprocess, FeatureTypes
        # Use feature type from config instead of hardcoded 'cqt'
        feature_type_str = config.feature.get('type', 'cqt')
        feature_type = FeatureTypes.hcqt if feature_type_str == 'hcqt' else FeatureTypes.cqt
        self.preprocessor = Preprocess(config, feature_type, (dataset_name,), root_dir)
    
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

# Initialize curriculum learning
curriculum = CurriculumLearning(config, train_dataset, logger=logger)

# Create data loaders
if config.curriculum['enabled']:
    # Use curriculum dataloader for training
    from data.audio_dataset import _collate_fn
    train_dataloader = CurriculumDataLoader(
        dataset=train_dataset,
        curriculum=curriculum,
        batch_size=config.experiment['batch_size'],
        collate_fn=_collate_fn,
        drop_last=False
    )
    logger.info("Using CurriculumDataLoader for training")
else:
    # Use standard dataloader
    train_dataloader = AudioDataLoader(dataset=train_dataset, batch_size=config.experiment['batch_size'], drop_last=False, shuffle=True)
    logger.info("Using standard AudioDataLoader for training")

valid_dataloader = AudioDataLoader(dataset=valid_dataset, batch_size=config.experiment['batch_size'], drop_last=False)
test_dataloader = AudioDataLoader(dataset=test_dataset, batch_size=config.experiment['batch_size'], drop_last=False)

# Compute class weights if enabled
class_weights = None
root_class_weights = None
quality_class_weights = None
bass_class_weights = None

if config.class_weights.get('enabled', False):
    logger.info("==== Computing Class Weights for Reweighted Loss ====")
    class_weights = compute_class_weights(
        train_dataset=train_dataset,
        num_classes=config.model['num_chords'],
        gamma=config.class_weights.get('gamma', 0.5),
        w_max=config.class_weights.get('w_max', 10.0),
        device=device
    )
    logger.info(f"Class weights computed: min={class_weights.min():.4f}, max={class_weights.max():.4f}, mean={class_weights.mean():.4f}")
    # Log weight distribution
    logger.info(f"Weight statistics: std={class_weights.std():.4f}, median={class_weights.median():.4f}")
    
    # Compute per-component class weights for structured models (ChordFormer approach)
    if args.model in ['btc_structured', 'chordformer']:
        logger.info("==== Computing Per-Component Class Weights for Structured Output ====")
        root_class_weights, quality_class_weights, bass_class_weights = compute_structured_class_weights(
            train_dataset=train_dataset,
            num_roots=config.model.get('num_roots', 13),
            num_qualities=config.model.get('num_qualities', 16),
            num_bass=config.model.get('num_bass', 13),
            gamma=config.class_weights.get('gamma', 0.5),
            w_max=config.class_weights.get('w_max', 10.0),
            device=device
        )
        if root_class_weights is not None:
            logger.info(f"Root weights: min={root_class_weights.min():.4f}, max={root_class_weights.max():.4f}, mean={root_class_weights.mean():.4f}")
            logger.info(f"Quality weights: min={quality_class_weights.min():.4f}, max={quality_class_weights.max():.4f}, mean={quality_class_weights.mean():.4f}")
            logger.info(f"Bass weights: min={bass_class_weights.min():.4f}, max={bass_class_weights.max():.4f}, mean={bass_class_weights.mean():.4f}")
        else:
            logger.warning("Structured labels not found in dataset. Per-component weights not computed.")

# Model and Optimizer
# Add feature type and related params to model config
config.model['feature_type'] = config.feature['type']
config.model['n_harmonics'] = config.feature.get('n_harmonics', 6)
config.model['n_bins'] = config.feature.get('n_bins', 144)

if args.model == 'cnn':
    model = CNN(config=config.model).to(device)
elif args.model == 'crnn':
    model = CRNN(config=config.model).to(device)
elif args.model == 'btc':
    model = BTC_model(config=config.model, class_weights=class_weights).to(device)
elif args.model == 'btc_structured':
    from models.btc_model import BTC_model_structured
    config.model['use_structured_output'] = True
    model = BTC_model_structured(
        config=config.model, 
        class_weights=class_weights,
        root_class_weights=root_class_weights,
        quality_class_weights=quality_class_weights,
        bass_class_weights=bass_class_weights
    ).to(device)
    logger.info("==== Using BTC model with structured output (Root, Quality, Bass) ====")
elif args.model == 'chordformer':
    from models.btc_model import ChordFormer_model
    config.model['use_structured_output'] = True
    model = ChordFormer_model(
        config=config.model,
        class_weights=class_weights,
        root_class_weights=root_class_weights,
        quality_class_weights=quality_class_weights,
        bass_class_weights=bass_class_weights
    ).to(device)
    logger.info("==== Using ChordFormer model (Conformer + Structured Output) ====")
    logger.info(f"  Conv kernel size: {config.model.get('conv_kernel_size', 31)}")
    logger.info(f"  FF expansion factor: {config.model.get('ff_expansion_factor', 4)}")
    logger.info(f"  Conv expansion factor: {config.model.get('conv_expansion_factor', 2)}")
else: 
    raise NotImplementedError
optimizer = optim.Adam(model.parameters(), lr=config.experiment['learning_rate'], weight_decay=config.experiment['weight_decay'], betas=(0.9, 0.98), eps=1e-9)

# Make asset directory
if experiment_folder and kfold_num:
    # New structure: ckpt_path is already the full path
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)
else:
    # Legacy structure
    if not os.path.exists(os.path.join(asset_path, ckpt_path)):
        os.makedirs(os.path.join(asset_path, ckpt_path))
        os.makedirs(os.path.join(asset_path, result_path), exist_ok=True)

# Load model
if experiment_folder and kfold_num:
    checkpoint_file = os.path.join(ckpt_path, ckpt_file_name % restore_epoch)
else:
    checkpoint_file = os.path.join(asset_path, ckpt_path, ckpt_file_name % restore_epoch)

if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    logger.info("restore model with %d epochs" % restore_epoch)
else:
    logger.info("no checkpoint with %d epochs" % restore_epoch)
    restore_epoch = 0

# Global mean and variance calculate
mp3_config = config.mp3
feature_config = config.feature
mp3_string = "%d_%.1f_%.1f" % (mp3_config['song_hz'], mp3_config['inst_len'], mp3_config['skip_interval'])
# Use actual feature type from config instead of hardcoded 'cqt'
feature_type = feature_config.get('type', 'cqt')
feature_string = "_%s_%d_%d_%d_" % (feature_type, feature_config['n_bins'], feature_config['bins_per_octave'], feature_config['hop_length'])
z_path = os.path.join(config.path['root_path'], 'result', mp3_string + feature_string + 'mix_kfold_'+ str(args.kfold) +'_normalization.pt')
if os.path.exists(z_path):
    normalization = torch.load(z_path)
    mean = normalization['mean']
    std = normalization['std']
    # Ensure mean and std are Python floats, not tensors
    if isinstance(mean, torch.Tensor):
        mean = mean.item()
    if isinstance(std, torch.Tensor):
        std = std.item()
    logger.info("Global mean and std (k fold index %d) load complete" % args.kfold)
else:
    mean = 0
    square_mean = 0
    k = 0
    # Use a temporary loader for normalization calculation
    if config.curriculum['enabled']:
        temp_loader = AudioDataLoader(dataset=train_dataset, batch_size=config.experiment['batch_size'], drop_last=False, shuffle=False)
    else:
        temp_loader = train_dataloader
    
    for i, data in enumerate(temp_loader):
        # Handle both structured and non-structured output cases
        if config.model.get('use_structured_output', False):
            features, input_percentages, chords, collapsed_chords, chord_lens, boundaries, roots, qualities, basses = data
        else:
            features, input_percentages, chords, collapsed_chords, chord_lens, boundaries = data
        features = features.to(device)
        mean += torch.mean(features).item()
        square_mean += torch.mean(features.pow(2)).item()
        k += 1
    square_mean = square_mean / k
    mean = mean / k
    std = np.sqrt(square_mean - mean * mean)
    normalization = dict()
    normalization['mean'] = mean
    normalization['std'] = std
    torch.save(normalization, z_path)
    logger.info("Global mean and std (training set, k fold index %d) calculation complete" % args.kfold)

current_step = 0
best_acc = 0
before_acc = 0
early_stop_idx = 0
last_best_epoch = restore_epoch

for epoch in range(restore_epoch, config.experiment['max_epoch']):
    # Update curriculum for current epoch
    if config.curriculum['enabled']:
        train_dataloader.set_epoch(epoch)
        curriculum_stats = curriculum.get_stats(epoch)
        logger.info("==== Curriculum Learning Stats (Epoch %d) ====" % (epoch + 1))
        logger.info(f"  Data Ratio: {curriculum_stats['ratio']:.2%} ({curriculum_stats['num_samples']}/{curriculum_stats['total_samples']} samples)")
        logger.info(f"  Mean Difficulty: {curriculum_stats['mean_difficulty']:.4f}")
        logger.info(f"  Difficulty Range: [{curriculum_stats['min_difficulty']:.4f}, {curriculum_stats['max_difficulty']:.4f}]")
        
        # Log to tensorboard
        tf_logger.scalar_summary('curriculum/ratio', curriculum_stats['ratio'], epoch+1)
        tf_logger.scalar_summary('curriculum/mean_difficulty', curriculum_stats['mean_difficulty'], epoch+1)
        tf_logger.scalar_summary('curriculum/num_samples', curriculum_stats['num_samples'], epoch+1)
        
        # Log to wandb
        if WANDB_AVAILABLE:
            wandb.log({
                'curriculum/ratio': curriculum_stats['ratio'],
                'curriculum/mean_difficulty': curriculum_stats['mean_difficulty'],
                'curriculum/num_samples': curriculum_stats['num_samples'],
                'curriculum/min_difficulty': curriculum_stats['min_difficulty'],
                'curriculum/max_difficulty': curriculum_stats['max_difficulty'],
            }, step=epoch+1)
    
    # Training
    model.train()
    train_loss_list = []
    total = 0.
    correct = 0.
    second_correct = 0.
    
    # Additional metrics for structured output
    root_correct = 0.
    quality_correct = 0.
    bass_correct = 0.
    
    for i, data in enumerate(train_dataloader):
        # Handle both structured and non-structured data
        if len(data) == 9:  # Structured output
            features, input_percentages, chords, collapsed_chords, chord_lens, boundaries, roots, qualities, basses = data
            features = features.to(device)
            roots = roots.to(device)
            qualities = qualities.to(device)
            basses = basses.to(device)
            use_structured = True
        else:  # Non-structured output
            features, input_percentages, chords, collapsed_chords, chord_lens, boundaries = data
            features = features.to(device)
            chords = chords.to(device)
            use_structured = False

        features.requires_grad = True
        features = (features - mean) / std

        # forward
        features = features.squeeze(1).permute(0,2,1)
        optimizer.zero_grad()
        
        if use_structured and args.model in ['btc_structured', 'chordformer']:
            # Structured model forward pass
            output = model(features, root_labels=roots, quality_labels=qualities, bass_labels=basses)
            total_loss = output['loss']
            
            # Calculate accuracy for each component
            total += roots.size(0)
            root_correct += (output['root_pred'] == roots.view(-1)).type_as(roots).sum()
            quality_correct += (output['quality_pred'] == qualities.view(-1)).type_as(qualities).sum()
            bass_correct += (output['bass_pred'] == basses.view(-1)).type_as(basses).sum()
            
            # For compatibility, compute overall correctness (all components must match)
            correct_mask = ((output['root_pred'] == roots.view(-1)) & 
                           (output['quality_pred'] == qualities.view(-1)) & 
                           (output['bass_pred'] == basses.view(-1)))
            correct += correct_mask.type_as(roots).sum()
            
            # Second best (at least one component matches second best)
            second_mask = ((output['root_second'] == roots.view(-1)) | 
                          (output['quality_second'] == qualities.view(-1)) | 
                          (output['bass_second'] == basses.view(-1)))
            second_correct += second_mask.type_as(roots).sum()
        else:
            # Original model forward pass
            # Ensure chords is on the correct device before passing to model
            assert chords.device == features.device, f"Device mismatch: features on {features.device}, chords on {chords.device}"
            prediction, total_loss, weights, second = model(features, chords)
            
            # save accuracy and loss
            total += chords.size(0)
            correct += (prediction == chords).type_as(chords).sum()
            second_correct += (second == chords).type_as(chords).sum()
        
        train_loss_list.append(total_loss.item())

        # optimize step
        total_loss.backward()
        optimizer.step()

        current_step += 1

    # logging loss and accuracy using tensorboard
    result = {'loss/tr': np.mean(train_loss_list), 'acc/tr': correct.item() / total, 'top2/tr': (correct.item()+second_correct.item()) / total}
    
    # Add structured metrics if using structured model
    if args.model in ['btc_structured', 'chordformer'] and root_correct > 0:
        result['acc/tr_root'] = root_correct.item() / total
        result['acc/tr_quality'] = quality_correct.item() / total
        result['acc/tr_bass'] = bass_correct.item() / total
    
    for tag, value in result.items(): 
        tf_logger.scalar_summary(tag, value, epoch+1)
    logger.info("training loss for %d epoch: %.4f" % (epoch + 1, np.mean(train_loss_list)))
    logger.info("training accuracy for %d epoch: %.4f" % (epoch + 1, (correct.item() / total)))
    logger.info("training top2 accuracy for %d epoch: %.4f" % (epoch + 1, ((correct.item() + second_correct.item()) / total)))
    
    if args.model in ['btc_structured', 'chordformer'] and root_correct > 0:
        logger.info("training root accuracy for %d epoch: %.4f" % (epoch + 1, (root_correct.item() / total)))
        logger.info("training quality accuracy for %d epoch: %.4f" % (epoch + 1, (quality_correct.item() / total)))
        logger.info("training bass accuracy for %d epoch: %.4f" % (epoch + 1, (bass_correct.item() / total)))
    
    # Log to wandb
    if WANDB_AVAILABLE:
        wandb_metrics = {
            'train/loss': np.mean(train_loss_list),
            'train/accuracy': correct.item() / total,
            'train/top2_accuracy': (correct.item()+second_correct.item()) / total,
        }
        if args.model in ['btc_structured', 'chordformer'] and root_correct > 0:
            wandb_metrics.update({
                'train/root_accuracy': root_correct.item() / total,
                'train/quality_accuracy': quality_correct.item() / total,
                'train/bass_accuracy': bass_correct.item() / total,
            })
        wandb.log(wandb_metrics, step=epoch+1)

    # Validation
    with torch.no_grad():
        model.eval()
        val_total = 0.
        val_correct = 0.
        val_second_correct = 0.
        validation_loss = 0
        n = 0
        
        # Additional metrics for structured output
        val_root_correct = 0.
        val_quality_correct = 0.
        val_bass_correct = 0.
        
        for i, data in enumerate(valid_dataloader):
            # Handle both structured and non-structured data
            if len(data) == 9:  # Structured output
                val_features, val_input_percentages, val_chords, val_collapsed_chords, val_chord_lens, val_boundaries, val_roots, val_qualities, val_basses = data
                val_features = val_features.to(device)
                val_roots = val_roots.to(device)
                val_qualities = val_qualities.to(device)
                val_basses = val_basses.to(device)
                use_structured = True
            else:  # Non-structured output
                val_features, val_input_percentages, val_chords, val_collapsed_chords, val_chord_lens, val_boundaries = data
                val_features, val_chords = val_features.to(device), val_chords.to(device)
                use_structured = False

            val_features = (val_features - mean) / std
            val_features = val_features.squeeze(1).permute(0, 2, 1)
            
            if use_structured and args.model in ['btc_structured', 'chordformer']:
                # Structured model forward pass
                output = model(val_features, root_labels=val_roots, quality_labels=val_qualities, bass_labels=val_basses)
                val_loss = output['loss']
                
                # Calculate accuracy for each component
                val_total += val_roots.size(0)
                val_root_correct += (output['root_pred'] == val_roots.view(-1)).type_as(val_roots).sum()
                val_quality_correct += (output['quality_pred'] == val_qualities.view(-1)).type_as(val_qualities).sum()
                val_bass_correct += (output['bass_pred'] == val_basses.view(-1)).type_as(val_basses).sum()
                
                # For compatibility, compute overall correctness
                correct_mask = ((output['root_pred'] == val_roots.view(-1)) & 
                               (output['quality_pred'] == val_qualities.view(-1)) & 
                               (output['bass_pred'] == val_basses.view(-1)))
                val_correct += correct_mask.type_as(val_roots).sum()
                
                # Second best
                second_mask = ((output['root_second'] == val_roots.view(-1)) | 
                              (output['quality_second'] == val_qualities.view(-1)) | 
                              (output['bass_second'] == val_basses.view(-1)))
                val_second_correct += second_mask.type_as(val_roots).sum()
            else:
                # Original model forward pass
                val_prediction, val_loss, weights, val_second = model(val_features, val_chords)
                
                val_total += val_chords.size(0)
                val_correct += (val_prediction == val_chords).type_as(val_chords).sum()
                val_second_correct += (val_second == val_chords).type_as(val_chords).sum()
            
            validation_loss += val_loss.item()
            n += 1

        # logging loss and accuracy using tensorboard
        validation_loss /= n
        result = {'loss/val': validation_loss, 'acc/val': val_correct.item() / val_total, 'top2/val': (val_correct.item()+val_second_correct.item()) / val_total}
        
        # Add structured metrics if using structured model
        if args.model in ['btc_structured', 'chordformer'] and val_root_correct > 0:
            result['acc/val_root'] = val_root_correct.item() / val_total
            result['acc/val_quality'] = val_quality_correct.item() / val_total
            result['acc/val_bass'] = val_bass_correct.item() / val_total
        
        for tag, value in result.items(): 
            tf_logger.scalar_summary(tag, value, epoch + 1)
        logger.info("validation loss(%d): %.4f" % (epoch + 1, validation_loss))
        logger.info("validation accuracy(%d): %.4f" % (epoch + 1, (val_correct.item() / val_total)))
        logger.info("validation top2 accuracy(%d): %.4f" % (epoch + 1, ((val_correct.item() + val_second_correct.item()) / val_total)))
        
        if args.model in ['btc_structured', 'chordformer'] and val_root_correct > 0:
            logger.info("validation root accuracy(%d): %.4f" % (epoch + 1, (val_root_correct.item() / val_total)))
            logger.info("validation quality accuracy(%d): %.4f" % (epoch + 1, (val_quality_correct.item() / val_total)))
            logger.info("validation bass accuracy(%d): %.4f" % (epoch + 1, (val_bass_correct.item() / val_total)))
        
        # Log to wandb
        if WANDB_AVAILABLE:
            wandb_val_metrics = {
                'val/loss': validation_loss,
                'val/accuracy': val_correct.item() / val_total,
                'val/top2_accuracy': (val_correct.item()+val_second_correct.item()) / val_total,
                'val/best_accuracy': best_acc,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
            }
            if args.model in ['btc_structured', 'chordformer'] and val_root_correct > 0:
                wandb_val_metrics.update({
                    'val/root_accuracy': val_root_correct.item() / val_total,
                    'val/quality_accuracy': val_quality_correct.item() / val_total,
                    'val/bass_accuracy': val_bass_correct.item() / val_total,
                })
            wandb.log(wandb_val_metrics, step=epoch+1)

        current_acc = val_correct.item() / val_total

        if best_acc < val_correct.item() / val_total:
            early_stop_idx = 0
            best_acc = val_correct.item() / val_total
            logger.info('==== best accuracy is %.4f and epoch is %d' % (best_acc, epoch + 1))
            logger.info('saving model, Epoch %d, step %d' % (epoch + 1, current_step + 1))
            if experiment_folder and kfold_num:
                model_save_path = os.path.join(ckpt_path, ckpt_file_name % (epoch + 1))
            else:
                model_save_path = os.path.join(asset_path, 'model', ckpt_file_name % (epoch + 1))
            state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
            torch.save(state_dict, model_save_path)
            last_best_epoch = epoch + 1
            
            # Log best accuracy to wandb
            if WANDB_AVAILABLE:
                wandb.log({'val/best_accuracy': best_acc, 'val/best_epoch': epoch + 1}, step=epoch+1)

        # save model
        elif (epoch + 1) % config.experiment['save_step'] == 0:
            logger.info('saving model, Epoch %d, step %d' % (epoch + 1, current_step + 1))
            if experiment_folder and kfold_num:
                model_save_path = os.path.join(ckpt_path, ckpt_file_name % (epoch + 1))
            else:
                model_save_path = os.path.join(asset_path, 'model', ckpt_file_name % (epoch + 1))
            state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
            torch.save(state_dict, model_save_path)
            early_stop_idx += 1
        else:
            early_stop_idx += 1

    if (args.early_stop == True) and (early_stop_idx > 9):
        logger.info('==== early stopped and epoch is %d' % (epoch + 1))
        break
    # learning rate decay
    if before_acc > current_acc:
        adjusting_learning_rate(optimizer=optimizer, factor=0.95, min_lr=5e-6)
    before_acc = current_acc

# Load best model after training
if experiment_folder and kfold_num:
    best_checkpoint_file = os.path.join(ckpt_path, ckpt_file_name % last_best_epoch)
else:
    best_checkpoint_file = os.path.join(asset_path, ckpt_path, ckpt_file_name % last_best_epoch)

if os.path.isfile(best_checkpoint_file):
    checkpoint = torch.load(best_checkpoint_file)
    model.load_state_dict(checkpoint['model'])
    logger.info("restore model with %d epochs for testing" % last_best_epoch)
else:
    raise NotImplementedError

# ===== AUTOMATIC TESTING ON RWC DATASET =====
logger.info("==== Starting automatic testing on %s dataset ====" % args.test_dataset)

# Test on entire rwc dataset
if args.voca == True:
    score_metrics = ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']
    score_list_dict, song_length_list, average_score_dict, wcsr_dict = large_voca_score_calculation(valid_dataset=test_dataset, config=config, model=model, model_type=args.model, mean=mean, std=std, device=device)
    for m in score_metrics:
        logger.info('==== TEST %s score on %s: %.4f' % (m, args.test_dataset, average_score_dict[m]))
        logger.info('==== TEST %s WCSR on %s: %.2f%%' % (m, args.test_dataset, wcsr_dict[m]))
else:
    score_metrics = ['root', 'majmin']
    score_list_dict, song_length_list, average_score_dict, wcsr_dict = root_majmin_score_calculation(valid_dataset=test_dataset, config=config, model=model, model_type=args.model, mean=mean, std=std, device=device)
    for m in score_metrics:
        logger.info('==== TEST %s score on %s: %.4f' % (m, args.test_dataset, average_score_dict[m]))
        logger.info('==== TEST %s WCSR on %s: %.2f%%' % (m, args.test_dataset, wcsr_dict[m]))

# Log test scores to wandb
if WANDB_AVAILABLE:
    test_metrics = {f'test/{m}': average_score_dict[m] for m in score_metrics}
    # Add WCSR metrics
    wcsr_metrics = {f'test/{m}_wcsr': wcsr_dict[m] for m in score_metrics}
    test_metrics.update(wcsr_metrics)
    wandb.log(test_metrics)
    logger.info("==== Test scores and WCSR logged to wandb ====")

logger.info("==== Testing completed ====")

# Finalize wandb run
if WANDB_AVAILABLE:
    wandb.finish()
    logger.info("==== Weights & Biases run completed ====")

