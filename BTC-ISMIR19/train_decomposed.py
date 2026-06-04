#!/usr/bin/env python
# encoding: utf-8
"""
Training script for chord recognition with structure decomposition.

This script demonstrates how to train the decomposed chord recognition model
with the 9-component architecture.

Usage:
    python train_decomposed.py --config run_config.yaml --device cuda:0
    python train_decomposed.py --backbone chordformer --run_name chordformer_decomp
"""

import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import logging
from pathlib import Path
import json
from datetime import datetime
import os
import sys
import importlib
import hashlib

from models.btc_model_decomposed import (
    BTC_model_decomposed,
    ChordFormer_model_decomposed,
    MultiTaskLoss,
)
from data.audio_dataset import EpochSampler
from data.audio_dataset_structured import AudioDatasetStructured, AudioDataLoaderStructured
from utils.decomposed_inference import DecomposedChordTrainer, DecomposedChordInference, ChordMetrics
from utils.chord_decomposition import COMPONENT_NAMES
from utils.hparams import HParams

# Optional wandb integration with protection against local-module shadowing.
def _import_wandb_package():
    """
    Import real wandb package even if a local ./wandb directory exists.

    Returns:
        tuple[module|None, bool, str]
        (wandb_module, available, diagnostic_message)
    """
    try:
        mod = importlib.import_module("wandb")
        if hasattr(mod, "init"):
            return mod, True, ""
    except ImportError:
        return None, False, "wandb package is not installed."

    # A local "wandb" folder may shadow the real SDK (common when wandb logs are
    # stored under ./wandb). Retry import with shadowing paths removed.
    original_sys_path = list(sys.path)
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()

    def _path_shadows_wandb(path_entry):
        try:
            base = cwd if path_entry in ("", ".") else Path(path_entry).resolve()
        except Exception:
            return False
        if base in (script_dir, cwd) and (base / "wandb").exists():
            return True
        return (base / "wandb").exists() and str(base).startswith(str(script_dir))

    try:
        sys.path = [p for p in sys.path if not _path_shadows_wandb(p)]
        sys.modules.pop("wandb", None)
        mod = importlib.import_module("wandb")
        if hasattr(mod, "init"):
            return mod, True, "Recovered wandb import after bypassing local shadowing path."
        return mod, False, "Imported module named wandb but it lacks wandb.init()."
    except ImportError:
        return None, False, "Failed to import wandb after removing shadowing paths."
    finally:
        sys.path = original_sys_path


wandb, WANDB_AVAILABLE, WANDB_IMPORT_DIAGNOSTIC = _import_wandb_package()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_component_weights(spec: str):
    """
    Parse component weight spec from CLI.

    Expected format:
      "root=1,bass=1,triad=1,misc=1,6th=1,7th=1,9th=1,11th=0.3,13th=0.3"
    """
    if spec is None:
        return None

    out = {}
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f"Invalid component weight item '{item}'. Expected key=value.")
        key, value = item.split('=', 1)
        key = key.strip()
        value = value.strip()
        if key not in COMPONENT_NAMES:
            raise ValueError(f"Unknown component '{key}'. Valid: {COMPONENT_NAMES}")
        try:
            out[key] = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value for '{key}': '{value}'") from exc

    # Fill unspecified components with default weight 1.0
    for comp in COMPONENT_NAMES:
        out.setdefault(comp, 1.0)
    return out


def _flatten_dict_for_wandb(data, prefix=""):
    """Flatten nested dicts so hyperparameters are easy to filter in wandb UI."""
    flat = {}
    for key, value in data.items():
        flat_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict_for_wandb(value, flat_key))
        else:
            flat[flat_key] = value
    return flat



def _hash_string_list(values):
    digest = hashlib.sha1()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:12]


def _to_serializable_component_weights(weights_dict):
    serializable = {}
    for component, weights in weights_dict.items():
        if isinstance(weights, torch.Tensor):
            serializable[component] = weights.detach().cpu()
        else:
            serializable[component] = torch.tensor(weights, dtype=torch.float32)
    return serializable


def _load_class_weights_file(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and 'class_weights' in payload:
        loaded_weights = payload['class_weights']
    elif isinstance(payload, dict):
        loaded_weights = payload
    else:
        raise ValueError("Invalid class-weights file format.")

    class_weights = {}
    for component, weights in loaded_weights.items():
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)
        class_weights[component] = weights.to(device)
    return class_weights, payload


def _build_class_weights_cache_path(
    cache_dir,
    dataset_names,
    kfold,
    gamma,
    w_max,
    train_paths,
):
    datasets_sig = _hash_string_list(dataset_names)
    paths_sig = _hash_string_list(train_paths)
    filename = (
        f"class_weights_k{kfold}_g{gamma:.4f}_w{w_max:.4f}_"
        f"ds{datasets_sig}_p{paths_sig}.pt"
    )
    return Path(cache_dir) / filename


def main():
    parser = argparse.ArgumentParser(
        description='Train chord recognition model with structure decomposition'
    )
    parser.add_argument('--config', type=str, default='run_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on (cuda, cpu, etc.)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Base directory to save checkpoints')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Name for this training run (creates subdirectory)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='Class weighting gamma parameter')
    parser.add_argument('--w_max', type=float, default=10.0,
                       help='Class weighting maximum cap')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Force enable class reweighting')
    parser.add_argument('--no_class_weights', action='store_true',
                       help='Force disable class reweighting')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval (batches)')
    parser.add_argument('--val_interval', type=int, default=1,
                       help='Validation interval (epochs)')
    parser.add_argument('--backbone', type=str, default='btc',
                       choices=['btc', 'chordformer'],
                       help='Backbone encoder for decomposed model')
    parser.add_argument('--use_head_ffn', action='store_true',
                       help='Add FFN bottleneck in each output head (ChordFormer only)')
    parser.add_argument('--head_ffn_dim', type=int, default=None,
                       help='FFN hidden dim in output heads (default: hidden_size//2)')
    parser.add_argument('--kfold', type=int, default=4, choices=[0, 1, 2, 3, 4],
                       help='5-fold split index used for validation (default: 4)')
    parser.add_argument('--component_weights', type=str, default=None,
                       help='Optional per-component loss weights as comma-separated key=value list')
    parser.add_argument('--use_gradnorm', action='store_true',
                       help='Enable GradNorm adaptive task balancing')
    parser.add_argument('--no_gradnorm', action='store_true',
                       help='Disable GradNorm adaptive task balancing')
    parser.add_argument('--gradnorm_alpha', type=float, default=None,
                       help='GradNorm asymmetry alpha (higher means stronger balancing)')
    parser.add_argument('--gradnorm_lr', type=float, default=None,
                       help='Learning rate for GradNorm task weights update')
    parser.add_argument('--gradnorm_eps', type=float, default=None,
                       help='Numerical epsilon for GradNorm')
    parser.add_argument('--gradnorm_w_min', type=float, default=None,
                       help='Minimum allowed GradNorm task weight before renormalization')
    parser.add_argument('--gradnorm_w_max', type=float, default=None,
                       help='Maximum allowed GradNorm task weight before renormalization')
    parser.add_argument('--focal_gamma', type=float, default=None,
                       help='Focal loss focusing parameter (0=standard CE, 2=recommended). Default: 0.0')
    parser.add_argument(
        '--class_weights_mode',
        type=str,
        default='auto',
        choices=['auto', 'compute', 'load'],
        help="Class-weight strategy when enabled: auto(load->compute), compute, or load-only."
    )
    parser.add_argument(
        '--class_weights_path',
        type=str,
        default=None,
        help='Optional explicit .pt file path for precomputed class weights.'
    )
    parser.add_argument(
        '--class_weights_cache_dir',
        type=str,
        default='./class_weights_cache',
        help='Cache directory for computed class weights (used by auto/compute modes).'
    )
    parser.add_argument('--train_datasets', type=str, nargs='+', default=None,
                       help='Datasets for training/validation (overrides config). Example: --train_datasets billboard jaah queen')
    parser.add_argument('--split_mode', type=str, default='legacy',
                       choices=['legacy', 'paper'],
                       help="Dataset split strategy. "
                            "'legacy' (default): 80/20 — train uses 4 folds, val uses fold `kfold`; "
                            "no test set. Compatible with existing runs. "
                            "'paper': 60/20/20 rotation (ChordFormer, Akram et al. 2026, IV-B). "
                            "test_fold=kfold%5, val_fold=(kfold+1)%5, train_folds=remaining 3. "
                            "All 3 splits come from --train_datasets (paper-faithful). "
                            "Enables EpochSampler (one random segment per song per epoch) on the train loader "
                            "and persists test_paths_kfold{k}.txt for downstream evaluation.")
    parser.add_argument('--wandb_api_key', type=str, default=None,
                       help='Weights & Biases API key (or set WANDB_API_KEY env var)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Weights & Biases entity (username or team)')
    parser.add_argument('--wandb_project', type=str, default='chordMax',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_disabled', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--normalization', type=str, default=None,
                       help='Path to normalization .pt file (mean/std). '
                            'When provided, features are standardized before training.')
    parser.add_argument('--no_class_distribution', action='store_true',
                       help='Disable per-head class distribution logging during validation')

    # ------------------------------------------------------------------
    # ChordFormer-replication knobs (optimizer / scheduler / CRF)
    # ------------------------------------------------------------------
    parser.add_argument('--optimizer', type=str, default=None,
                       choices=['adam', 'adamw'],
                       help="Optimizer family (ChordMax default: adam; ChordFormer paper: adamw). "
                            "When omitted, falls back to experiment.optimizer in the YAML, then 'adam'.")
    parser.add_argument('--scheduler', type=str, default=None,
                       choices=['cosine', 'plateau'],
                       help="LR scheduler (ChordMax default: cosine; ChordFormer paper: plateau). "
                            "When omitted, falls back to experiment.scheduler in the YAML, then 'cosine'.")
    parser.add_argument('--scheduler_factor', type=float, default=None,
                       help='ReduceLROnPlateau factor (default 0.1 = divide LR by 10).')
    parser.add_argument('--scheduler_patience', type=int, default=None,
                       help='ReduceLROnPlateau patience in epochs (default 5).')
    parser.add_argument('--scheduler_min_lr', type=float, default=None,
                       help='Minimum LR. With plateau, training early-stops when LR drops below this. '
                            'Default 1e-6 (ChordFormer paper).')
    parser.add_argument('--crf', type=str, default=None,
                       choices=['none', 'trainable', 'linear'],
                       help="CRF decoding stage (ChordMax default: trainable; ChordFormer paper: linear lambda=30). "
                            "Persisted to training_config so the post-training CRF stage knows which kind to instantiate. "
                            "When omitted, falls back to crf.type in the YAML, then 'none'.")
    parser.add_argument('--crf_lambda', type=float, default=None,
                       help='Self-transition bonus for the LinearCRF (default 30, used only when --crf=linear).')
    parser.add_argument('--disable_gradnorm', action='store_true',
                       help='Shortcut for --no_gradnorm (force-disable GradNorm).')

    # ------------------------------------------------------------------
    # GPU / throughput knobs
    # ------------------------------------------------------------------
    parser.add_argument('--no_amp', action='store_true',
                       help='Disable mixed precision (AMP). By default AMP is ON when running on CUDA.')
    parser.add_argument('--amp_dtype', type=str, default='bfloat16',
                       choices=['bfloat16', 'float16'],
                       help="AMP dtype. 'bfloat16' (default; great on A100/H100, no GradScaler needed) or 'float16'. "
                            "Note: float16 is currently incompatible with GradNorm (multiple backward paths).")
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of DataLoader workers (default: 4).')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                       help='DataLoader prefetch_factor (number of batches each worker prefetches; default: 4).')
    parser.add_argument('--no_pin_memory', action='store_true',
                       help='Disable pinned-memory in DataLoader (default: enabled when CUDA).')
    parser.add_argument('--no_persistent_workers', action='store_true',
                       help='Disable persistent_workers in DataLoader (default: enabled when num_workers > 0).')
    parser.add_argument('--no_cudnn_benchmark', action='store_true',
                       help='Disable cudnn.benchmark (helps when input shapes vary; we have fixed timestep=1000, so benchmark is recommended).')
    parser.add_argument('--compile', action='store_true',
                       help="Enable torch.compile(model). WARNING: may not be compatible with GradNorm "
                            "(secondary autograd.grad through last_shared_features). Default: off.")
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                       choices=['default', 'reduce-overhead', 'max-autotune'],
                       help='torch.compile mode (only when --compile is set).')

    args = parser.parse_args()
    args.log_class_distribution = not args.no_class_distribution

    # --disable_gradnorm is a friendlier alias for --no_gradnorm.
    if args.disable_gradnorm:
        args.no_gradnorm = True

    if args.use_class_weights and args.no_class_weights:
        parser.error("Use only one of --use_class_weights or --no_class_weights")
    if args.use_gradnorm and args.no_gradnorm:
        parser.error("Use only one of --use_gradnorm or --no_gradnorm (or --disable_gradnorm)")

    try:
        component_weights = parse_component_weights(args.component_weights)
    except ValueError as e:
        parser.error(str(e))

    # Never store/log credential values in run metadata.
    safe_cli_args = vars(args).copy()
    safe_cli_args.pop('wandb_api_key', None)
    
    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # ------------------------------------------------------------------
    # GPU throughput flags. Safe defaults for fixed-shape Conformer input.
    # ------------------------------------------------------------------
    is_cuda = (device.type == 'cuda')
    if is_cuda:
        # TF32 on Ampere+: ~10% free speedup with negligible accuracy impact.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except AttributeError:
            pass
        # Conv1D in Conformer benefits from cuDNN autotuner; we use a fixed
        # timestep so the autotuner pays off after the first batch.
        torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark
        logger.info(
            f"GPU perf flags: TF32=on, cudnn.benchmark={torch.backends.cudnn.benchmark}, "
            f"matmul_precision=high"
        )
        try:
            dev_name = torch.cuda.get_device_name(device)
            cap = torch.cuda.get_device_capability(device)
            logger.info(f"CUDA device: {dev_name} (cc {cap[0]}.{cap[1]})")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # AMP (mixed precision). Default: ON for CUDA, OFF for CPU.
    # ------------------------------------------------------------------
    amp_enabled = is_cuda and (not args.no_amp)
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
    logger.info(
        f"AMP: enabled={amp_enabled} dtype={args.amp_dtype}"
        + ("" if amp_enabled else " (set --no_amp to keep disabled)")
    )

    # Setup run name and output directory
    if args.run_name:
        run_name = args.run_name
    else:
        # Generate structured default name for easier run browsing in wandb.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"decomp_{args.backbone}_kfold{args.kfold}_{timestamp}"
    
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Run name: {run_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = HParams.load(args.config)
    
    # Keep model input size aligned with preprocessing bins to avoid projection mismatch.
    feature_n_bins = config.feature.get('n_bins', None) if hasattr(config, 'feature') else None
    model_feature_size = config.model.get('feature_size', None) if hasattr(config, 'model') else None
    if feature_n_bins is not None and model_feature_size != feature_n_bins:
        logger.warning(
            f"Config mismatch detected: model.feature_size={model_feature_size} "
            f"!= feature.n_bins={feature_n_bins}. Overriding model.feature_size to {feature_n_bins}."
        )
        config.model['feature_size'] = feature_n_bins

    # Resolve GradNorm settings (CLI override > config > defaults).
    gradnorm_cfg = config.get('gradnorm', {}) if hasattr(config, 'get') else {}
    if args.use_gradnorm:
        gradnorm_enabled = True
    elif args.no_gradnorm:
        gradnorm_enabled = False
    else:
        gradnorm_enabled = bool(gradnorm_cfg.get('enabled', False))

    gradnorm_alpha = float(args.gradnorm_alpha) if args.gradnorm_alpha is not None else float(gradnorm_cfg.get('alpha', 1.5))
    gradnorm_lr = float(args.gradnorm_lr) if args.gradnorm_lr is not None else float(gradnorm_cfg.get('lr', 0.025))
    gradnorm_eps = float(args.gradnorm_eps) if args.gradnorm_eps is not None else float(gradnorm_cfg.get('eps', 1e-8))
    gradnorm_w_min = float(args.gradnorm_w_min) if args.gradnorm_w_min is not None else float(gradnorm_cfg.get('w_min', 1e-3))
    gradnorm_w_max = float(args.gradnorm_w_max) if args.gradnorm_w_max is not None else float(gradnorm_cfg.get('w_max', 10.0))

    config.model['gradnorm_enabled'] = gradnorm_enabled
    config.model['gradnorm_alpha'] = gradnorm_alpha
    config.model['gradnorm_lr'] = gradnorm_lr
    config.model['gradnorm_eps'] = gradnorm_eps
    config.model['gradnorm_w_min'] = gradnorm_w_min
    config.model['gradnorm_w_max'] = gradnorm_w_max
    logger.info(
        "GradNorm: enabled=%s alpha=%.4f lr=%.5f eps=%g w_min=%g w_max=%g",
        gradnorm_enabled,
        gradnorm_alpha,
        gradnorm_lr,
        gradnorm_eps,
        gradnorm_w_min,
        gradnorm_w_max,
    )

    # Resolve Focal Loss settings (CLI override > config > defaults).
    focal_cfg = config.get('focal', {}) if hasattr(config, 'get') else {}
    focal_gamma = float(args.focal_gamma) if args.focal_gamma is not None else float(focal_cfg.get('gamma', 0.0))
    config.model['focal_gamma'] = focal_gamma
    logger.info("Focal Loss: gamma=%.2f%s", focal_gamma, " (disabled)" if focal_gamma == 0.0 else "")

    # Resolve output-head FFN settings (CLI override > config).
    if args.use_head_ffn:
        config.model['use_head_ffn'] = True
    if args.head_ffn_dim is not None:
        config.model['head_ffn_dim'] = args.head_ffn_dim
    logger.info(
        "Output head FFN: enabled=%s dim=%s",
        config.model.get('use_head_ffn', False),
        config.model.get('head_ffn_dim', 'hidden_size//2'),
    )
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    
    # Get data root and dataset names (CLI overrides config)
    data_root = config.experiment.get('data_root', config.path.get('root_path', '/data/music/chord_recognition'))
    dataset_names = args.train_datasets if args.train_datasets else config.experiment.get('dataset_names', ['billboard'])
    
    normalization = None
    if args.normalization:
        normalization = torch.load(args.normalization, weights_only=False)
        logger.info(f"Normalization: mean={normalization['mean']:.6f}, std={normalization['std']:.6f}  ({args.normalization})")
    else:
        logger.info("Normalization: disabled (raw log-CQT features)")

    logger.info(f"Data root: {data_root}")
    logger.info(f"Datasets: {dataset_names}")
    logger.info(f"K-Fold: {args.kfold}")
    logger.info(f"Split mode: {args.split_mode}")
    if component_weights is not None:
        logger.info(f"Component weights: {component_weights}")

    # Build datasets according to split mode.
    #   legacy: 2 datasets (train / val), train=True/False, split=None.
    #   paper : 3 datasets (train / val / test) with split=... and train=False.
    #           The `train` flag is ignored by AudioDataset when split is given;
    #           we keep train=False for clarity and to avoid downstream branches
    #           that key off `self.train`.
    test_dataset = None
    if args.split_mode == 'paper':
        train_dataset = AudioDatasetStructured(
            config,
            root_dir=data_root,
            dataset_names=tuple(dataset_names),
            train=False,
            decompose=True,
            kfold=args.kfold,
            split='train',
            normalization=normalization,
        )
        val_dataset = AudioDatasetStructured(
            config,
            root_dir=data_root,
            dataset_names=tuple(dataset_names),
            train=False,
            decompose=True,
            kfold=args.kfold,
            split='val',
            normalization=normalization,
        )
        test_dataset = AudioDatasetStructured(
            config,
            root_dir=data_root,
            dataset_names=tuple(dataset_names),
            train=False,
            decompose=True,
            kfold=args.kfold,
            split='test',
            normalization=normalization,
        )
    else:
        train_dataset = AudioDatasetStructured(
            config,
            root_dir=data_root,
            dataset_names=tuple(dataset_names),
            train=True,
            decompose=True,
            kfold=args.kfold,
            normalization=normalization,
        )
        val_dataset = AudioDatasetStructured(
            config,
            root_dir=data_root,
            dataset_names=tuple(dataset_names),
            train=False,
            decompose=True,
            kfold=args.kfold,
            normalization=normalization,
        )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    if test_dataset is not None:
        logger.info(f"Test samples: {len(test_dataset)}")

        # Persist the test path list so post-training evaluation does not need
        # to rebuild the dataset (paper-faithful workflow: train selects on val,
        # test is only consumed downstream).
        test_paths_file = output_dir / f"test_paths_kfold{args.kfold}.txt"
        with open(test_paths_file, "w", encoding="utf-8") as f:
            for p in test_dataset.paths:
                f.write(f"{p}\n")
        logger.info(f"Saved test paths -> {test_paths_file}")
    
    # Final guardrail: infer actual feature width from data and keep model in sync.
    if len(train_dataset) > 0:
        sample_feature_size = int(train_dataset[0]['feature'].shape[-1])
        if config.model.get('feature_size') != sample_feature_size:
            logger.warning(
                f"Feature width mismatch from data: model.feature_size={config.model.get('feature_size')} "
                f"but dataset provides {sample_feature_size}. Overriding model.feature_size to {sample_feature_size}."
            )
            config.model['feature_size'] = sample_feature_size
    
    # Create data loaders
    pin_memory = is_cuda and (not args.no_pin_memory)
    num_workers = max(0, int(args.num_workers))
    persistent_workers = (num_workers > 0) and (not args.no_persistent_workers)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = max(2, int(args.prefetch_factor))
    logger.info(
        f"DataLoader: num_workers={num_workers}, pin_memory={pin_memory}, "
        f"persistent_workers={persistent_workers}, "
        f"prefetch_factor={loader_kwargs.get('prefetch_factor', 'n/a')}"
    )

    if args.split_mode == 'paper':
        # Paper-faithful: one random segment per song per epoch (EpochSampler
        # already shuffles internally, so shuffle MUST be False here).
        train_sampler = EpochSampler(train_dataset)
        logger.info(
            f"EpochSampler enabled: {len(train_sampler)} songs -> "
            f"{len(train_sampler)} segments/epoch"
        )
        train_loader = AudioDataLoaderStructured(
            train_dataset,
            sampler=train_sampler,
            shuffle=False,
            **loader_kwargs,
        )
    else:
        train_loader = AudioDataLoaderStructured(
            train_dataset,
            shuffle=True,
            **loader_kwargs,
        )

    val_loader = AudioDataLoaderStructured(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    
    # Resolve class weighting mode (CLI override > config file)
    config_class_weights_enabled = config.class_weights.get('enabled', True) if hasattr(config, 'class_weights') else True
    if args.use_class_weights:
        class_weights_enabled = True
    elif args.no_class_weights:
        class_weights_enabled = False
    else:
        class_weights_enabled = config_class_weights_enabled
    
    class_weights = None
    class_weights_source = 'disabled'
    class_weights_file = None
    if class_weights_enabled:
        cache_path = _build_class_weights_cache_path(
            cache_dir=args.class_weights_cache_dir,
            dataset_names=dataset_names,
            kfold=args.kfold,
            gamma=args.gamma,
            w_max=args.w_max,
            train_paths=train_dataset.paths,
        )
        explicit_path = Path(args.class_weights_path) if args.class_weights_path else None

        if args.class_weights_mode == 'load':
            load_path = explicit_path if explicit_path else cache_path
            if not load_path.exists():
                raise FileNotFoundError(
                    f"class_weights_mode=load but file was not found: {load_path}\n"
                    f"Provide --class_weights_path or precompute cache first."
                )
            logger.info(f"Loading class weights from {load_path}")
            class_weights, _ = _load_class_weights_file(load_path, device=device)
            class_weights_source = 'load'
            class_weights_file = str(load_path)
        elif args.class_weights_mode == 'auto':
            load_path = explicit_path if explicit_path else cache_path
            if load_path.exists():
                logger.info(f"Loading cached class weights from {load_path}")
                class_weights, _ = _load_class_weights_file(load_path, device=device)
                class_weights_source = 'cache'
                class_weights_file = str(load_path)
            else:
                logger.info("Cached class weights not found; computing class weights...")
                class_weights, class_counts = MultiTaskLoss.compute_class_weights(
                    train_dataset,
                    gamma=args.gamma,
                    w_max=args.w_max,
                    device=device,
                    return_counts=True,
                )
                class_weights_source = 'compute'
                class_weights_file = str(load_path)
                load_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    'class_weights': _to_serializable_component_weights(class_weights),
                    'class_counts': {
                        c: torch.tensor(v, dtype=torch.float32) for c, v in class_counts.items()
                    },
                    'meta': {
                        'kfold': int(args.kfold),
                        'gamma': float(args.gamma),
                        'w_max': float(args.w_max),
                        'dataset_names': list(dataset_names),
                        'n_train_samples': int(len(train_dataset)),
                    },
                }
                torch.save(payload, load_path)
                logger.info(f"Saved class weights cache to {load_path}")
        else:
            logger.info("Computing class weights (compute mode)...")
            class_weights, class_counts = MultiTaskLoss.compute_class_weights(
                train_dataset,
                gamma=args.gamma,
                w_max=args.w_max,
                device=device,
                return_counts=True,
            )
            class_weights_source = 'compute'
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'class_weights': _to_serializable_component_weights(class_weights),
                'class_counts': {
                    c: torch.tensor(v, dtype=torch.float32) for c, v in class_counts.items()
                },
                'meta': {
                    'kfold': int(args.kfold),
                    'gamma': float(args.gamma),
                    'w_max': float(args.w_max),
                    'dataset_names': list(dataset_names),
                    'n_train_samples': int(len(train_dataset)),
                },
            }
            torch.save(payload, cache_path)
            class_weights_file = str(cache_path)
            logger.info(f"Saved class weights cache to {cache_path}")

        logger.info(f"Class weights source: {class_weights_source}")
        logger.info("Class weights ready:")
        for component, weights in class_weights.items():
            logger.info(f"  {component}: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")
    else:
        logger.info("Class reweighting disabled (using unweighted CrossEntropy for all components)")

    # Initialize wandb if available/enabled
    wandb_run = None
    wandb_enabled = False
    if args.wandb_disabled:
        logger.info("wandb logging disabled by --wandb_disabled")
    elif not WANDB_AVAILABLE:
        logger.warning("wandb package not available. Install with: pip install wandb")
        if WANDB_IMPORT_DIAGNOSTIC:
            logger.warning(f"wandb import diagnostic: {WANDB_IMPORT_DIAGNOSTIC}")
    else:
        try:
            if WANDB_IMPORT_DIAGNOSTIC:
                logger.info(f"wandb import diagnostic: {WANDB_IMPORT_DIAGNOSTIC}")
            wandb_api_key = args.wandb_api_key or os.getenv("WANDB_API_KEY")
            wandb_entity = args.wandb_entity or os.getenv("WANDB_ENTITY")
            wandb_version = getattr(wandb, "__version__", "unknown")
            logger.info(f"Detected wandb version: {wandb_version}")

            if wandb_api_key:
                # Always export key to env so older SDKs can still pick it up.
                os.environ["WANDB_API_KEY"] = wandb_api_key
                if hasattr(wandb, "login"):
                    try:
                        wandb.login(key=wandb_api_key, relogin=True)
                    except TypeError:
                        # Older wandb versions may not support relogin kwarg.
                        wandb.login(key=wandb_api_key)
                else:
                    logger.warning(
                        "This wandb module does not expose wandb.login(); "
                        "using WANDB_API_KEY environment variable fallback."
                    )
            else:
                api_key_available = bool(os.getenv("WANDB_API_KEY"))
                api_obj = getattr(wandb, "api", None)
                if api_obj is not None and getattr(api_obj, "api_key", None):
                    api_key_available = True

                if not api_key_available:
                    logger.warning(
                        "No wandb API key provided via --wandb_api_key or WANDB_API_KEY. "
                        "Run 'wandb login' or pass the key in CLI/env."
                    )

            # Initialize run even if entity is None (wandb uses default account/workspace).
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=wandb_entity,
                name=run_name,
                config={
                    # Keep a tiny top-level run namespace in init config.
                    # Full hyperparameters are published later as hparams.*.
                    'run.name': run_name,
                    'run.backbone': args.backbone,
                    'run.kfold': args.kfold,
                    'run.output_dir': str(output_dir),
                    'run.device': str(device),
                },
                tags=[args.backbone, f"kfold{args.kfold}", "decomposed"],
            )

            wandb_enabled = True
            run_url = getattr(wandb_run, "url", None)
            if run_url:
                logger.info(f"wandb initialized: {run_url}")
            else:
                logger.info("wandb initialized successfully")

            # Explicit metric mapping prevents hidden step conflicts.
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("model/*", step_metric="epoch")

            if class_weights is not None:
                class_weight_stats = {}
                for component, weights in class_weights.items():
                    class_weight_stats[f"class_weights/{component}_min"] = float(weights.min().item())
                    class_weight_stats[f"class_weights/{component}_max"] = float(weights.max().item())
                    class_weight_stats[f"class_weights/{component}_mean"] = float(weights.mean().item())
                if class_weight_stats:
                    class_weight_stats['epoch'] = 0
                    wandb.log(class_weight_stats)
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")
            module_path = getattr(wandb, "__file__", "unknown")
            module_type = type(wandb).__name__
            logger.error(f"wandb module info: type={module_type}, path={module_path}")
            wandb_enabled = False
    
    # Initialize model
    logger.info("Initializing model...")
    if args.backbone == 'chordformer':
        model = ChordFormer_model_decomposed(
            config,
            class_weights=class_weights,
            component_weights=component_weights
        )
    else:
        model = BTC_model_decomposed(
            config,
            class_weights=class_weights,
            component_weights=component_weights
        )
    model = model.to(device)
    logger.info(f"Selected backbone: {args.backbone}")

    # ------------------------------------------------------------------
    # torch.compile (opt-in). GradNorm runs autograd.grad through
    # self.model.last_shared_features set as a module attribute during
    # forward, which can confuse the compiler; warn explicitly.
    # ------------------------------------------------------------------
    if args.compile:
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile not available in this PyTorch version; --compile ignored.")
        else:
            if gradnorm_enabled:
                logger.warning(
                    "torch.compile + GradNorm: secondary autograd.grad through "
                    "model.last_shared_features may not be supported by Inductor. "
                    "If you see compile failures, retry with --no_gradnorm."
                )
            logger.info(f"Compiling model with torch.compile(mode={args.compile_mode!r})...")
            try:
                model = torch.compile(model, mode=args.compile_mode)
            except Exception as exc:
                logger.error(f"torch.compile failed; running uncompiled. Error: {exc}")

    # AMP + fp16 + GradNorm: explicitly unsupported (double backward + GradScaler).
    if amp_enabled and amp_dtype == torch.float16 and gradnorm_enabled:
        logger.warning(
            "AMP fp16 + GradNorm is not currently wired (would need GradScaler around "
            "two separate backward passes). Falling back to bfloat16 for AMP."
        )
        amp_dtype = torch.bfloat16

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer and scheduler
    model_trainable_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and n != 'criterion.gradnorm_weights'
    ]

    # ------------------------------------------------------------------
    # Optimizer: CLI > YAML(experiment.optimizer) > 'adam' (ChordMax default).
    # Use AdamW to mirror the ChordFormer paper (Tabela 6).
    # ------------------------------------------------------------------
    exp_cfg = config.experiment if hasattr(config, 'experiment') else {}
    optimizer_name = (
        args.optimizer
        or (exp_cfg.get('optimizer', None) if hasattr(exp_cfg, 'get') else None)
        or 'adam'
    ).lower()
    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model_trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            model_trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    logger.info(
        f"Optimizer: {optimizer.__class__.__name__}(lr={args.learning_rate}, weight_decay={args.weight_decay})"
    )

    # ------------------------------------------------------------------
    # Scheduler: CLI > YAML(experiment.scheduler) > 'cosine' (ChordMax default).
    # ChordFormer paper uses ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6).
    # ------------------------------------------------------------------
    scheduler_name = (
        args.scheduler
        or (exp_cfg.get('scheduler', None) if hasattr(exp_cfg, 'get') else None)
        or 'cosine'
    ).lower()

    scheduler_factor = (
        args.scheduler_factor
        if args.scheduler_factor is not None
        else float(exp_cfg.get('scheduler_factor', 0.1)) if hasattr(exp_cfg, 'get') else 0.1
    )
    scheduler_patience = (
        args.scheduler_patience
        if args.scheduler_patience is not None
        else int(exp_cfg.get('scheduler_patience', 5)) if hasattr(exp_cfg, 'get') else 5
    )
    scheduler_min_lr = (
        args.scheduler_min_lr
        if args.scheduler_min_lr is not None
        else float(exp_cfg.get('scheduler_min_lr', 1e-6)) if hasattr(exp_cfg, 'get') else 1e-6
    )

    scheduler_t_max = int(config.experiment.get('scheduler_t_max', args.num_epochs))
    if scheduler_t_max <= 0:
        logger.warning(
            f"Invalid scheduler_t_max={scheduler_t_max}; falling back to num_epochs={args.num_epochs}."
        )
        scheduler_t_max = args.num_epochs

    if scheduler_name == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr,
        )
        logger.info(
            f"Scheduler: ReduceLROnPlateau(factor={scheduler_factor}, "
            f"patience={scheduler_patience}, min_lr={scheduler_min_lr}). "
            f"Training will early-stop when LR <= {scheduler_min_lr}."
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_t_max)
        logger.info(f"Scheduler: CosineAnnealingLR(T_max={scheduler_t_max})")

    # ------------------------------------------------------------------
    # CRF stage (CLI > YAML(crf.type) > 'none').
    # The CRF itself is trained separately by train_harmonic_crf.py on top of
    # the frozen backbone; here we only resolve and persist the choice so the
    # checkpoint records which CRF variant the user intends to use afterwards.
    # ------------------------------------------------------------------
    crf_cfg = config.get('crf', {}) if hasattr(config, 'get') else {}
    crf_choice = (
        args.crf
        or (crf_cfg.get('type', None) if hasattr(crf_cfg, 'get') else None)
        or 'none'
    ).lower()
    if crf_choice not in {'none', 'trainable', 'linear'}:
        logger.warning(f"Unknown crf.type='{crf_choice}' in config; defaulting to 'none'.")
        crf_choice = 'none'
    crf_lambda = float(
        args.crf_lambda
        if args.crf_lambda is not None
        else (crf_cfg.get('lambda', 30.0) if hasattr(crf_cfg, 'get') else 30.0)
    )
    logger.info(
        f"CRF stage (post-training): type={crf_choice}"
        + (f", lambda={crf_lambda}" if crf_choice == 'linear' else "")
    )
    
    # Setup trainer and inference
    trainer = DecomposedChordTrainer(
        model,
        device=device,
        verbose=True,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    inference = DecomposedChordInference(model, device=device)
    metrics_fn = ChordMetrics()
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    best_epoch = 0
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    # Store training config for checkpoints
    training_config = {
        'run_name': run_name,
        'optimizer': optimizer.__class__.__name__,
        'scheduler': scheduler.__class__.__name__,
        'scheduler_factor': scheduler_factor,
        'scheduler_patience': scheduler_patience,
        'scheduler_min_lr': scheduler_min_lr,
        'crf_type': crf_choice,
        'crf_lambda': crf_lambda if crf_choice == 'linear' else None,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'gamma': args.gamma,
        'w_max': args.w_max,
        'scheduler_t_max': scheduler_t_max,
        'class_weights_enabled': class_weights_enabled,
        'class_weights_mode': args.class_weights_mode,
        'class_weights_source': class_weights_source,
        'class_weights_file': class_weights_file,
        'class_weights_cache_dir': args.class_weights_cache_dir,
        'backbone': args.backbone,
        'kfold': args.kfold,
        'component_weights': component_weights if component_weights is not None else 'default(all=1.0)',
        'gradnorm': {
            'enabled': gradnorm_enabled,
            'alpha': gradnorm_alpha,
            'lr': gradnorm_lr,
            'eps': gradnorm_eps,
            'w_min': gradnorm_w_min,
            'w_max': gradnorm_w_max,
        },
        'focal_gamma': focal_gamma,
        'model_config': {
            'hidden_size': config.model.get('hidden_size', 128),
            'num_layers': config.model.get('num_layers', 8),
            'num_heads': config.model.get('num_heads', 4),
            'feature_size': config.model.get('feature_size', 144),
            'timestep': config.model.get('timestep', 108),
            'input_dropout': config.model.get('input_dropout', 0.2),
            'layer_dropout': config.model.get('layer_dropout', 0.2),
            'attention_dropout': config.model.get('attention_dropout', 0.2),
            'conv_kernel_size': config.model.get('conv_kernel_size', 31),
            'ff_expansion_factor': config.model.get('ff_expansion_factor', 4),
            'conv_expansion_factor': config.model.get('conv_expansion_factor', 2),
            'gradnorm_enabled': config.model.get('gradnorm_enabled', False),
            'gradnorm_alpha': config.model.get('gradnorm_alpha', 1.5),
            'gradnorm_lr': config.model.get('gradnorm_lr', 0.025),
            'gradnorm_eps': config.model.get('gradnorm_eps', 1e-8),
            'gradnorm_w_min': config.model.get('gradnorm_w_min', 1e-3),
            'gradnorm_w_max': config.model.get('gradnorm_w_max', 10.0),
            'focal_gamma': config.model.get('focal_gamma', 0.0),
            'use_head_ffn': config.model.get('use_head_ffn', False),
            'head_ffn_dim': config.model.get('head_ffn_dim', None),
        },
        'datasets': list(dataset_names),
        'data_root': data_root,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'start_time': datetime.now().isoformat(),
        'output_dir': str(output_dir),
        'perf': {
            'amp_enabled': amp_enabled,
            'amp_dtype': args.amp_dtype if amp_enabled else None,
            'tf32': is_cuda,
            'cudnn_benchmark': bool(torch.backends.cudnn.benchmark) if is_cuda else False,
            'pin_memory': pin_memory,
            'num_workers': num_workers,
            'persistent_workers': persistent_workers,
            'prefetch_factor': loader_kwargs.get('prefetch_factor'),
            'compile': bool(args.compile),
            'compile_mode': args.compile_mode if args.compile else None,
        },
    }

    if wandb_enabled:
        # Publish full effective training hyperparameters to wandb config.
        # This makes sweeps/comparisons easier than inspecting checkpoint JSON.
        effective_hparams = {
            'training': training_config,
            'config': {
                'experiment': dict(config.experiment),
                'model': dict(config.model),
                'feature': dict(config.feature),
                'mp3': dict(config.mp3),
                'class_weights': dict(config.class_weights) if hasattr(config, 'class_weights') else {},
            },
            'runtime': {
                'device': str(device),
                'optimizer': optimizer.__class__.__name__,
                'scheduler': scheduler.__class__.__name__,
                'crf_type': crf_choice,
                'crf_lambda': crf_lambda if crf_choice == 'linear' else None,
            },
            'cli_args': safe_cli_args,
        }
        wandb.config.update(
            _flatten_dict_for_wandb(effective_hparams, prefix="hparams"),
            allow_val_change=True
        )
    
    # Track the most recent validation loss (used by ReduceLROnPlateau and the
    # LR-based early-stop condition; falls back to train_loss when validation
    # has not run yet in this epoch).
    last_val_loss = None
    early_stopped = False

    for epoch in range(args.num_epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{args.num_epochs} ===")

        # Train
        train_loss, component_losses = trainer.train_epoch(train_loader, optimizer)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Log component losses
        if component_losses:
            comp_str = " | ".join([f"{k[:4]}:{v:.3f}" for k, v in component_losses.items()])
            logger.info(f"  Components: {comp_str}")
            weighted = getattr(trainer, 'last_component_weighted_losses', None)
            weights_used = getattr(trainer, 'last_component_weights', None)
            if weighted:
                comp_str_w = " | ".join([f"{k[:4]}:{v:.3f}" for k, v in weighted.items()])
                logger.info(f"  Components(weighted): {comp_str_w}")
            if weights_used:
                comp_str_alpha = " | ".join([f"{k[:4]}:{v:.2f}" for k, v in weights_used.items()])
                logger.info(f"  Component weights: {comp_str_alpha}")
            if gradnorm_enabled:
                logger.info(f"  GradNorm loss: {trainer.last_gradnorm_loss:.4f}")
        
        training_history['train_loss'].append(train_loss)

        if wandb_enabled:
            train_log = {
                'train/loss': float(train_loss),
                'train/learning_rate': float(optimizer.param_groups[0]['lr']),
                'epoch': epoch + 1,
            }
            if component_losses:
                for name, value in component_losses.items():
                    train_log[f"train/components/{name}"] = float(value)

            weighted = getattr(trainer, 'last_component_weighted_losses', None)
            if weighted:
                for name, value in weighted.items():
                    train_log[f"train/components_weighted/{name}"] = float(value)

            weights_used = getattr(trainer, 'last_component_weights', None)
            if weights_used:
                for name, value in weights_used.items():
                    train_log[f"train/component_weights/{name}"] = float(value)
            if gradnorm_enabled:
                train_log['train/gradnorm/loss'] = float(trainer.last_gradnorm_loss)
                for name, value in trainer.last_gradnorm_inv_rate.items():
                    train_log[f"train/gradnorm/inv_rate/{name}"] = float(value)
                for name, value in trainer.last_gradnorm_grad_norm.items():
                    train_log[f"train/gradnorm/grad_norm/{name}"] = float(value)
                for name, value in trainer.last_gradnorm_target.items():
                    train_log[f"train/gradnorm/target/{name}"] = float(value)
            wandb.log(train_log)
        
        # Validate
        if (epoch + 1) % args.val_interval == 0:
            log_dist = getattr(args, 'log_class_distribution', True)
            val_metrics = trainer.validate(val_loader, compute_class_distribution=log_dist)
            val_loss = val_metrics['val_loss']
            last_val_loss = val_loss
            val_component_losses = val_metrics.get('component_losses', {})
            logger.info(f"Val Loss: {val_loss:.4f}")
            
            # Log validation component losses
            if val_component_losses:
                comp_str = " | ".join([f"{k[:4]}:{v:.3f}" for k, v in val_component_losses.items()])
                logger.info(f"  Val Components: {comp_str}")
                val_weighted = getattr(trainer, 'last_component_weighted_losses', None)
                val_weights_used = getattr(trainer, 'last_component_weights', None)
                if val_weighted:
                    comp_str_w = " | ".join([f"{k[:4]}:{v:.3f}" for k, v in val_weighted.items()])
                    logger.info(f"  Val Components(weighted): {comp_str_w}")
                if val_weights_used:
                    comp_str_alpha = " | ".join([f"{k[:4]}:{v:.2f}" for k, v in val_weights_used.items()])
                    logger.info(f"  Val Component weights: {comp_str_alpha}")

            # Log per-head class distributions
            class_dist = val_metrics.get('class_distribution')
            if class_dist:
                logger.info("  --- Per-head class distribution (val) ---")
                for comp, info in class_dist.items():
                    acc = info['accuracy'] * 100
                    names = info['class_names']
                    pred_pct = info['pred_pct']
                    tgt_pct = info['target_pct']
                    recall = info['per_class_recall']
                    pred_str = " ".join(f"{n}:{p:.0f}%" for n, p in zip(names, pred_pct) if p >= 0.5)
                    tgt_str = " ".join(f"{n}:{p:.0f}%" for n, p in zip(names, tgt_pct) if p >= 0.5)
                    recall_str = " ".join(f"{n}:{r*100:.0f}%" for i, (n, r) in enumerate(zip(names, recall)) if info['target_counts'][i] > 0)
                    logger.info(f"  [{comp:5s}] Acc:{acc:5.1f}% | Pred: {pred_str}")
                    logger.info(f"  {'':7s}          | GT:   {tgt_str}")
                    logger.info(f"  {'':7s}          | Recall: {recall_str}")

            training_history['val_loss'].append(val_loss)

            if wandb_enabled:
                current_best_loss = best_val_loss
                current_best_epoch = best_epoch
                if val_loss < current_best_loss:
                    current_best_loss = val_loss
                    current_best_epoch = epoch + 1

                val_log = {
                    'val/loss': float(val_loss),
                    'val/best_loss': float(current_best_loss),
                    'val/best_epoch': int(current_best_epoch),
                }
                if val_component_losses:
                    for name, value in val_component_losses.items():
                        val_log[f"val/components/{name}"] = float(value)

                val_weighted = getattr(trainer, 'last_component_weighted_losses', None)
                if val_weighted:
                    for name, value in val_weighted.items():
                        val_log[f"val/components_weighted/{name}"] = float(value)

                val_weights_used = getattr(trainer, 'last_component_weights', None)
                if val_weights_used:
                    for name, value in val_weights_used.items():
                        val_log[f"val/component_weights/{name}"] = float(value)

                if class_dist:
                    for comp, info in class_dist.items():
                        val_log[f"val/accuracy/{comp}"] = float(info['accuracy'])
                        names = info['class_names']
                        for i, cls_name in enumerate(names):
                            val_log[f"val/pred_dist/{comp}/{cls_name}"] = float(info['pred_pct'][i])
                            val_log[f"val/target_dist/{comp}/{cls_name}"] = float(info['target_pct'][i])
                            val_log[f"val/recall/{comp}/{cls_name}"] = float(info['per_class_recall'][i])

                    columns = ["component", "class", "pred_pct", "target_pct", "recall", "pred_count", "target_count"]
                    table_data = []
                    for comp, info in class_dist.items():
                        for i, cls_name in enumerate(info['class_names']):
                            table_data.append([
                                comp, cls_name,
                                round(float(info['pred_pct'][i]), 1),
                                round(float(info['target_pct'][i]), 1),
                                round(float(info['per_class_recall'][i]) * 100, 1),
                                int(info['pred_counts'][i]),
                                int(info['target_counts'][i]),
                            ])
                    val_log[f"val/class_distribution_table"] = wandb.Table(
                        columns=columns, data=table_data
                    )

                val_log['epoch'] = int(epoch + 1)
                wandb.log(val_log)
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                
                # Create descriptive checkpoint with all info
                checkpoint_data = {
                    # Epoch info
                    'epoch': epoch + 1,
                    'total_epochs': args.num_epochs,
                    
                    # Model state
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),

                    # Feature normalization (None when training without normalization)
                    'normalization': {
                        'mean': normalization['mean'],
                        'std': normalization['std'],
                    } if normalization is not None else None,
                    
                    # Metrics at save time
                    'metrics': {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_component_losses': {k: v for k, v in component_losses.items()} if component_losses else {},
                        'val_component_losses': {k: v for k, v in val_component_losses.items()} if val_component_losses else {},
                    },
                    
                    # Training configuration
                    'training_config': training_config,
                    
                    # Timestamp
                    'saved_at': datetime.now().isoformat(),
                }
                
                checkpoint_path = output_dir / f"model_best.pt"
                torch.save(checkpoint_data, checkpoint_path)
                logger.info(f"Saved best checkpoint to {checkpoint_path}")
                
                # Also save a human-readable summary
                summary_path = output_dir / f"model_best_info.json"
                summary = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'saved_at': datetime.now().isoformat(),
                    'training_config': training_config,
                }
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"Saved checkpoint info to {summary_path}")

                if wandb_enabled:
                    wandb.log({
                        'model/best_val_loss': float(best_val_loss),
                        'model/best_epoch': int(best_epoch),
                        'artifacts/best_checkpoint_path': str(checkpoint_path),
                        'epoch': int(epoch + 1),
                    })
        
        # Update learning rate
        # ReduceLROnPlateau needs a metric (val_loss); other schedulers don't.
        if isinstance(scheduler, ReduceLROnPlateau):
            metric = last_val_loss if last_val_loss is not None else train_loss
            scheduler.step(metric)
        else:
            scheduler.step()

        # LR-based early stop (mainly for ChordFormer's "until LR <= 1e-6").
        current_lr = optimizer.param_groups[0]['lr']
        if isinstance(scheduler, ReduceLROnPlateau) and current_lr <= scheduler_min_lr + 1e-12:
            logger.info(
                f"Early stop: optimizer LR ({current_lr:g}) reached scheduler_min_lr "
                f"({scheduler_min_lr:g}) after epoch {epoch + 1}."
            )
            early_stopped = True
            break

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"model_epoch_{epoch + 1:03d}.pt"
            torch.save({
                'epoch': epoch + 1,
                'total_epochs': args.num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'normalization': {
                    'mean': normalization['mean'],
                    'std': normalization['std'],
                } if normalization is not None else None,
                'metrics': {
                    'train_loss': train_loss,
                    'val_loss': val_loss if 'val_loss' in dir() else None,
                },
                'training_config': training_config,
                'saved_at': datetime.now().isoformat(),
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    training_config['end_time'] = datetime.now().isoformat()
    training_config['early_stopped'] = bool(early_stopped)
    final_path = output_dir / "model_final.pt"
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'normalization': {
            'mean': normalization['mean'],
            'std': normalization['std'],
        } if normalization is not None else None,
        'metrics': {
            'final_train_loss': training_history['train_loss'][-1] if training_history['train_loss'] else None,
            'final_val_loss': training_history['val_loss'][-1] if training_history['val_loss'] else None,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
        },
        'training_config': training_config,
        'saved_at': datetime.now().isoformat(),
    }, final_path)
    logger.info(f"Saved final model to {final_path}")
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")
    
    logger.info(f"\n=== Training Complete ===")
    logger.info(f"Best validation loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    logger.info(f"Checkpoints saved to: {output_dir}")

    if wandb_enabled:
        final_wandb_log = {
            'model/final_train_loss': float(training_history['train_loss'][-1]) if training_history['train_loss'] else None,
            'model/final_val_loss': float(training_history['val_loss'][-1]) if training_history['val_loss'] else None,
            'model/best_val_loss': float(best_val_loss),
            'model/best_epoch': int(best_epoch),
            'artifacts/final_model_path': str(final_path),
            'artifacts/history_path': str(history_path),
            'artifacts/output_dir': str(output_dir),
            'epoch': int(args.num_epochs),
        }
        wandb.log(final_wandb_log)

        wandb.run.summary['model/best_val_loss'] = float(best_val_loss)
        wandb.run.summary['model/best_epoch'] = int(best_epoch)
        wandb.run.summary['artifacts/output_dir'] = str(output_dir)
        wandb.finish()


if __name__ == '__main__':
    main()
