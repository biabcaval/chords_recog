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


def _log_wandb_artifact_safe(run, file_path, artifact_name, artifact_type, aliases=None, metadata=None):
    """Log a single file as wandb artifact without interrupting training on failure."""
    if run is None:
        return
    try:
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type, metadata=metadata or {})
        artifact.add_file(str(file_path))
        run.log_artifact(artifact, aliases=aliases or [])
    except Exception as exc:
        logger.warning(f"Failed to log wandb artifact {artifact_name}: {exc}")


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
    parser.add_argument('--kfold', type=int, default=4, choices=[0, 1, 2, 3, 4],
                       help='5-fold split index used for validation (default: 4)')
    parser.add_argument('--component_weights', type=str, default=None,
                       help='Optional per-component loss weights as comma-separated key=value list')
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
    parser.add_argument('--wandb_api_key', type=str, default=None,
                       help='Weights & Biases API key (or set WANDB_API_KEY env var)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Weights & Biases entity (username or team)')
    parser.add_argument('--wandb_project', type=str, default='chordMax',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_disabled', action='store_true',
                       help='Disable Weights & Biases logging')
    
    args = parser.parse_args()
    
    if args.use_class_weights and args.no_class_weights:
        parser.error("Use only one of --use_class_weights or --no_class_weights")

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
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    
    # Get data root and dataset names from config
    data_root = config.experiment.get('data_root', config.path.get('root_path', '/data/music/chord_recognition'))
    dataset_names = config.experiment.get('dataset_names', ['billboard'])
    
    logger.info(f"Data root: {data_root}")
    logger.info(f"Datasets: {dataset_names}")
    logger.info(f"K-Fold: {args.kfold}")
    if component_weights is not None:
        logger.info(f"Component weights: {component_weights}")
    
    train_dataset = AudioDatasetStructured(
        config,
        root_dir=data_root,
        dataset_names=tuple(dataset_names),
        train=True,
        decompose=True,
        kfold=args.kfold
    )
    
    val_dataset = AudioDatasetStructured(
        config,
        root_dir=data_root,
        dataset_names=tuple(dataset_names),
        train=False,
        decompose=True,
        kfold=args.kfold
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
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
    train_loader = AudioDataLoaderStructured(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = AudioDataLoaderStructured(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
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

            if class_weights is not None:
                class_weight_stats = {}
                for component, weights in class_weights.items():
                    class_weight_stats[f"class_weights/{component}_min"] = float(weights.min().item())
                    class_weight_stats[f"class_weights/{component}_max"] = float(weights.max().item())
                    class_weight_stats[f"class_weights/{component}_mean"] = float(weights.mean().item())
                if class_weight_stats:
                    wandb.log(class_weight_stats, step=0)
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
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    # Alternative: use ReduceLROnPlateau for adaptive scheduling
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # Setup trainer and inference
    trainer = DecomposedChordTrainer(model, device=device, verbose=True)
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
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'gamma': args.gamma,
        'w_max': args.w_max,
        'class_weights_enabled': class_weights_enabled,
        'class_weights_mode': args.class_weights_mode,
        'class_weights_source': class_weights_source,
        'class_weights_file': class_weights_file,
        'class_weights_cache_dir': args.class_weights_cache_dir,
        'backbone': args.backbone,
        'kfold': args.kfold,
        'component_weights': component_weights if component_weights is not None else 'default(all=1.0)',
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
        },
        'datasets': dataset_names,
        'data_root': data_root,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'start_time': datetime.now().isoformat(),
        'output_dir': str(output_dir),
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
                'optimizer': 'Adam',
                'scheduler': 'CosineAnnealingLR',
            },
            'cli_args': safe_cli_args,
        }
        wandb.config.update(
            _flatten_dict_for_wandb(effective_hparams, prefix="hparams"),
            allow_val_change=True
        )
    
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
            wandb.log(train_log, step=epoch + 1)
        
        # Validate
        if (epoch + 1) % args.val_interval == 0:
            val_metrics = trainer.validate(val_loader)
            val_loss = val_metrics['val_loss']
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
                wandb.log(val_log, step=epoch + 1)
            
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
                    }, step=epoch + 1)
                    _log_wandb_artifact_safe(
                        wandb_run,
                        checkpoint_path,
                        artifact_name=f"{run_name}-model-best",
                        artifact_type="model",
                        aliases=["best", f"epoch-{epoch + 1:03d}"],
                        metadata={
                            'epoch': int(epoch + 1),
                            'val_loss': float(val_loss),
                            'kfold': int(args.kfold),
                            'backbone': args.backbone,
                        },
                    )
                    _log_wandb_artifact_safe(
                        wandb_run,
                        summary_path,
                        artifact_name=f"{run_name}-best-info",
                        artifact_type="metadata",
                        aliases=["best", f"epoch-{epoch + 1:03d}"],
                        metadata={
                            'epoch': int(epoch + 1),
                            'val_loss': float(val_loss),
                        },
                    )
        
        # Update learning rate
        scheduler.step()
        
        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"model_epoch_{epoch + 1:03d}.pt"
            torch.save({
                'epoch': epoch + 1,
                'total_epochs': args.num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
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
    final_path = output_dir / "model_final.pt"
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
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
        wandb.log({
            'model/final_train_loss': float(training_history['train_loss'][-1]) if training_history['train_loss'] else None,
            'model/final_val_loss': float(training_history['val_loss'][-1]) if training_history['val_loss'] else None,
            'model/best_val_loss': float(best_val_loss),
            'model/best_epoch': int(best_epoch),
            'artifacts/final_model_path': str(final_path),
            'artifacts/history_path': str(history_path),
            'artifacts/output_dir': str(output_dir),
        }, step=args.num_epochs)

        _log_wandb_artifact_safe(
            wandb_run,
            final_path,
            artifact_name=f"{run_name}-model-final",
            artifact_type="model",
            aliases=["final"],
            metadata={
                'epoch': int(args.num_epochs),
                'best_epoch': int(best_epoch),
                'best_val_loss': float(best_val_loss),
                'kfold': int(args.kfold),
                'backbone': args.backbone,
            },
        )
        _log_wandb_artifact_safe(
            wandb_run,
            history_path,
            artifact_name=f"{run_name}-training-history",
            artifact_type="metrics",
            aliases=["final"],
            metadata={
                'num_epochs': int(args.num_epochs),
                'kfold': int(args.kfold),
            },
        )

        wandb.run.summary['model/best_val_loss'] = float(best_val_loss)
        wandb.run.summary['model/best_epoch'] = int(best_epoch)
        wandb.run.summary['artifacts/output_dir'] = str(output_dir)
        wandb.finish()


if __name__ == '__main__':
    main()
