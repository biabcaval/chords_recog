#!/usr/bin/env python
# encoding: utf-8
"""
Precompute class weights for decomposed chord training.

This script decouples class-weight computation from train_decomposed.py by
building and saving a reusable cache file.
"""

import argparse
import hashlib
import logging
from pathlib import Path

import torch

from data.audio_dataset_structured import AudioDatasetStructured
from models.btc_model_decomposed import MultiTaskLoss
from utils.hparams import HParams


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _hash_string_list(values):
    digest = hashlib.sha1()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:12]


def _build_cache_path(cache_dir, dataset_names, kfold, gamma, w_max, train_paths):
    datasets_sig = _hash_string_list(dataset_names)
    paths_sig = _hash_string_list(train_paths)
    filename = (
        f"class_weights_k{kfold}_g{gamma:.4f}_w{w_max:.4f}_"
        f"ds{datasets_sig}_p{paths_sig}.pt"
    )
    return Path(cache_dir) / filename


def _to_serializable_component_weights(weights_dict):
    serializable = {}
    for component, weights in weights_dict.items():
        if isinstance(weights, torch.Tensor):
            serializable[component] = weights.detach().cpu()
        else:
            serializable[component] = torch.tensor(weights, dtype=torch.float32)
    return serializable


def main():
    parser = argparse.ArgumentParser(description="Precompute class weights for decomposed model training.")
    parser.add_argument('--config', type=str, default='run_config.yaml', help='Path to configuration file')
    parser.add_argument('--kfold', type=int, default=4, choices=[0, 1, 2, 3, 4], help='K-fold index')
    parser.add_argument('--gamma', type=float, default=0.5, help='Class weighting gamma')
    parser.add_argument('--w_max', type=float, default=10.0, help='Class weighting cap')
    parser.add_argument('--cache_dir', type=str, default='./class_weights_cache', help='Directory to store cache file')
    parser.add_argument('--output_path', type=str, default=None, help='Optional explicit output file path')
    parser.add_argument('--device', type=str, default='cpu', help='Device for resulting tensors (cpu/cuda)')
    parser.add_argument('--force', action='store_true', help='Overwrite output if it already exists')
    args = parser.parse_args()

    logger.info(f"Loading config from {args.config}")
    config = HParams.load(args.config)

    data_root = config.experiment.get('data_root', config.path.get('root_path', '/data/music/chord_recognition'))
    dataset_names = config.experiment.get('dataset_names', ['billboard'])

    logger.info(f"Data root: {data_root}")
    logger.info(f"Datasets: {dataset_names}")
    logger.info(f"K-fold: {args.kfold}")

    train_dataset = AudioDatasetStructured(
        config,
        root_dir=data_root,
        dataset_names=tuple(dataset_names),
        train=True,
        decompose=True,
        kfold=args.kfold,
    )
    logger.info(f"Loaded training split with {len(train_dataset)} samples")

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = _build_cache_path(
            cache_dir=args.cache_dir,
            dataset_names=dataset_names,
            kfold=args.kfold,
            gamma=args.gamma,
            w_max=args.w_max,
            train_paths=train_dataset.paths,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.force:
        logger.info(f"Class-weight cache already exists: {output_path}")
        logger.info("Use --force to overwrite.")
        return

    device = torch.device(args.device)
    logger.info("Computing class weights...")
    class_weights, class_counts = MultiTaskLoss.compute_class_weights(
        train_dataset=train_dataset,
        gamma=args.gamma,
        w_max=args.w_max,
        device=device,
        return_counts=True,
    )

    payload = {
        'class_weights': _to_serializable_component_weights(class_weights),
        'class_counts': {
            component: torch.tensor(counts, dtype=torch.float32)
            for component, counts in class_counts.items()
        },
        'meta': {
            'kfold': int(args.kfold),
            'gamma': float(args.gamma),
            'w_max': float(args.w_max),
            'dataset_names': list(dataset_names),
            'data_root': str(data_root),
            'n_train_samples': int(len(train_dataset)),
            'output_path': str(output_path),
        },
    }
    torch.save(payload, output_path)

    logger.info(f"Saved class weights to: {output_path}")
    logger.info("Weight summary:")
    for component, weights in class_weights.items():
        logger.info(
            f"  {component}: min={weights.min().item():.3f}, "
            f"max={weights.max().item():.3f}, mean={weights.mean().item():.3f}"
        )


if __name__ == '__main__':
    main()
