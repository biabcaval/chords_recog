#!/usr/bin/env python
# encoding: utf-8
"""
Train the HarmonicCRF sequence decoder on top of a frozen ChordFormer.

This script loads a pre-trained ChordFormer checkpoint, freezes all its
parameters, and trains only the HarmonicCRF transition matrix (~8k params)
to learn plausible harmonic progressions from the model's own logits.

The CRF operates on root x triad (13 x 7 = 91 joint tags) and learns
which chord-to-chord transitions the model tends to produce, penalising
implausible jumps via Viterbi decoding at inference time.

Usage:
    python train_harmonic_crf.py \\
        --checkpoint checkpoints/my_run/model_best.pt \\
        --config run_config.yaml \\
        --train_datasets billboard queen robbiewilliams rwc jaah dj_avan_songbook2 \\
        --crf_run_name harmonic_crf_BiQuRoRwJaDj2 \\
        --num_epochs 50

    # Then use the CRF at inference:
    python run_inference_batch_decomposed.py \\
        --checkpoint checkpoints/my_run/model_best.pt \\
        --harmonic_crf checkpoints/harmonic_crf_BiQuRoRwJaDj2/crf_best.pt \\
        --test_dataset dj_avan_songbook1
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from models.btc_model_decomposed import (
    BTC_model_decomposed,
    ChordFormer_model_decomposed,
)
from models.harmonic_crf import HarmonicCRF
from data.audio_dataset_structured import (
    AudioDatasetStructured,
    AudioDataLoaderStructured,
)
from utils.chord_decomposition import COMPONENT_NAMES
from utils.hparams import HParams

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def _build_model(config, backbone='auto', checkpoint_meta=None):
    """Instantiate the correct decomposed model from checkpoint metadata."""
    if checkpoint_meta is not None:
        tc = checkpoint_meta.get('training_config', {})
        if backbone == 'auto':
            backbone = tc.get('backbone', 'btc')
        model_cfg = tc.get('model_config', {})
        for key in ('use_head_ffn', 'head_ffn_dim'):
            if key in model_cfg:
                config.model[key] = model_cfg[key]

    if backbone == 'chordformer':
        return ChordFormer_model_decomposed(config=config), backbone
    return BTC_model_decomposed(config=config), backbone


def main():
    parser = argparse.ArgumentParser(
        description='Train HarmonicCRF sequence decoder on frozen ChordFormer logits',
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained ChordFormer checkpoint (.pt)')
    parser.add_argument('--config', type=str, default='run_config.yaml',
                        help='Path to run_config.yaml')
    parser.add_argument('--backbone', type=str, default='auto',
                        choices=['auto', 'btc', 'chordformer'],
                        help='Model backbone (auto detects from checkpoint)')
    parser.add_argument('--train_datasets', type=str, nargs='+', default=None,
                        help='Datasets for training/validation')
    parser.add_argument('--kfold', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='K-fold split index for validation')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for CRF parameters (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--crf_run_name', type=str, default=None,
                        help='Name for this CRF training run')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Base directory to save CRF checkpoints')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device')
    parser.add_argument('--normalization', type=str, default=None,
                        help='Path to normalization .pt file (mean/std)')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Stop after N epochs without improvement')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5,
                        help='LR decay factor when accuracy drops')
    parser.add_argument('--lr_min', type=float, default=1e-5,
                        help='Minimum learning rate')

    args = parser.parse_args()
    device = torch.device(args.device)

    # ── Run name and output ──

    if args.crf_run_name:
        run_name = args.crf_run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"harmonic_crf_{timestamp}"

    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("HarmonicCRF Training")
    logger.info("=" * 60)
    logger.info(f"Run name:    {run_name}")
    logger.info(f"Checkpoint:  {args.checkpoint}")
    logger.info(f"Output dir:  {output_dir}")
    logger.info(f"Device:      {device}")

    # ── Load config ──

    config = HParams.load(args.config)
    config.feature['large_voca'] = True
    config.model['num_chords'] = 170

    feature_n_bins = config.feature.get('n_bins', None)
    if feature_n_bins is not None:
        config.model['feature_size'] = feature_n_bins

    # ── Load ChordFormer and freeze ──

    logger.info("Loading ChordFormer checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # The model must return logits (not argmax predictions)
    config.model['probs_out'] = True

    model, detected_backbone = _build_model(
        config, backbone=args.backbone, checkpoint_meta=checkpoint,
    )
    model = model.to(device)

    state_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'model'
    model.load_state_dict(checkpoint[state_key], strict=False)
    logger.info(f"Loaded ChordFormer (backbone: {detected_backbone})")

    # Freeze all ChordFormer parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    frozen_params = sum(p.numel() for p in model.parameters())
    logger.info(f"ChordFormer parameters (all frozen): {frozen_params:,}")

    # ── Normalization ──

    normalization = None
    if args.normalization:
        normalization = torch.load(args.normalization, weights_only=False)
        logger.info(f"Normalization: mean={normalization['mean']:.6f}, "
                     f"std={normalization['std']:.6f}")
    elif 'normalization' in checkpoint and checkpoint['normalization'] is not None:
        normalization = checkpoint['normalization']
        logger.info(f"Normalization from checkpoint: mean={normalization['mean']:.6f}, "
                     f"std={normalization['std']:.6f}")

    # ── Datasets ──

    data_root = config.experiment.get(
        'data_root', config.path.get('root_path', '')
    )
    dataset_names = (args.train_datasets
                     if args.train_datasets
                     else config.experiment.get('dataset_names', ['billboard']))

    logger.info(f"Data root:   {data_root}")
    logger.info(f"Datasets:    {dataset_names}")
    logger.info(f"K-Fold:      {args.kfold}")

    train_dataset = AudioDatasetStructured(
        config, root_dir=data_root,
        dataset_names=tuple(dataset_names),
        train=True, decompose=True, kfold=args.kfold,
        normalization=normalization,
    )
    val_dataset = AudioDatasetStructured(
        config, root_dir=data_root,
        dataset_names=tuple(dataset_names),
        train=False, decompose=True, kfold=args.kfold,
        normalization=normalization,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples:   {len(val_dataset)}")

    train_loader = AudioDataLoaderStructured(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
    )
    val_loader = AudioDataLoaderStructured(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
    )

    # ── HarmonicCRF ──

    harmonic_crf = HarmonicCRF(n_roots=13, n_triads=7).to(device)
    crf_params = sum(p.numel() for p in harmonic_crf.parameters())
    logger.info(f"HarmonicCRF parameters (trainable): {crf_params:,}")

    optimizer = optim.Adam(
        harmonic_crf.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # ── Training loop ──

    logger.info("=" * 60)
    logger.info("Starting CRF training...")
    logger.info("=" * 60)

    best_val_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0
    prev_val_acc = 0.0

    for epoch in range(args.num_epochs):
        # ── Train ──
        harmonic_crf.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            features = batch['features'].to(device)
            components = {
                comp: batch['components'][comp].to(device)
                for comp in COMPONENT_NAMES
            }

            batch_size = features.shape[0]
            seq_len = features.shape[3]

            labels = {}
            for comp in COMPONENT_NAMES:
                labels[comp] = components[comp].reshape(batch_size, seq_len)

            # Forward through frozen ChordFormer (returns logits because probs_out=True)
            with torch.no_grad():
                logits = model(features)

            # CRF loss (only CRF params have gradients)
            crf_loss = harmonic_crf.loss(logits, labels)

            optimizer.zero_grad()
            crf_loss.backward()
            optimizer.step()

            train_losses.append(crf_loss.item())

            # Accuracy: compare CRF-decoded root+triad vs GT
            with torch.no_grad():
                crf_preds = harmonic_crf(logits)
                root_correct = (crf_preds['root'] == labels['root']).sum().item()
                triad_correct = (crf_preds['triad'] == labels['triad']).sum().item()
                both_correct = (
                    (crf_preds['root'] == labels['root']) &
                    (crf_preds['triad'] == labels['triad'])
                ).sum().item()
                n = labels['root'].numel()
                train_correct += both_correct
                train_total += n

        train_loss = np.mean(train_losses)
        train_acc = train_correct / train_total if train_total > 0 else 0

        # ── Validate ──
        harmonic_crf.eval()
        val_losses = []
        val_correct_root = 0
        val_correct_triad = 0
        val_correct_both = 0
        val_total = 0

        # Also track argmax-only accuracy for comparison
        val_argmax_correct_root = 0
        val_argmax_correct_triad = 0
        val_argmax_correct_both = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                components = {
                    comp: batch['components'][comp].to(device)
                    for comp in COMPONENT_NAMES
                }

                batch_size = features.shape[0]
                seq_len = features.shape[3]

                labels = {}
                for comp in COMPONENT_NAMES:
                    labels[comp] = components[comp].reshape(batch_size, seq_len)

                logits = model(features)
                val_loss = harmonic_crf.loss(logits, labels)
                val_losses.append(val_loss.item())

                # CRF-decoded accuracy
                crf_preds = harmonic_crf(logits)
                val_correct_root += (crf_preds['root'] == labels['root']).sum().item()
                val_correct_triad += (crf_preds['triad'] == labels['triad']).sum().item()
                val_correct_both += (
                    (crf_preds['root'] == labels['root']) &
                    (crf_preds['triad'] == labels['triad'])
                ).sum().item()

                # Argmax baseline (what we'd get without CRF)
                argmax_root = torch.argmax(logits['root'], dim=-1)
                argmax_triad = torch.argmax(logits['triad'], dim=-1)
                val_argmax_correct_root += (argmax_root == labels['root']).sum().item()
                val_argmax_correct_triad += (argmax_triad == labels['triad']).sum().item()
                val_argmax_correct_both += (
                    (argmax_root == labels['root']) &
                    (argmax_triad == labels['triad'])
                ).sum().item()

                val_total += labels['root'].numel()

        val_loss = np.mean(val_losses)
        val_acc_root = val_correct_root / val_total if val_total > 0 else 0
        val_acc_triad = val_correct_triad / val_total if val_total > 0 else 0
        val_acc_both = val_correct_both / val_total if val_total > 0 else 0

        argmax_acc_root = val_argmax_correct_root / val_total if val_total > 0 else 0
        argmax_acc_triad = val_argmax_correct_triad / val_total if val_total > 0 else 0
        argmax_acc_both = val_argmax_correct_both / val_total if val_total > 0 else 0

        # ── Logging ──

        logger.info(
            f"Epoch {epoch+1}/{args.num_epochs} | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )
        logger.info(
            f"  CRF     root: {val_acc_root:.4f}  triad: {val_acc_triad:.4f}  "
            f"both: {val_acc_both:.4f}"
        )
        logger.info(
            f"  Argmax  root: {argmax_acc_root:.4f}  triad: {argmax_acc_triad:.4f}  "
            f"both: {argmax_acc_both:.4f}"
        )

        delta_root = val_acc_root - argmax_acc_root
        delta_triad = val_acc_triad - argmax_acc_triad
        delta_both = val_acc_both - argmax_acc_both
        logger.info(
            f"  Delta   root: {delta_root:+.4f}  triad: {delta_triad:+.4f}  "
            f"both: {delta_both:+.4f}"
        )

        # ── Checkpoint ──

        if val_acc_both > best_val_acc:
            best_val_acc = val_acc_both
            best_epoch = epoch + 1
            early_stop_counter = 0

            save_path = output_dir / 'crf_best.pt'
            torch.save({
                'harmonic_crf_state_dict': harmonic_crf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_val_acc': best_val_acc,
                'val_acc_root': val_acc_root,
                'val_acc_triad': val_acc_triad,
                'argmax_acc_root': argmax_acc_root,
                'argmax_acc_triad': argmax_acc_triad,
                'n_roots': harmonic_crf.n_roots,
                'n_triads': harmonic_crf.n_triads,
                'chordformer_checkpoint': args.checkpoint,
                'config': {
                    'learning_rate': args.learning_rate,
                    'weight_decay': args.weight_decay,
                    'datasets': list(dataset_names),
                    'kfold': args.kfold,
                },
            }, save_path)
            logger.info(f"  ** New best! acc={best_val_acc:.4f} saved to {save_path}")
        else:
            early_stop_counter += 1

        # LR decay when accuracy drops
        if val_acc_both < prev_val_acc:
            for pg in optimizer.param_groups:
                old_lr = pg['lr']
                new_lr = max(old_lr * args.lr_decay_factor, args.lr_min)
                pg['lr'] = new_lr
            if new_lr != old_lr:
                logger.info(f"  LR decay: {old_lr:.6f} -> {new_lr:.6f}")
        prev_val_acc = val_acc_both

        if early_stop_counter >= args.early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch+1} "
                        f"(no improvement for {args.early_stop_patience} epochs)")
            break

    # ── Summary ──

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"  Best epoch:     {best_epoch}")
    logger.info(f"  Best val acc:   {best_val_acc:.4f} (root+triad joint)")
    logger.info(f"  Checkpoint:     {output_dir / 'crf_best.pt'}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
