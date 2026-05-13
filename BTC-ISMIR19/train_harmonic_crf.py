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
from models.harmonic_crf import (
    HarmonicCRF,
    FullChordCRF,
    CRF_MODE_CHOICES,
    CRF_KIND_CHOICES,
)
from data.audio_dataset_structured import (
    AudioDatasetStructured,
    AudioDataLoaderStructured,
)
from utils.chord_decomposition import COMPONENT_NAMES
from utils.chord_vocab_builder import (
    build_vocab_from_pt_files,
    validate_vocab,
)
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
    parser.add_argument('--crf_mode', type=str, default='root_triad',
                        choices=CRF_MODE_CHOICES,
                        help='CRF mode: root_triad (91 tags) or full (~2000 tags)')
    parser.add_argument('--crf_kind', type=str, default=None,
                        choices=CRF_KIND_CHOICES,
                        help="CRF transition matrix kind: 'trainable' (ChordMax default) "
                             "or 'linear' (ChordFormer: fixed lambda*I). "
                             "When omitted, falls back to crf.type in the YAML, then 'trainable'.")
    parser.add_argument('--crf_lambda', type=float, default=None,
                        help='Self-transition bonus when --crf_kind=linear (default 30, ChordFormer paper).')
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

    # ── CRF ──

    crf_mode = args.crf_mode
    logger.info(f"CRF mode: {crf_mode}")

    # Resolve transition-matrix kind: CLI > YAML(crf.type) > 'trainable'.
    crf_cfg = config.get('crf', {}) if hasattr(config, 'get') else {}
    crf_kind = (
        args.crf_kind
        or (crf_cfg.get('type', None) if hasattr(crf_cfg, 'get') else None)
        or 'trainable'
    ).lower()
    if crf_kind == 'none':
        # 'none' only makes sense for backbone-only runs; here we still need a
        # CRF to train, so warn and fall back to 'trainable'.
        logger.warning("crf.type='none' but train_harmonic_crf.py requires a CRF; using 'trainable'.")
        crf_kind = 'trainable'
    if crf_kind not in CRF_KIND_CHOICES:
        logger.warning(f"Unknown CRF kind '{crf_kind}'; defaulting to 'trainable'.")
        crf_kind = 'trainable'

    crf_lambda = float(
        args.crf_lambda
        if args.crf_lambda is not None
        else (crf_cfg.get('lambda', 30.0) if hasattr(crf_cfg, 'get') else 30.0)
    )
    logger.info(
        f"CRF kind: {crf_kind}"
        + (f" (lambda={crf_lambda})" if crf_kind == 'linear' else "")
    )

    chord_vocab = None
    chord_to_idx = None
    component_matrix = None

    if crf_mode == 'full':
        logger.info("Building full-chord vocabulary from training data...")
        chord_vocab, chord_to_idx, component_matrix = build_vocab_from_pt_files(
            data_root, dataset_names, config,
        )
        valid = validate_vocab(chord_vocab, component_matrix)
        if not valid:
            logger.error("Vocab validation failed — aborting")
            sys.exit(1)
        logger.info(f"Full-chord vocab: {len(chord_vocab)} tags "
                     f"(transition matrix: {len(chord_vocab)}x{len(chord_vocab)} "
                     f"= {len(chord_vocab)**2:,} params)")
        harmonic_crf = FullChordCRF(
            chord_vocab=chord_vocab,
            component_matrix=component_matrix,
            chord_to_idx=chord_to_idx,
            crf_kind=crf_kind,
            crf_lambda=crf_lambda,
        ).to(device)
    else:
        harmonic_crf = HarmonicCRF(
            n_roots=13, n_triads=7,
            crf_kind=crf_kind, crf_lambda=crf_lambda,
        ).to(device)

    crf_params = sum(p.numel() for p in harmonic_crf.parameters() if p.requires_grad)
    logger.info(f"CRF parameters (trainable): {crf_params:,}")
    if crf_kind == 'linear' and crf_params == 0:
        logger.warning(
            "LinearCRF has no trainable parameters; the optimizer step will be a no-op. "
            "Use this kind for inference-side comparison; loss values are still informative."
        )

    optimizer = optim.Adam(
        [p for p in harmonic_crf.parameters() if p.requires_grad] or [torch.zeros(1, requires_grad=True)],
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

            with torch.no_grad():
                crf_preds = harmonic_crf(logits)
                if crf_mode == 'full':
                    all_match = torch.ones_like(labels['root'], dtype=torch.bool)
                    for comp in COMPONENT_NAMES:
                        all_match &= (crf_preds[comp] == labels[comp])
                    train_correct += all_match.sum().item()
                else:
                    both_correct = (
                        (crf_preds['root'] == labels['root']) &
                        (crf_preds['triad'] == labels['triad'])
                    ).sum().item()
                    train_correct += both_correct
                train_total += labels['root'].numel()

        train_loss = np.mean(train_losses)
        train_acc = train_correct / train_total if train_total > 0 else 0

        # ── Validate ──
        harmonic_crf.eval()
        val_losses = []
        val_correct_per_comp = {c: 0 for c in COMPONENT_NAMES}
        val_correct_joint = 0
        val_total = 0

        val_argmax_correct_per_comp = {c: 0 for c in COMPONENT_NAMES}
        val_argmax_correct_joint = 0

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

                crf_preds = harmonic_crf(logits)

                crf_joint_match = torch.ones(batch_size, seq_len,
                                             dtype=torch.bool, device=device)
                argmax_joint_match = torch.ones(batch_size, seq_len,
                                                dtype=torch.bool, device=device)

                for comp in COMPONENT_NAMES:
                    crf_match = (crf_preds[comp] == labels[comp])
                    val_correct_per_comp[comp] += crf_match.sum().item()
                    crf_joint_match &= crf_match

                    argmax_pred = torch.argmax(logits[comp], dim=-1)
                    argmax_match = (argmax_pred == labels[comp])
                    val_argmax_correct_per_comp[comp] += argmax_match.sum().item()
                    argmax_joint_match &= argmax_match

                val_correct_joint += crf_joint_match.sum().item()
                val_argmax_correct_joint += argmax_joint_match.sum().item()
                val_total += labels['root'].numel()

        val_loss = np.mean(val_losses)
        val_acc_joint = val_correct_joint / val_total if val_total > 0 else 0
        argmax_acc_joint = val_argmax_correct_joint / val_total if val_total > 0 else 0

        # ── Logging ──

        logger.info(
            f"Epoch {epoch+1}/{args.num_epochs} | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )

        crf_parts = []
        argmax_parts = []
        delta_parts = []
        for comp in ['root', 'triad', 'bass', '7th']:
            crf_a = val_correct_per_comp[comp] / val_total if val_total else 0
            arg_a = val_argmax_correct_per_comp[comp] / val_total if val_total else 0
            crf_parts.append(f"{comp}:{crf_a:.3f}")
            argmax_parts.append(f"{comp}:{arg_a:.3f}")
            delta_parts.append(f"{comp}:{crf_a - arg_a:+.3f}")

        crf_parts.append(f"joint:{val_acc_joint:.3f}")
        argmax_parts.append(f"joint:{argmax_acc_joint:.3f}")
        delta_parts.append(f"joint:{val_acc_joint - argmax_acc_joint:+.3f}")

        logger.info(f"  CRF    {' | '.join(crf_parts)}")
        logger.info(f"  Argmax {' | '.join(argmax_parts)}")
        logger.info(f"  Delta  {' | '.join(delta_parts)}")

        # ── Checkpoint ──

        if val_acc_joint > best_val_acc:
            best_val_acc = val_acc_joint
            best_epoch = epoch + 1
            early_stop_counter = 0

            save_dict = {
                'harmonic_crf_state_dict': harmonic_crf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_val_acc': best_val_acc,
                'crf_mode': crf_mode,
                'chordformer_checkpoint': args.checkpoint,
                'config': {
                    'learning_rate': args.learning_rate,
                    'weight_decay': args.weight_decay,
                    'datasets': list(dataset_names),
                    'kfold': args.kfold,
                },
            }
            if crf_mode == 'full':
                save_dict['chord_vocab'] = chord_vocab
                save_dict['chord_to_idx'] = chord_to_idx
                save_dict['component_matrix'] = component_matrix
            else:
                save_dict['n_roots'] = harmonic_crf.n_roots
                save_dict['n_triads'] = harmonic_crf.n_triads

            save_path = output_dir / 'crf_best.pt'
            torch.save(save_dict, save_path)
            logger.info(f"  ** New best! acc={best_val_acc:.4f} saved to {save_path}")
        else:
            early_stop_counter += 1

        if val_acc_joint < prev_val_acc:
            for pg in optimizer.param_groups:
                old_lr = pg['lr']
                new_lr = max(old_lr * args.lr_decay_factor, args.lr_min)
                pg['lr'] = new_lr
            if new_lr != old_lr:
                logger.info(f"  LR decay: {old_lr:.6f} -> {new_lr:.6f}")
        prev_val_acc = val_acc_joint

        if early_stop_counter >= args.early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch+1} "
                        f"(no improvement for {args.early_stop_patience} epochs)")
            break

    # ── Summary ──

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"  CRF mode:       {crf_mode}")
    logger.info(f"  Best epoch:     {best_epoch}")
    logger.info(f"  Best val acc:   {best_val_acc:.4f}")
    if crf_mode == 'full':
        logger.info(f"  Vocab size:     {len(chord_vocab)} tags")
    logger.info(f"  Checkpoint:     {output_dir / 'crf_best.pt'}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
