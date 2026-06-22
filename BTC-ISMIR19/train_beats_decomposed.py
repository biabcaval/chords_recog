#!/usr/bin/env python
# encoding: utf-8
"""
Train the BEATs chord decomposer on pre-extracted embeddings.

Mirrors ``train_decomposed.py`` conventions (config/CLI, optimizer, scheduler,
per-component CrossEntropy losses summed/weighted, gradient clipping at 1.0,
checkpointing, logging) but consumes frozen-BEATs embeddings instead of CQT
features and trains only the lightweight multi-head classifier.

Usage::

    # Pre-extract first (see scripts/preextract_beats_embeddings.py), then:
    python train_beats_decomposed.py --config run_config.yaml \
        --data_root /data --train_datasets billboard queen \
        --head_type mlp --kfold 4 --num_epochs 50

    # Self-contained smoke test (no data / no BEATs checkpoint required):
    python train_beats_decomposed.py --smoke_test
"""

import argparse
import importlib
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from models.beats_chord_model import (BEATsChordDecomposer, BEATS_EMBED_DIM,
                                       load_beats_backbone, set_beats_trainable)
from models.btc_model_decomposed import MultiTaskLoss
from data.beats_dataset import BEATsEmbeddingDataset, BEATsDataLoader
from data.beats_audio_dataset import BEATsAudioDataset, BEATsAudioDataLoader
from utils.decomposition_registry import get_decomposition, DECOMPOSITION_CHOICES
from utils.hparams import HParams

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Optional wandb integration with protection against local-module shadowing
# (mirrors train_decomposed.py so BEATs runs log to the same W&B account).
def _import_wandb_package():
    """Import the real wandb package even if a local ./wandb directory exists.

    Returns:
        tuple[module|None, bool, str]: (wandb_module, available, diagnostic).
    """
    try:
        mod = importlib.import_module("wandb")
        if hasattr(mod, "init"):
            return mod, True, ""
    except ImportError:
        return None, False, "wandb package is not installed."

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


def _flatten_dict_for_wandb(data, prefix=""):
    """Flatten nested dicts so hyperparameters are easy to filter in the W&B UI."""
    flat = {}
    for key, value in data.items():
        flat_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict_for_wandb(value, flat_key))
        else:
            flat[flat_key] = value
    return flat


def init_wandb(args, training_config, class_weights=None):
    """Initialize a W&B run for BEATs training, or return None when disabled.

    Failures never abort training: any problem logs a warning and returns None.
    """
    if args.wandb_disabled:
        logger.info("wandb logging disabled by --wandb_disabled")
        return None
    if not WANDB_AVAILABLE:
        logger.warning("wandb not available (%s). Skipping W&B logging.",
                       WANDB_IMPORT_DIAGNOSTIC or "package not installed")
        return None

    try:
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        api_key = args.wandb_api_key or os.getenv("WANDB_API_KEY")
        if api_key and hasattr(wandb, "login"):
            try:
                wandb.login(key=api_key, relogin=True)
            except TypeError:
                wandb.login(key=api_key)

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or os.getenv("WANDB_ENTITY"),
            name=training_config.get("run_name"),
            group=args.wandb_group,
            job_type="train_beats",
            config=_flatten_dict_for_wandb(training_config),
        )
        if class_weights:
            cw_stats = {}
            for comp, w in class_weights.items():
                cw_stats[f"class_weights/{comp}_min"] = float(w.min())
                cw_stats[f"class_weights/{comp}_max"] = float(w.max())
                cw_stats[f"class_weights/{comp}_mean"] = float(w.mean())
            wandb.log(cw_stats)
        logger.info("wandb run initialized: project=%s name=%s",
                    args.wandb_project, training_config.get("run_name"))
        return run
    except Exception as exc:  # noqa: BLE001 - logging must never break training
        logger.error("Failed to initialize wandb: %s", exc)
        return None


def parse_component_weights(spec, component_names):
    if spec is None:
        return None
    out = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid component weight item '{item}'. Expected key=value.")
        key, value = item.split("=", 1)
        key, value = key.strip(), value.strip()
        if key not in component_names:
            raise ValueError(f"Unknown component '{key}'. Valid: {component_names}")
        out[key] = float(value)
    for comp in component_names:
        out.setdefault(comp, 1.0)
    return out


def _batch_inputs(batch, device):
    """Return the model input tensor: raw waveforms (fine-tuning) or
    pre-extracted embeddings, whichever the collate produced."""
    inputs = batch["waveforms"] if "waveforms" in batch else batch["embeddings"]
    return inputs.to(device)


def train_one_epoch(model, loader, optimizer, device, component_names, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    comp_sums = {c: 0.0 for c in component_names}
    n_batches = 0
    for batch in loader:
        inputs = _batch_inputs(batch, device)
        labels = {c: batch["components"][c].to(device) for c in component_names}

        optimizer.zero_grad()
        _, loss, _, comp_losses = model(inputs, labels=labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        if comp_losses:
            for c, v in comp_losses.items():
                comp_sums[c] += v

    avg = total_loss / n_batches if n_batches else 0.0
    comp_avg = {c: (comp_sums[c] / n_batches if n_batches else 0.0) for c in component_names}
    return avg, comp_avg


@torch.no_grad()
def validate(model, loader, device, component_names):
    model.eval()
    total_loss = 0.0
    comp_sums = {c: 0.0 for c in component_names}
    correct = {c: 0 for c in component_names}
    seen = 0
    n_batches = 0
    for batch in loader:
        inputs = _batch_inputs(batch, device)
        labels = {c: batch["components"][c].to(device) for c in component_names}
        predictions, loss, _, comp_losses = model(inputs, labels=labels)
        total_loss += loss.item()
        n_batches += 1
        if comp_losses:
            for c, v in comp_losses.items():
                comp_sums[c] += v
        for c in component_names:
            p = predictions[c].reshape(-1)
            t = labels[c].reshape(-1)
            correct[c] += (p == t).sum().item()
        seen += labels[component_names[0]].reshape(-1).shape[0]

    avg = total_loss / n_batches if n_batches else 0.0
    comp_avg = {c: (comp_sums[c] / n_batches if n_batches else 0.0) for c in component_names}
    acc = {c: (correct[c] / seen if seen else 0.0) for c in component_names}
    return {"val_loss": avg, "component_losses": comp_avg, "accuracy": acc}


def _write_dummy_segment(path, n_patches=62, embed_dim=BEATS_EMBED_DIM):
    """Write one synthetic BEATs embedding .pt file with random labels."""
    rng = np.random.default_rng(abs(hash(path)) % (2 ** 32))
    roots = ["N", "C", "D", "E", "F", "G", "A", "B"]
    quals = ["maj", "min", "maj7", "7", "min7", "dim", "sus4"]
    labels = []
    for _ in range(n_patches):
        if rng.random() < 0.2:
            labels.append("N")
        else:
            r = roots[rng.integers(1, len(roots))]
            q = quals[rng.integers(0, len(quals))]
            labels.append(f"{r}:{q}")
    data = {
        "embedding": torch.randn(n_patches, embed_dim, dtype=torch.float32),
        "original_chord_labels": labels,
        "etc": "0.0_10.0",
        "patch_rate": n_patches / 10.0,
    }
    torch.save(data, path)


def run_smoke_test(args):
    """Validate heads, dataset collate, loss, and one training+val step with
    synthetic data only (no real audio, no BEATs checkpoint)."""
    decomposition = getattr(args, "decomposition", "paper6")
    logger.info("=== BEATs decomposer smoke test (synthetic data, scheme=%s) ===",
                decomposition)
    device = torch.device(args.device)
    decomp = get_decomposition(decomposition)
    component_names = list(decomp.COMPONENT_NAMES)
    chord_vocab = decomp.CHORD_VOCAB

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(8):
            # Vary patch counts to exercise padding in the collate fn.
            p = os.path.join(tmp, f"seg_{i}.pt")
            _write_dummy_segment(p, n_patches=60 + (i % 4))
            paths.append(p)

        dataset = BEATsEmbeddingDataset(paths=paths, decompose=True,
                                        decomposition=decomposition)
        loader = BEATsDataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

        sample = dataset[0]
        assert sample["embedding"].shape[1] == BEATS_EMBED_DIM
        assert "components" in sample
        logger.info("Dataset OK: %d samples, embedding %s, components=%d",
                    len(dataset), tuple(sample["embedding"].shape), len(sample["components"]))

        batch = next(iter(loader))
        assert batch["embeddings"].dim() == 3
        for c in component_names:
            assert batch["components"][c].shape == batch["embeddings"].shape[:2]
        logger.info("Collate OK: embeddings %s, lengths %s",
                    tuple(batch["embeddings"].shape), batch["lengths"].tolist())

        for head_type in ("linear", "mlp"):
            model = BEATsChordDecomposer(head_type=head_type, focal_gamma=args.focal_gamma,
                                         decomposition=decomposition).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            emb = batch["embeddings"].to(device)
            labels = {c: batch["components"][c].to(device) for c in component_names}
            logits = model.get_logits(emb)
            for c in component_names:
                b, p, v = logits[c].shape
                assert v == len(chord_vocab[c]), f"{c}: {v} != {len(chord_vocab[c])}"
            logger.info("[%s] logits shapes OK (e.g. %s=%s)", head_type,
                        component_names[0], tuple(logits[component_names[0]].shape))

            before = sum(pr.detach().abs().sum().item() for pr in model.parameters())
            train_loss, comp = train_one_epoch(model, loader, optimizer, device, component_names)
            after = sum(pr.detach().abs().sum().item() for pr in model.parameters())
            val = validate(model, loader, device, component_names)
            logger.info("[%s] train_loss=%.4f val_loss=%.4f params_changed=%s",
                        head_type, train_loss, val["val_loss"], before != after)
            assert np.isfinite(train_loss) and np.isfinite(val["val_loss"])
            assert before != after, "Parameters did not update during training step"

    logger.info("=== Smoke test PASSED ===")


def main():
    parser = argparse.ArgumentParser(description="Train BEATs chord decomposer on pre-extracted embeddings")
    parser.add_argument("--config", type=str, default="run_config.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--train_datasets", type=str, nargs="+", default=None)
    parser.add_argument("--kfold", type=int, default=4, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--beats_tag", type=str, default="beats_iter3_plus")
    parser.add_argument("--head_type", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--decomposition", type=str, default="paper6",
                        choices=list(DECOMPOSITION_CHOICES),
                        help="Chord decomposition scheme: 'paper6' (ChordFormer's 6 heads) "
                             "or 'full9' (project's 9 heads).")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Max epochs (cap). With --scheduler plateau, training "
                             "early-stops when LR <= scheduler_min_lr (paper criterion).")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay (paper uses AdamW ~0.01).")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # --- Optimizer / scheduler (aligned to the ChordFormer paper, Tabela 6) ---
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"],
                        help="Optimizer (paper: AdamW).")
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine"],
                        help="LR scheduler (paper: ReduceLROnPlateau, /10 after 5 stale epochs).")
    parser.add_argument("--scheduler_factor", type=float, default=0.1,
                        help="ReduceLROnPlateau factor (paper: 0.1 = divide LR by 10).")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="ReduceLROnPlateau patience in epochs (paper: 5).")
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-6,
                        help="Min LR; training early-stops at this value (paper: 1e-6).")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--w_max", type=float, default=10.0)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=0.0)
    parser.add_argument("--component_weights", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--wandb_project", type=str, default="chordMax-beats",
                        help="W&B project for BEATs runs (kept separate from the ChordFormer 'chordMax').")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_api_key", type=str, default=None,
                        help="W&B API key (falls back to WANDB_API_KEY env var).")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="Optional W&B group to cluster related BEATs runs.")
    parser.add_argument("--wandb_offline", action="store_true",
                        help="Log to W&B in offline mode (sync later with 'wandb sync').")
    parser.add_argument("--wandb_disabled", action="store_true",
                        help="Disable W&B logging entirely (local JSON still written).")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run a self-contained smoke test on synthetic data and exit.")
    # --- End-to-end fine-tuning of the BEATs backbone (Strategy A) ---
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune the top BEATs layers end-to-end on raw audio "
                             "(instead of training heads on pre-extracted embeddings). "
                             "Requires --beats_checkpoint (or config.beats).")
    parser.add_argument("--unfreeze_last_n", type=int, default=2,
                        help="With --finetune: number of top BEATs encoder layers to "
                             "unfreeze (default 2).")
    parser.add_argument("--backbone_lr", type=float, default=1e-5,
                        help="With --finetune: learning rate for the unfrozen backbone "
                             "layers (kept << head LR to avoid catastrophic forgetting).")
    parser.add_argument("--beats_checkpoint", type=str, default=None,
                        help="Path to a BEATs .pt checkpoint (defaults to config.beats.checkpoint_path).")
    parser.add_argument("--beats_source", type=str, default=None,
                        help="Path to the cloned unilm/beats dir (defaults to config.beats.source_path).")
    parser.add_argument("--audio_cache_size", type=int, default=8,
                        help="With --finetune: number of decoded songs kept in the "
                             "per-worker audio LRU cache.")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test(args)
        return

    if args.use_class_weights and args.no_class_weights:
        parser.error("Use only one of --use_class_weights or --no_class_weights")

    decomp = get_decomposition(args.decomposition)
    component_names = list(decomp.COMPONENT_NAMES)
    chord_vocab = decomp.CHORD_VOCAB
    logger.info("Decomposition scheme: %s (%d heads: %s)",
                decomp.scheme, len(component_names), component_names)

    component_weights = parse_component_weights(args.component_weights, component_names)
    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    config = HParams.load(args.config)
    data_root = args.data_root or config.experiment.get("data_root")
    dataset_names = args.train_datasets or config.experiment.get("dataset_names", ["billboard"])
    mp3_config = config.mp3
    mp3_str = "%d_%.1f_%.1f" % (mp3_config["song_hz"], mp3_config["inst_len"], mp3_config["skip_interval"])

    run_name = args.run_name or f"beats_{args.head_type}_kfold{args.kfold}_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run name: %s | Output: %s", run_name, output_dir)

    backbone = None
    target_patches = None
    if args.finetune:
        beats_cfg = config.beats if "beats" in config else {}
        ckpt = args.beats_checkpoint or beats_cfg.get("checkpoint_path")
        src = args.beats_source or beats_cfg.get("source_path")
        if not ckpt:
            parser.error("--finetune requires --beats_checkpoint (or config.beats.checkpoint_path).")
        logger.info("Fine-tuning mode: loading BEATs backbone for end-to-end training...")
        backbone = load_beats_backbone(ckpt, source_path=src, freeze=True, device=device)
        ft_info = set_beats_trainable(backbone, args.unfreeze_last_n)
        logger.info("Unfroze %d/%d top BEATs layers (indices %s); backbone trainable params: %d",
                    len(ft_info["unfrozen_layers"]), ft_info["n_total_layers"],
                    ft_info["unfrozen_layers"], ft_info["n_trainable_params"])

        # Determine the (fixed) patch count for one inst_len window via a single
        # backbone forward, so labels resample to align with backbone output.
        seg_samples = int(round(mp3_config["inst_len"] * 16000))
        with torch.no_grad():
            _dummy = torch.zeros(1, seg_samples, device=device)
            _pm = torch.zeros(1, seg_samples, dtype=torch.bool, device=device)
            target_patches = int(backbone.extract_features(_dummy, padding_mask=_pm)[0].shape[1])
        logger.info("BEATs patch count per %.1fs window: %d", mp3_config["inst_len"], target_patches)
        if args.batch_size > 8:
            logger.warning("Fine-tuning runs the full ~90M backbone in the autograd graph "
                           "(only the top %d layers receive gradients, but activations for "
                           "all layers are retained). batch_size=%d may OOM on most GPUs; "
                           "consider --batch_size 4-8.", args.unfreeze_last_n, args.batch_size)

        train_dataset = BEATsAudioDataset(
            config=config, data_root=data_root, dataset_names=tuple(dataset_names),
            train=True, kfold=args.kfold, target_patches=target_patches,
            decomposition=args.decomposition, audio_cache_size=args.audio_cache_size)
        val_dataset = BEATsAudioDataset(
            config=config, data_root=data_root, dataset_names=tuple(dataset_names),
            train=False, kfold=args.kfold, target_patches=target_patches,
            decomposition=args.decomposition, audio_cache_size=args.audio_cache_size)
        train_loader = BEATsAudioDataLoader(train_dataset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.num_workers)
        val_loader = BEATsAudioDataLoader(val_dataset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers)
    else:
        train_dataset = BEATsEmbeddingDataset(
            root_dir=data_root, dataset_names=tuple(dataset_names), train=True,
            kfold=args.kfold, beats_tag=args.beats_tag, mp3_string=mp3_str, decompose=True,
            decomposition=args.decomposition)
        val_dataset = BEATsEmbeddingDataset(
            root_dir=data_root, dataset_names=tuple(dataset_names), train=False,
            kfold=args.kfold, beats_tag=args.beats_tag, mp3_string=mp3_str, decompose=True,
            decomposition=args.decomposition)
        train_loader = BEATsDataLoader(train_dataset, batch_size=args.batch_size,
                                       shuffle=True, num_workers=args.num_workers)
        val_loader = BEATsDataLoader(val_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers)
    logger.info("Train samples: %d | Val samples: %d", len(train_dataset), len(val_dataset))

    config_cw_enabled = config.class_weights.get("enabled", True) if hasattr(config, "class_weights") else True
    if args.use_class_weights:
        cw_enabled = True
    elif args.no_class_weights:
        cw_enabled = False
    else:
        cw_enabled = config_cw_enabled

    class_weights = None
    if cw_enabled:
        logger.info("Computing class weights (gamma=%.2f, w_max=%.2f)...", args.gamma, args.w_max)
        # For the audio dataset, count labels without decoding audio.
        had_labels_only = getattr(train_dataset, "labels_only", None)
        if had_labels_only is not None:
            train_dataset.labels_only = True
        try:
            class_weights = MultiTaskLoss.compute_class_weights(
                train_dataset, gamma=args.gamma, w_max=args.w_max, device=device,
                component_names=component_names, chord_vocab=chord_vocab)
        finally:
            if had_labels_only is not None:
                train_dataset.labels_only = had_labels_only
        for component, weights in class_weights.items():
            logger.info("  %s: min=%.3f max=%.3f mean=%.3f",
                        component, weights.min(), weights.max(), weights.mean())
    else:
        logger.info("Class reweighting disabled.")

    model = BEATsChordDecomposer(
        input_dim=BEATS_EMBED_DIM, head_type=args.head_type, hidden_dim=args.hidden_dim,
        dropout=args.dropout, class_weights=class_weights,
        component_weights=component_weights, focal_gamma=args.focal_gamma,
        backbone=backbone, backbone_trainable=args.finetune,
        decomposition=args.decomposition).to(device)
    logger.info("Trainable params: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Optimizer (paper: AdamW). In fine-tuning mode use discriminative LRs: the
    # heads train at --learning_rate while the unfrozen backbone layers train at
    # the much smaller --backbone_lr (avoids wrecking pretrained features).
    optim_cls = optim.AdamW if args.optimizer == "adamw" else optim.Adam
    if args.finetune:
        backbone_params, head_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            (backbone_params if name.startswith("backbone.") else head_params).append(param)
        optimizer = optim_cls(
            [{"params": head_params, "lr": args.learning_rate},
             {"params": backbone_params, "lr": args.backbone_lr}],
            weight_decay=args.weight_decay)
        logger.info("Optimizer: %s(head_lr=%g, backbone_lr=%g, weight_decay=%g) | "
                    "head params=%d, backbone params=%d",
                    optimizer.__class__.__name__, args.learning_rate, args.backbone_lr,
                    args.weight_decay, sum(p.numel() for p in head_params),
                    sum(p.numel() for p in backbone_params))
    else:
        optimizer = optim_cls(model.parameters(), lr=args.learning_rate,
                              weight_decay=args.weight_decay)
        logger.info("Optimizer: %s(lr=%g, weight_decay=%g)",
                    optimizer.__class__.__name__, args.learning_rate, args.weight_decay)

    # Scheduler (paper: ReduceLROnPlateau /10 after 5 stale epochs, early-stop at min_lr).
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=args.scheduler_factor,
            patience=args.scheduler_patience, min_lr=args.scheduler_min_lr)
        logger.info("Scheduler: ReduceLROnPlateau(factor=%g, patience=%d, min_lr=%g); "
                    "early-stop when LR <= min_lr.",
                    args.scheduler_factor, args.scheduler_patience, args.scheduler_min_lr)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
        logger.info("Scheduler: CosineAnnealingLR(T_max=%d)", args.num_epochs)

    training_config = {
        "run_name": run_name,
        "head_type": args.head_type,
        "decomposition": decomp.scheme,
        "component_names": component_names,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "grad_clip": args.grad_clip,
        "gamma": args.gamma,
        "w_max": args.w_max,
        "focal_gamma": args.focal_gamma,
        "class_weights_enabled": cw_enabled,
        "component_weights": component_weights if component_weights is not None else "default(all=1.0)",
        "kfold": args.kfold,
        "beats_tag": args.beats_tag,
        "datasets": list(dataset_names),
        "data_root": data_root,
        "input_dim": BEATS_EMBED_DIM,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_min_lr": args.scheduler_min_lr,
        "finetune": args.finetune,
        "unfreeze_last_n": args.unfreeze_last_n if args.finetune else 0,
        "backbone_lr": args.backbone_lr if args.finetune else None,
        "target_patches": target_patches,
        "training_mode": "finetune_e2e" if args.finetune else "frozen_embeddings",
        "wandb_project": args.wandb_project,
        "start_time": datetime.now().isoformat(),
    }

    wandb_run = init_wandb(args, training_config, class_weights=class_weights)

    best_val_loss = float("inf")
    best_epoch = 0
    # Enriched history: keep flat train_loss/val_loss lists (consumed by the
    # _inventario_experimentos tooling) AND per-component curves + LR so the
    # overfitting signal can be traced per head, even offline (no W&B).
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "train_component_losses": {c: [] for c in component_names},
        "val_component_losses": {c: [] for c in component_names},
        "val_accuracy": {c: [] for c in component_names},
        "decomposition": decomp.scheme,
        "component_names": component_names,
    }

    logger.info("Starting training...")
    for epoch in range(args.num_epochs):
        logger.info("\n=== Epoch %d/%d ===", epoch + 1, args.num_epochs)
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, comp_losses = train_one_epoch(
            model, train_loader, optimizer, device, component_names, grad_clip=args.grad_clip)
        logger.info("Train Loss: %.4f (lr=%.2e)", train_loss, current_lr)
        comp_str = " | ".join(f"{k[:4]}:{v:.3f}" for k, v in comp_losses.items())
        logger.info("  Components: %s", comp_str)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["lr"].append(current_lr)
        for c in component_names:
            history["train_component_losses"][c].append(comp_losses.get(c, None))

        wandb_log = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/lr": current_lr,
        }
        for c in component_names:
            wandb_log[f"train/loss_{c}"] = comp_losses.get(c, None)

        val_loss = None
        if (epoch + 1) % args.val_interval == 0:
            val_metrics = validate(model, val_loader, device, component_names)
            val_loss = val_metrics["val_loss"]
            history["val_loss"].append(val_loss)
            for c in component_names:
                history["val_component_losses"][c].append(val_metrics["component_losses"].get(c, None))
                history["val_accuracy"][c].append(val_metrics["accuracy"].get(c, None))
            acc_str = " | ".join(f"{k[:4]}:{v * 100:.1f}%" for k, v in val_metrics["accuracy"].items())
            logger.info("Val Loss: %.4f", val_loss)
            logger.info("  Acc: %s", acc_str)

            wandb_log["val/loss"] = val_loss
            wandb_log["train_val_gap"] = train_loss - val_loss
            for c in component_names:
                wandb_log[f"val/loss_{c}"] = val_metrics["component_losses"].get(c, None)
                wandb_log[f"val/acc_{c}"] = val_metrics["accuracy"].get(c, None)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                ckpt = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "metrics": {"train_loss": train_loss, "val_loss": val_loss,
                                "val_accuracy": val_metrics["accuracy"]},
                    "training_config": training_config,
                    "saved_at": datetime.now().isoformat(),
                }
                torch.save(ckpt, output_dir / "model_best.pt")
                with open(output_dir / "model_best_info.json", "w") as f:
                    json.dump({"epoch": epoch + 1, "train_loss": train_loss,
                               "val_loss": val_loss,
                               "val_accuracy": val_metrics["accuracy"],
                               "training_config": training_config}, f, indent=2)
                logger.info("Saved best checkpoint (val_loss=%.4f)", val_loss)
                if wandb_run is not None:
                    wandb_log["model/best_val_loss"] = float(best_val_loss)
                    wandb_log["model/best_epoch"] = int(best_epoch)

        if wandb_run is not None:
            wandb.log(wandb_log)

        # ReduceLROnPlateau needs the monitored metric (val_loss); fall back to
        # train_loss on non-validation epochs. Other schedulers step blindly.
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss if val_loss is not None else train_loss)
            lr_now = optimizer.param_groups[0]["lr"]
            if lr_now <= args.scheduler_min_lr + 1e-12:
                logger.info("Early stop: LR (%.2e) reached scheduler_min_lr (%.2e) "
                            "after epoch %d (paper criterion).",
                            lr_now, args.scheduler_min_lr, epoch + 1)
                break
        else:
            scheduler.step()

    training_config["end_time"] = datetime.now().isoformat()
    history["best_val_loss"] = best_val_loss
    history["best_epoch"] = best_epoch

    torch.save({
        "epoch": args.num_epochs,
        "model_state_dict": model.state_dict(),
        "metrics": {
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "best_val_loss": best_val_loss, "best_epoch": best_epoch,
        },
        "training_config": training_config,
        "saved_at": datetime.now().isoformat(),
    }, output_dir / "model_final.pt")
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("\n=== Training Complete ===")
    logger.info("Best val loss: %.4f (Epoch %d)", best_val_loss, best_epoch)
    logger.info("Checkpoints saved to: %s", output_dir)

    if wandb_run is not None:
        try:
            wandb.run.summary["model/best_val_loss"] = float(best_val_loss)
            wandb.run.summary["model/best_epoch"] = int(best_epoch)
            wandb.run.summary["artifacts/output_dir"] = str(output_dir)
            wandb.finish()
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb finalization issue: %s", exc)


if __name__ == "__main__":
    main()
