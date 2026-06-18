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
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.beats_chord_model import BEATsChordDecomposer, BEATS_EMBED_DIM
from models.btc_model_decomposed import MultiTaskLoss
from data.beats_dataset import BEATsEmbeddingDataset, BEATsDataLoader
from utils.chord_decomposition import COMPONENT_NAMES, CHORD_VOCAB
from utils.hparams import HParams

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_component_weights(spec):
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
        if key not in COMPONENT_NAMES:
            raise ValueError(f"Unknown component '{key}'. Valid: {COMPONENT_NAMES}")
        out[key] = float(value)
    for comp in COMPONENT_NAMES:
        out.setdefault(comp, 1.0)
    return out


def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    comp_sums = {c: 0.0 for c in COMPONENT_NAMES}
    n_batches = 0
    for batch in loader:
        embeddings = batch["embeddings"].to(device)
        labels = {c: batch["components"][c].to(device) for c in COMPONENT_NAMES}

        optimizer.zero_grad()
        _, loss, _, comp_losses = model(embeddings, labels=labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        if comp_losses:
            for c, v in comp_losses.items():
                comp_sums[c] += v

    avg = total_loss / n_batches if n_batches else 0.0
    comp_avg = {c: (comp_sums[c] / n_batches if n_batches else 0.0) for c in COMPONENT_NAMES}
    return avg, comp_avg


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    comp_sums = {c: 0.0 for c in COMPONENT_NAMES}
    correct = {c: 0 for c in COMPONENT_NAMES}
    seen = 0
    n_batches = 0
    for batch in loader:
        embeddings = batch["embeddings"].to(device)
        labels = {c: batch["components"][c].to(device) for c in COMPONENT_NAMES}
        predictions, loss, _, comp_losses = model(embeddings, labels=labels)
        total_loss += loss.item()
        n_batches += 1
        if comp_losses:
            for c, v in comp_losses.items():
                comp_sums[c] += v
        for c in COMPONENT_NAMES:
            p = predictions[c].reshape(-1)
            t = labels[c].reshape(-1)
            correct[c] += (p == t).sum().item()
        seen += labels[COMPONENT_NAMES[0]].reshape(-1).shape[0]

    avg = total_loss / n_batches if n_batches else 0.0
    comp_avg = {c: (comp_sums[c] / n_batches if n_batches else 0.0) for c in COMPONENT_NAMES}
    acc = {c: (correct[c] / seen if seen else 0.0) for c in COMPONENT_NAMES}
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
    logger.info("=== BEATs decomposer smoke test (synthetic data) ===")
    device = torch.device(args.device)

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(8):
            # Vary patch counts to exercise padding in the collate fn.
            p = os.path.join(tmp, f"seg_{i}.pt")
            _write_dummy_segment(p, n_patches=60 + (i % 4))
            paths.append(p)

        dataset = BEATsEmbeddingDataset(paths=paths, decompose=True)
        loader = BEATsDataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

        sample = dataset[0]
        assert sample["embedding"].shape[1] == BEATS_EMBED_DIM
        assert "components" in sample
        logger.info("Dataset OK: %d samples, embedding %s, components=%d",
                    len(dataset), tuple(sample["embedding"].shape), len(sample["components"]))

        batch = next(iter(loader))
        assert batch["embeddings"].dim() == 3
        for c in COMPONENT_NAMES:
            assert batch["components"][c].shape == batch["embeddings"].shape[:2]
        logger.info("Collate OK: embeddings %s, lengths %s",
                    tuple(batch["embeddings"].shape), batch["lengths"].tolist())

        for head_type in ("linear", "mlp"):
            model = BEATsChordDecomposer(head_type=head_type, focal_gamma=args.focal_gamma).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            emb = batch["embeddings"].to(device)
            labels = {c: batch["components"][c].to(device) for c in COMPONENT_NAMES}
            logits = model.get_logits(emb)
            for c in COMPONENT_NAMES:
                b, p, v = logits[c].shape
                assert v == len(CHORD_VOCAB[c]), f"{c}: {v} != {len(CHORD_VOCAB[c])}"
            logger.info("[%s] logits shapes OK (e.g. root=%s)", head_type, tuple(logits["root"].shape))

            before = sum(pr.detach().abs().sum().item() for pr in model.parameters())
            train_loss, comp = train_one_epoch(model, loader, optimizer, device)
            after = sum(pr.detach().abs().sum().item() for pr in model.parameters())
            val = validate(model, loader, device)
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
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--w_max", type=float, default=10.0)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=0.0)
    parser.add_argument("--component_weights", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run a self-contained smoke test on synthetic data and exit.")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test(args)
        return

    if args.use_class_weights and args.no_class_weights:
        parser.error("Use only one of --use_class_weights or --no_class_weights")

    component_weights = parse_component_weights(args.component_weights)
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

    train_dataset = BEATsEmbeddingDataset(
        root_dir=data_root, dataset_names=tuple(dataset_names), train=True,
        kfold=args.kfold, beats_tag=args.beats_tag, mp3_string=mp3_str, decompose=True)
    val_dataset = BEATsEmbeddingDataset(
        root_dir=data_root, dataset_names=tuple(dataset_names), train=False,
        kfold=args.kfold, beats_tag=args.beats_tag, mp3_string=mp3_str, decompose=True)
    logger.info("Train samples: %d | Val samples: %d", len(train_dataset), len(val_dataset))

    train_loader = BEATsDataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers)
    val_loader = BEATsDataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)

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
        class_weights = MultiTaskLoss.compute_class_weights(
            train_dataset, gamma=args.gamma, w_max=args.w_max, device=device)
        for component, weights in class_weights.items():
            logger.info("  %s: min=%.3f max=%.3f mean=%.3f",
                        component, weights.min(), weights.max(), weights.mean())
    else:
        logger.info("Class reweighting disabled.")

    model = BEATsChordDecomposer(
        input_dim=BEATS_EMBED_DIM, head_type=args.head_type, hidden_dim=args.hidden_dim,
        dropout=args.dropout, class_weights=class_weights,
        component_weights=component_weights, focal_gamma=args.focal_gamma).to(device)
    logger.info("Trainable params: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    training_config = {
        "run_name": run_name,
        "head_type": args.head_type,
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
        "start_time": datetime.now().isoformat(),
    }

    best_val_loss = float("inf")
    best_epoch = 0
    history = {"train_loss": [], "val_loss": []}

    logger.info("Starting training...")
    for epoch in range(args.num_epochs):
        logger.info("\n=== Epoch %d/%d ===", epoch + 1, args.num_epochs)
        train_loss, comp_losses = train_one_epoch(
            model, train_loader, optimizer, device, grad_clip=args.grad_clip)
        logger.info("Train Loss: %.4f", train_loss)
        comp_str = " | ".join(f"{k[:4]}:{v:.3f}" for k, v in comp_losses.items())
        logger.info("  Components: %s", comp_str)
        history["train_loss"].append(train_loss)

        val_loss = None
        if (epoch + 1) % args.val_interval == 0:
            val_metrics = validate(model, val_loader, device)
            val_loss = val_metrics["val_loss"]
            history["val_loss"].append(val_loss)
            acc_str = " | ".join(f"{k[:4]}:{v * 100:.1f}%" for k, v in val_metrics["accuracy"].items())
            logger.info("Val Loss: %.4f", val_loss)
            logger.info("  Acc: %s", acc_str)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                ckpt = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "metrics": {"train_loss": train_loss, "val_loss": val_loss},
                    "training_config": training_config,
                    "saved_at": datetime.now().isoformat(),
                }
                torch.save(ckpt, output_dir / "model_best.pt")
                with open(output_dir / "model_best_info.json", "w") as f:
                    json.dump({"epoch": epoch + 1, "train_loss": train_loss,
                               "val_loss": val_loss, "training_config": training_config}, f, indent=2)
                logger.info("Saved best checkpoint (val_loss=%.4f)", val_loss)

        scheduler.step()

    training_config["end_time"] = datetime.now().isoformat()
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


if __name__ == "__main__":
    main()
