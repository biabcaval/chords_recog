#!/usr/bin/env python3
"""
Diagnose which chord classes appear/disappear during inference.

Given a trained checkpoint and an audio file (or dataset), runs inference
and reports per-component class statistics: which classes were predicted,
which were never predicted, and the softmax probability distributions.

Usage:
    # Single audio file
    python scripts/diagnose_inference_classes.py \\
        --checkpoint model_best.pt --audio_file song.mp3

    # Batch over dataset annotations (compares predictions with ground truth)
    python scripts/diagnose_inference_classes.py \\
        --checkpoint model_best.pt --lab_dir /path/to/annotations

    # Show top-k softmax probabilities per component
    python scripts/diagnose_inference_classes.py \\
        --checkpoint model_best.pt --audio_file song.mp3 --show_probs
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.chord_decomposition import (
    ChordDecomposer, ChordReassembler, COMPONENT_NAMES, CHORD_VOCAB
)
from utils.hparams import HParams


def load_model(checkpoint_path, config, device):
    from models.btc_model_decomposed import BTC_model_decomposed, ChordFormer_model_decomposed

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    tc = checkpoint.get("training_config", {})
    backbone = tc.get("backbone", "btc")
    model_cfg = tc.get("model_config", {})
    for key in ("use_head_ffn", "head_ffn_dim"):
        if key in model_cfg:
            config.model[key] = model_cfg[key]

    if backbone == "chordformer":
        model = ChordFormer_model_decomposed(config=config)
    else:
        model = BTC_model_decomposed(config=config)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    normalization = checkpoint.get("normalization", None)
    return model, normalization, backbone


def extract_features(audio_path, config, normalization=None):
    import librosa

    sr = config.mp3["song_hz"]
    hop_length = config.feature["hop_length"]
    n_bins = config.feature["n_bins"]
    bins_per_octave = config.feature["bins_per_octave"]

    y, sr = librosa.load(audio_path, sr=sr)
    cqt = librosa.cqt(y, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave, hop_length=hop_length)
    feature = np.log(np.abs(cqt) + 1e-6)

    if normalization is not None:
        feature = (feature - normalization["mean"]) / normalization["std"]

    return feature, sr, hop_length


def run_inference(feature, model, device, timestep=108, return_probs=False):
    """Run inference and return per-component predictions (and optionally probs)."""
    n_bins, n_frames = feature.shape
    all_predictions = {comp: [] for comp in COMPONENT_NAMES}
    all_probs = {comp: [] for comp in COMPONENT_NAMES} if return_probs else None

    for start in range(0, n_frames, timestep):
        end = min(start + timestep, n_frames)
        chunk = feature[:, start:end]

        if chunk.shape[1] < timestep:
            pad_width = timestep - chunk.shape[1]
            chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode="constant", constant_values=-6)

        chunk_tensor = torch.tensor(chunk.T, dtype=torch.float32).unsqueeze(0).to(device)
        actual_frames = min(end - start, timestep)

        with torch.no_grad():
            if return_probs:
                probs = model.predict_probabilities(chunk_tensor)
                for comp in COMPONENT_NAMES:
                    p = probs[comp][0, :actual_frames].cpu().numpy()
                    all_probs[comp].append(p)
                    all_predictions[comp].extend(np.argmax(p, axis=-1))
            else:
                output = model(chunk_tensor)
                predictions = output[0] if isinstance(output, tuple) else model.decomposer.get_predictions(output)
                for comp in COMPONENT_NAMES:
                    pred = predictions[comp][0, :actual_frames].cpu().numpy()
                    all_predictions[comp].extend(pred)

    for comp in COMPONENT_NAMES:
        all_predictions[comp] = np.array(all_predictions[comp])

    if return_probs:
        for comp in COMPONENT_NAMES:
            all_probs[comp] = np.concatenate(all_probs[comp], axis=0)

    return all_predictions, all_probs


def report_class_statistics(predictions, probs=None):
    """Print per-component class usage statistics."""
    reassembler = ChordReassembler()

    print("\n" + "=" * 70)
    print("PER-COMPONENT CLASS STATISTICS")
    print("=" * 70)

    n_frames = len(predictions[COMPONENT_NAMES[0]])

    for comp in COMPONENT_NAMES:
        vocab = CHORD_VOCAB[comp]
        pred = predictions[comp]
        counts = Counter(int(p) for p in pred)

        print(f"\n--- {comp} ({len(vocab)} classes) ---")

        predicted_classes = set()
        never_predicted = []

        for idx, label in enumerate(vocab):
            count = counts.get(idx, 0)
            pct = 100 * count / n_frames if n_frames > 0 else 0
            marker = "" if count > 0 else " << NEVER PREDICTED"
            print(f"  [{idx}] {label:6s}: {count:7d} frames ({pct:5.1f}%){marker}")

            if count > 0:
                predicted_classes.add(idx)
            else:
                never_predicted.append(label)

        if never_predicted:
            print(f"  >> Never predicted: {never_predicted}")

    # Softmax probability analysis
    if probs is not None:
        print("\n" + "=" * 70)
        print("SOFTMAX PROBABILITY ANALYSIS")
        print("=" * 70)
        print("(Max probability each class ever receives across all frames)")

        for comp in COMPONENT_NAMES:
            vocab = CHORD_VOCAB[comp]
            p = probs[comp]
            max_per_class = p.max(axis=0)
            mean_per_class = p.mean(axis=0)

            print(f"\n--- {comp} ---")
            for idx, label in enumerate(vocab):
                max_p = max_per_class[idx]
                mean_p = mean_per_class[idx]
                marker = ""
                if max_p < 0.05:
                    marker = " << MAX PROB < 5%"
                elif max_p < 0.10:
                    marker = " << MAX PROB < 10%"
                print(f"  [{idx}] {label:6s}: max_prob={max_p:.4f}  mean_prob={mean_p:.6f}{marker}")

    # Chord-level statistics
    print("\n" + "=" * 70)
    print("REASSEMBLED CHORD STATISTICS (top 20)")
    print("=" * 70)

    chords = []
    for i in range(n_frames):
        indices = {comp: int(predictions[comp][i]) for comp in COMPONENT_NAMES}
        chord = reassembler.reassemble_from_indices(indices)
        chords.append(chord)

    chord_counts = Counter(chords)
    for chord, count in chord_counts.most_common(20):
        pct = 100 * count / n_frames
        print(f"  {chord:25s}: {count:7d} frames ({pct:5.1f}%)")

    print(f"\n  Total unique chords: {len(chord_counts)}")
    n_count = chord_counts.get("N", 0)
    print(f"  'N' (no chord) frames: {n_count} ({100*n_count/n_frames:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Diagnose inference class coverage")
    parser.add_argument("--config", type=str, default="run_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio_file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--show_probs", action="store_true", help="Show softmax probability analysis")
    args = parser.parse_args()

    config = HParams.load(args.config)
    device = torch.device(args.device)

    print(f"Loading model from {args.checkpoint}")
    model, normalization, backbone = load_model(args.checkpoint, config, device)
    print(f"Backbone: {backbone}")

    if normalization:
        print(f"Normalization: mean={normalization['mean']:.6f}, std={normalization['std']:.6f}")

    print(f"\nExtracting features from {args.audio_file}")
    feature, sr, hop_length = extract_features(args.audio_file, config, normalization)
    print(f"Feature shape: {feature.shape} ({feature.shape[1]} frames)")

    print("Running inference...")
    predictions, probs = run_inference(
        feature, model, device,
        timestep=config.model["timestep"],
        return_probs=args.show_probs,
    )

    report_class_statistics(predictions, probs)


if __name__ == "__main__":
    main()
