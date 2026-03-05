#!/usr/bin/env python
"""
Batch inference script for the decomposed chord recognition model (ChordFormer / BTC decomposed).

Generates .lab files from a trained checkpoint, which can then be evaluated
with generate_metrics_csv.py.

Usage:
    python run_inference_batch_decomposed.py \
        --checkpoint /path/to/model_best.pt \
        --audio_dir /path/to/dataset/audio \
        --output_dir ./inferences/exp_chordformer_test_Rw \
        --config run_config.yaml \
        --backbone chordformer

    # Or use --test_dataset to auto-resolve audio dir from data_root:
    python run_inference_batch_decomposed.py \
        --checkpoint /path/to/model_best.pt \
        --test_dataset rwc \
        --config run_config.yaml \
        --backbone chordformer
"""

import os
import sys
import argparse
import torch
import numpy as np
import librosa
from pathlib import Path
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.btc_model_decomposed import BTC_model_decomposed, ChordFormer_model_decomposed
from utils.chord_decomposition import ChordReassembler, COMPONENT_NAMES
from utils.hparams import HParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_AUDIO_SUBDIR = {
    "billboard": "billboard/audio",
    "jaah": "jaah/audio",
    "rwc": "rwc/audio",
    "dj_avan": "dj_avan/audio",
    "queen": "queen/audio",
    "robbiewilliams": "robbiewilliams/audio",
}

DATASET_SHORT = {
    "billboard": "Bi",
    "jaah": "Ja",
    "rwc": "Rw",
    "dj_avan": "Dj",
    "queen": "Qu",
    "robbiewilliams": "Ro",
}


def build_model(config, backbone="auto", checkpoint_meta=None):
    """Instantiate the correct decomposed model based on backbone choice."""
    if backbone == "auto" and checkpoint_meta is not None:
        tc = checkpoint_meta.get("training_config", {})
        backbone = tc.get("backbone", "btc")
        logger.info(f"Auto-detected backbone from checkpoint: {backbone}")

    if backbone == "chordformer":
        model = ChordFormer_model_decomposed(config=config)
    else:
        model = BTC_model_decomposed(config=config)

    logger.info(f"Using backbone: {backbone}")
    return model


def load_checkpoint(checkpoint_path, config, backbone, device):
    """Load model and normalization from a decomposed-training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_model(config, backbone=backbone, checkpoint_meta=checkpoint)
    model = model.to(device)

    state_key = "model_state_dict" if "model_state_dict" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_key], strict=False)
    model.eval()

    norm = checkpoint.get("normalization", None)
    if norm is not None:
        logger.info(f"Normalization from checkpoint: mean={norm['mean']:.6f}, std={norm['std']:.6f}")
    else:
        logger.info("No normalization found in checkpoint (raw log-CQT)")

    epoch = checkpoint.get("epoch", "?")
    logger.info(f"Loaded checkpoint {checkpoint_path}  (epoch {epoch})")
    return model, norm


def audio_file_to_features(audio_path, config, normalization=None):
    """Extract log-CQT features from an audio file, optionally normalized."""
    sr = config.mp3["song_hz"]
    inst_len = config.mp3["inst_len"]
    n_bins = config.feature["n_bins"]
    bpo = config.feature["bins_per_octave"]
    hop = config.feature["hop_length"]

    wav, _ = librosa.load(audio_path, sr=sr, mono=True)

    feature = None
    pos = 0
    chunk_samples = int(sr * inst_len)

    while len(wav) > pos + chunk_samples:
        chunk_cqt = librosa.cqt(
            wav[pos : pos + chunk_samples],
            sr=sr, n_bins=n_bins, bins_per_octave=bpo, hop_length=hop,
        )
        feature = chunk_cqt if feature is None else np.concatenate((feature, chunk_cqt), axis=1)
        pos += chunk_samples

    tail_cqt = librosa.cqt(
        wav[pos:], sr=sr, n_bins=n_bins, bins_per_octave=bpo, hop_length=hop,
    )
    feature = tail_cqt if feature is None else np.concatenate((feature, tail_cqt), axis=1)

    feature = np.log(np.abs(feature) + 1e-6)

    if normalization is not None:
        feature = (feature - normalization['mean']) / normalization['std']

    feature_per_second = inst_len / config.model["timestep"]
    song_length_second = len(wav) / sr
    return feature, feature_per_second, song_length_second


def get_audio_paths(audio_dir):
    """Collect all mp3/wav files under *audio_dir* (recursive)."""
    paths = []
    for root, _, files in os.walk(audio_dir, followlinks=True):
        for fname in sorted(files):
            if fname.lower().endswith((".wav", ".mp3")):
                paths.append(os.path.join(root, fname))
    return sorted(paths)


def run_inference(model, audio_dir, output_dir, config, device, normalization=None):
    """Run inference on every audio file and write .lab outputs."""
    reassembler = ChordReassembler()
    os.makedirs(output_dir, exist_ok=True)

    audio_paths = get_audio_paths(audio_dir)
    if not audio_paths:
        logger.warning(f"No audio files found in {audio_dir}")
        return

    logger.info(f"Found {len(audio_paths)} audio files in {audio_dir}")
    n_timestep = config.model["timestep"]

    for audio_path in audio_paths:
        song_name = Path(audio_path).stem
        try:
            logger.info(f"Processing: {song_name}")

            feature, feature_per_second, _ = audio_file_to_features(audio_path, config, normalization)
            feature = feature.T  # (time, bins)
            time_unit = feature_per_second

            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            if num_pad == n_timestep:
                num_pad = 0
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
            num_instance = feature.shape[0] // n_timestep

            lines = []
            start_time = 0.0
            prev_chord = None

            with torch.no_grad():
                feat_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)

                for t in range(num_instance):
                    segment = feat_tensor[:, n_timestep * t : n_timestep * (t + 1), :]
                    predictions, _, _, _ = model(segment)

                    for i in range(n_timestep):
                        global_frame = n_timestep * t + i

                        if t == num_instance - 1 and i + num_pad >= n_timestep:
                            break

                        indices = {
                            comp: int(predictions[comp][0, i].item())
                            for comp in COMPONENT_NAMES
                        }
                        chord = reassembler.reassemble_from_indices(indices)

                        if prev_chord is None:
                            prev_chord = chord
                            continue

                        if chord != prev_chord:
                            lines.append(
                                f"{start_time:.3f} {time_unit * global_frame:.3f} {prev_chord}\n"
                            )
                            start_time = time_unit * global_frame
                            prev_chord = chord

                if prev_chord is not None:
                    total_frames = num_instance * n_timestep - num_pad
                    end_time = time_unit * total_frames
                    if start_time < end_time:
                        lines.append(f"{start_time:.3f} {end_time:.3f} {prev_chord}\n")

            out_path = os.path.join(output_dir, song_name + ".lab")
            with open(out_path, "w") as f:
                f.writelines(lines)
            logger.info(f"  -> {out_path}  ({len(lines)} segments)")

        except Exception as e:
            logger.error(f"Error processing {song_name}: {e}", exc_info=True)
            continue

    logger.info(f"Inference complete. Results saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch inference with decomposed chord model (ChordFormer / BTC decomposed)"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config", type=str, default="run_config.yaml",
        help="Path to run_config.yaml",
    )
    parser.add_argument(
        "--backbone", type=str, default="chordformer",
        choices=["auto", "btc", "chordformer"],
        help="Backbone encoder (default: chordformer)",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--audio_dir", type=str, default=None,
        help="Directory containing audio files (mp3/wav) for inference",
    )
    group.add_argument(
        "--test_dataset", type=str, default=None,
        choices=list(DATASET_AUDIO_SUBDIR.keys()),
        help="Test dataset name — audio_dir is resolved from data_root in config",
    )

    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save .lab files (auto-generated if not provided)",
    )
    parser.add_argument(
        "--output_base", type=str, default="./inferences_decomposed",
        help="Base directory for auto-generated output folder names",
    )
    parser.add_argument(
        "--exp_name", type=str, default=None,
        help="Experiment name for the output folder (e.g. 'chordformer_BiJaRw')",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    config = HParams.load(args.config)
    config.feature["large_voca"] = True
    config.model["num_chords"] = 170

    model, normalization = load_checkpoint(args.checkpoint, config, args.backbone, device)

    if args.audio_dir:
        audio_dir = args.audio_dir
    else:
        data_root = config.experiment.get(
            "data_root", config.path.get("root_path", "")
        )
        subdir = DATASET_AUDIO_SUBDIR[args.test_dataset]
        audio_dir = os.path.join(data_root, subdir)
        logger.info(f"Resolved audio_dir from config: {audio_dir}")

    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        ds_tag = DATASET_SHORT.get(args.test_dataset, "unk") if args.test_dataset else "custom"
        exp = args.exp_name or Path(args.checkpoint).parent.name
        output_dir = os.path.join(args.output_base, f"inference_{exp}_test_{ds_tag}")

    logger.info(f"Checkpoint : {args.checkpoint}")
    logger.info(f"Audio dir  : {audio_dir}")
    logger.info(f"Output dir : {output_dir}")

    run_inference(model, audio_dir, output_dir, config, device, normalization)


if __name__ == "__main__":
    main()
