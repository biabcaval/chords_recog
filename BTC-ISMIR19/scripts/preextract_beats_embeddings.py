#!/usr/bin/env python
# encoding: utf-8
"""
Offline pre-extraction of frozen-BEATs embeddings for chord recognition.

This is the BEATs analogue of ``Preprocess.generate_labels_features_voca``: it
segments each source song into ``inst_len``-second windows (``skip_interval``
hops), builds per-frame chord/root/quality/bass label lists exactly as the CQT
pipeline does, runs the frozen ~90M BEATs backbone ONCE per segment to obtain
frame-level embeddings ``(n_patches, 768)``, resamples the labels to the patch
rate, and writes one ``.pt`` per segment.

Training (``train_beats_decomposed.py``) then loads these embedding files and
trains only the lightweight classifier head -- the backbone is never run again.

Output layout (mirrors the CQT ``result_*_voca`` tree)::

    <data_root>/result_beats/<dataset>_beats/<mp3_string>/<beats_tag>/<song>/<aug>_<idx>.pt

Each ``.pt`` contains::

    {
      'embedding': FloatTensor (n_patches, 768),
      'chord': list[int]   (chord_id, patch rate),
      'root':  list[int],
      'quality': list[int],
      'bass': list[int],
      'original_chord_labels': list[str]  (patch rate),
      'etc': '<start>_<end>',
      'patch_rate': float (patches per second),
    }

Example (PowerShell)::

    python scripts/preextract_beats_embeddings.py `
        --config run_config.yaml `
        --data_root D:/datasets `
        --datasets billboard queen `
        --beats_checkpoint D:/BEATs/BEATs_iter3_plus_AS2M.pt `
        --beats_source D:/code/unilm/beats `
        --device cuda `
        --augment none

Storage caveat: enabling ``--augment full`` writes the 12-shift set
(``[-5..6]``), multiplying disk usage ~12x. Default ``--augment none`` keeps
only ``shift_factor=0``.
"""

import argparse
import os
import sys
import logging

import numpy as np
import torch
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.hparams import HParams
from utils.preprocess import Preprocess, FeatureTypes
from models.beats_chord_model import load_beats_backbone, extract_beats_embeddings, BEATS_EMBED_DIM
from data.beats_dataset import resample_sequence

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET_SR = 16000  # BEATs requires 16kHz mono input.

AUGMENT_SETS = {
    "none": [0],
    "full": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
}


def build_segment_labels(preprocessor, chord_info, inst_start_sec, shift_factor):
    """Build per-frame label lists for one window (CQT frame rate).

    Replicates the labelling logic of
    ``Preprocess.generate_labels_features_voca`` (chord_id/root/quality/bass +
    full label string), applying the pitch-shift transposition. Returns
    ``None`` if the window cannot produce a full-length label list.
    """
    time_interval = preprocessor.time_interval
    n_points = preprocessor.no_of_chord_datapoints_per_sequence
    inst_len = preprocessor.config.mp3["inst_len"]

    chord_list, root_list, quality_list, bass_list, label_list = [], [], [], [], []
    cur_sec = inst_start_sec
    while cur_sec < inst_start_sec + inst_len:
        available = chord_info.loc[(chord_info["start"] <= cur_sec) &
                                   (chord_info["end"] > cur_sec + time_interval)].copy()
        if len(available) == 0:
            available = chord_info.loc[
                ((chord_info["start"] >= cur_sec) & (chord_info["start"] <= cur_sec + time_interval)) |
                ((chord_info["end"] >= cur_sec) & (chord_info["end"] <= cur_sec + time_interval))
            ].copy()

        if len(available) == 1:
            chord = available["chord_id"].iloc[0]
            root = available["root"].iloc[0]
            quality = available["quality"].iloc[0]
            bass = available["bass"].iloc[0]
            chord_label = available["chord_label"].iloc[0]
        elif len(available) > 1:
            available["max_start"] = available.apply(lambda r: max(r["start"], cur_sec), axis=1)
            available["min_end"] = available.apply(lambda r: min(r.end, cur_sec + time_interval), axis=1)
            available["chord_length"] = available["min_end"] - available["max_start"]
            max_idx = available["chord_length"].idxmax()
            chord = available.loc[max_idx, "chord_id"]
            root = available.loc[max_idx, "root"]
            quality = available.loc[max_idx, "quality"]
            bass = available.loc[max_idx, "bass"]
            chord_label = available.loc[max_idx, "chord_label"]
        else:
            chord, root, quality, bass, chord_label = 169, 12, 14, 12, "N"

        if chord != 169 and chord != 168:
            chord = (chord + shift_factor * 14) % 168
        if root != 12:
            root = (root + shift_factor) % 12
        if bass != 12:
            bass = (bass + shift_factor) % 12
        chord_label = preprocessor.Chord_class.transpose_chord_label(chord_label, shift_factor)

        chord_list.append(int(chord))
        root_list.append(int(root))
        quality_list.append(int(quality))
        bass_list.append(int(bass))
        label_list.append(chord_label)
        cur_sec += time_interval

    if len(chord_list) != n_points:
        return None
    return chord_list, root_list, quality_list, bass_list, label_list


def process_song(preprocessor, backbone, song_name, lab_path, mp3_path, save_path,
                 shift_factors, beats_tag, mp3_str, device):
    mp3_config = preprocessor.config.mp3
    inst_len = mp3_config["inst_len"]
    skip_interval = mp3_config["skip_interval"]

    try:
        chord_info_orig = preprocessor.Chord_class.get_converted_chord_full(lab_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping '%s' (lab parse error: %s)", song_name, exc)
        return 0

    try:
        original_wav, _ = librosa.load(mp3_path, sr=TARGET_SR, mono=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping '%s' (audio load error: %s)", song_name, exc)
        return 0

    result_root = os.path.join(save_path + "_beats", mp3_str, beats_tag, song_name.strip())
    os.makedirs(result_root, exist_ok=True)

    total = 0
    for shift_factor in shift_factors:
        if shift_factor != 0:
            import pyrubberband as pyrb
            wav = pyrb.pitch_shift(original_wav, TARGET_SR, shift_factor)
        else:
            wav = original_wav

        chord_info = chord_info_orig  # stretch=1.0, so timing is unchanged.
        last_sec = chord_info.iloc[-1]["end"]
        last_sample = int(last_sec * TARGET_SR)
        if wav.shape[0] > last_sample:
            wav = wav[:last_sample]
        origin_len_sec = wav.shape[0] / TARGET_SR

        idx = 0
        current_start = 0.0
        while current_start + inst_len < origin_len_sec:
            labels = build_segment_labels(preprocessor, chord_info, current_start, shift_factor)
            if labels is None:
                current_start += skip_interval
                continue
            chord_list, root_list, quality_list, bass_list, label_list = labels

            etc = "%.1f_%.1f" % (current_start, current_start + inst_len)
            aug = "%.2f_%i" % (1.0, shift_factor)
            filename = "%s_%i.pt" % (aug, idx)
            output_file = os.path.join(result_root, filename)
            if os.path.exists(output_file):
                idx += 1
                total += 1
                current_start += skip_interval
                continue

            start_sample = int(current_start * TARGET_SR)
            end_sample = int((current_start + inst_len) * TARGET_SR)
            seg = wav[start_sample:end_sample]
            seg_tensor = torch.as_tensor(seg, dtype=torch.float32)

            embedding = extract_beats_embeddings(backbone, seg_tensor, device=device)
            n_patches = embedding.shape[0]
            seg_seconds = seg.shape[0] / TARGET_SR
            patch_rate = (n_patches / seg_seconds) if seg_seconds > 0 else 0.0

            result = {
                "embedding": embedding.contiguous(),
                "chord": resample_sequence(chord_list, n_patches),
                "root": resample_sequence(root_list, n_patches),
                "quality": resample_sequence(quality_list, n_patches),
                "bass": resample_sequence(bass_list, n_patches),
                "original_chord_labels": resample_sequence(label_list, n_patches),
                "etc": etc,
                "patch_rate": patch_rate,
            }
            torch.save(result, output_file)
            idx += 1
            total += 1
            current_start += skip_interval

    return total


def main():
    parser = argparse.ArgumentParser(description="Pre-extract frozen BEATs embeddings for chord recognition")
    parser.add_argument("--config", type=str, default="run_config.yaml")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Dataset root (overrides config.experiment.data_root).")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Datasets to process (overrides config).")
    parser.add_argument("--beats_checkpoint", type=str, default=None,
                        help="Path to BEATs checkpoint (e.g. BEATs_iter3_plus_AS2M.pt). "
                             "Defaults to config.beats['checkpoint_path'].")
    parser.add_argument("--beats_source", type=str, default=None,
                        help="Path to cloned unilm/beats dir containing BEATs.py. "
                             "Defaults to config.beats['source_path'].")
    parser.add_argument("--beats_tag", type=str, default="beats_iter3_plus",
                        help="Subfolder tag identifying this backbone/config.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--augment", type=str, default="none", choices=list(AUGMENT_SETS.keys()),
                        help="Pitch-shift augmentation set. 'none'=[0] (default, ~12x less disk), "
                             "'full'=[-5..6].")
    args = parser.parse_args()

    config = HParams.load(args.config)
    data_root = args.data_root or config.experiment.get("data_root")
    datasets = args.datasets or config.experiment.get("dataset_names", ["billboard"])
    shift_factors = AUGMENT_SETS[args.augment]

    # Resolve BEATs paths: CLI overrides config, otherwise fall back to run_config.yaml.
    beats_config = config.beats if "beats" in config else {}
    beats_checkpoint = args.beats_checkpoint or beats_config.get("checkpoint_path")
    beats_source = args.beats_source or beats_config.get("source_path")
    if not beats_checkpoint:
        parser.error("No BEATs checkpoint provided: pass --beats_checkpoint or set "
                     "beats.checkpoint_path in the config.")

    mp3_config = config.mp3
    mp3_str = "%d_%.1f_%.1f" % (mp3_config["song_hz"], mp3_config["inst_len"], mp3_config["skip_interval"])

    logger.info("Data root: %s", data_root)
    logger.info("Datasets: %s", datasets)
    logger.info("Shift factors: %s", shift_factors)
    logger.info("Loading frozen BEATs backbone...")
    backbone = load_beats_backbone(beats_checkpoint, source_path=beats_source,
                                   freeze=True, device=args.device)
    logger.info("Backbone loaded (embed_dim=%d, trainable params=%d).",
                BEATS_EMBED_DIM, sum(p.numel() for p in backbone.parameters() if p.requires_grad))

    preprocessor = Preprocess(config, FeatureTypes.cqt, tuple(datasets), data_root)
    all_files = preprocessor.get_all_files()
    logger.info("Found %d songs to process.", len(all_files))

    grand_total = 0
    for i, (song_name, lab_path, mp3_path, save_path) in enumerate(all_files, 1):
        logger.info("[%d/%d] %s", i, len(all_files), song_name)
        grand_total += process_song(preprocessor, backbone, song_name, lab_path,
                                    mp3_path, save_path, shift_factors, args.beats_tag,
                                    mp3_str, args.device)
    logger.info("Done. Total segments written: %d", grand_total)


if __name__ == "__main__":
    main()
