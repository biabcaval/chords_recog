#!/usr/bin/env python
# encoding: utf-8
"""
Full audio inference - processes entire audio file and shows all chord predictions.

Usage:
    python infer_full_audio.py --config run_config.yaml --checkpoint model_best.pt --audio_file song.mp3
    python infer_full_audio.py --config run_config.yaml --checkpoint model_best.pt --audio_file song.mp3 --backbone chordformer
    python infer_full_audio.py --config run_config.yaml --checkpoint model_best.pt --audio_file song.mp3 --output result.lab
"""

import argparse
import torch
import numpy as np
import librosa
from pathlib import Path
import logging

from models.btc_model_decomposed import BTC_model_decomposed, ChordFormer_model_decomposed
from utils.chord_decomposition import ChordReassembler, COMPONENT_NAMES
from utils.hparams import HParams


def _build_model(config, backbone='auto', checkpoint_meta=None):
    """
    Instantiate the correct decomposed model based on backbone choice.

    When backbone='auto', tries to detect from checkpoint metadata
    (training_config.backbone), falling back to 'btc'.
    """
    if backbone == 'auto' and checkpoint_meta is not None:
        tc = checkpoint_meta.get('training_config', {})
        backbone = tc.get('backbone', 'btc')

    if backbone == 'chordformer':
        return ChordFormer_model_decomposed(config=config)
    return BTC_model_decomposed(config=config)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_features_full(audio_path, config):
    """Extract CQT features from entire audio file."""
    sr = config.mp3['song_hz']
    hop_length = config.feature['hop_length']
    n_bins = config.feature['n_bins']
    bins_per_octave = config.feature['bins_per_octave']
    
    logger.info(f"Loading audio from {audio_path}")
    y, sr = librosa.load(audio_path, sr=sr)
    duration = len(y) / sr
    logger.info(f"Audio duration: {duration:.2f}s, Sample rate: {sr}")
    
    # Extract CQT
    logger.info("Extracting CQT features...")
    cqt = librosa.cqt(
        y, sr=sr,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length
    )
    
    # Log magnitude
    feature = np.log(np.abs(cqt) + 1e-6)
    logger.info(f"Feature shape: {feature.shape} (bins x frames)")
    
    return feature, sr, hop_length


def process_in_chunks(feature, model, device, timestep=108):
    """Process features in chunks matching model's expected input size."""
    n_bins, n_frames = feature.shape
    
    all_predictions = {comp: [] for comp in COMPONENT_NAMES}
    
    # Process in overlapping chunks
    chunk_size = timestep
    stride = timestep  # No overlap for simplicity
    
    for start in range(0, n_frames, stride):
        end = min(start + chunk_size, n_frames)
        chunk = feature[:, start:end]
        
        # Pad if necessary
        if chunk.shape[1] < chunk_size:
            pad_width = chunk_size - chunk.shape[1]
            chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode='constant', constant_values=-6)
        
        # Convert to tensor: (batch, seq_len, features)
        chunk_tensor = torch.tensor(chunk.T, dtype=torch.float32).unsqueeze(0)  # (1, timestep, n_bins)
        chunk_tensor = chunk_tensor.to(device)
        
        # Get predictions
        with torch.no_grad():
            output = model(chunk_tensor)
            
            if isinstance(output, tuple):
                predictions = output[0]
            else:
                predictions = model.decomposer.get_predictions(output)
        
        # Collect predictions (only up to actual frames, not padding)
        actual_frames = min(end - start, chunk_size)
        for comp in COMPONENT_NAMES:
            pred = predictions[comp][0, :actual_frames].cpu().numpy()
            all_predictions[comp].extend(pred)
    
    # Convert to arrays
    for comp in COMPONENT_NAMES:
        all_predictions[comp] = np.array(all_predictions[comp])
    
    return all_predictions


def main():
    parser = argparse.ArgumentParser(description='Full audio chord recognition')
    parser.add_argument('--config', type=str, default='run_config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--audio_file', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default=None, help='Save to file')
    parser.add_argument('--show_all', action='store_true', help='Show all frames (not just changes)')
    parser.add_argument('--max_frames', type=int, default=500, help='Max frames to display')
    parser.add_argument('--backbone', type=str, default='auto',
                       choices=['auto', 'btc', 'chordformer'],
                       help='Model backbone (auto detects from checkpoint)')
    
    args = parser.parse_args()
    
    # Load config
    config = HParams.load(args.config)
    device = torch.device(args.device)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = _build_model(config, backbone=args.backbone, checkpoint_meta=checkpoint).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    detected = checkpoint.get('training_config', {}).get('backbone', 'btc')
    logger.info(f"Backbone: {detected}")
    model.eval()

    normalization = checkpoint.get('normalization', None)
    if normalization is not None:
        logger.info(f"Normalization: mean={normalization['mean']:.6f}, std={normalization['std']:.6f}")
    else:
        logger.info("Normalization: disabled (raw log-CQT)")
    
    # Extract features
    feature, sr, hop_length = extract_features_full(args.audio_file, config)

    if normalization is not None:
        feature = (feature - normalization['mean']) / normalization['std']
    
    # Process
    logger.info("Running inference...")
    predictions = process_in_chunks(feature, model, device, config.model['timestep'])
    
    # Reassemble chords
    reassembler = ChordReassembler()
    n_frames = len(predictions['root'])
    
    logger.info(f"\nTotal frames: {n_frames}")
    logger.info(f"Frame duration: {hop_length/sr:.3f}s")
    logger.info(f"Total duration: {n_frames * hop_length / sr:.2f}s\n")
    
    # Build chord sequence
    chords = []
    for i in range(n_frames):
        indices = {comp: int(predictions[comp][i]) for comp in COMPONENT_NAMES}
        chord = reassembler.reassemble_from_indices(indices)
        chords.append(chord)
    
    # Format output
    frame_duration = hop_length / sr
    
    if args.show_all:
        # Show all frames (limited)
        logger.info("=== All Frames ===")
        for i in range(min(n_frames, args.max_frames)):
            t = i * frame_duration
            logger.info(f"{t:8.2f}s: {chords[i]}")
        if n_frames > args.max_frames:
            logger.info(f"... ({n_frames - args.max_frames} more frames)")
    else:
        # Show chord changes only
        logger.info("=== Chord Changes ===")
        current_chord = chords[0]
        start_time = 0.0
        
        for i in range(1, n_frames):
            if chords[i] != current_chord:
                end_time = i * frame_duration
                logger.info(f"{start_time:8.2f}s - {end_time:8.2f}s: {current_chord}")
                current_chord = chords[i]
                start_time = end_time
        
        # Final chord
        end_time = n_frames * frame_duration
        logger.info(f"{start_time:8.2f}s - {end_time:8.2f}s: {current_chord}")
    
    # Chord statistics
    from collections import Counter
    chord_counts = Counter(chords)
    logger.info("\n=== Chord Statistics ===")
    for chord, count in chord_counts.most_common(15):
        pct = 100 * count / n_frames
        logger.info(f"  {chord:15s}: {count:5d} frames ({pct:5.1f}%)")
    
    # Save to file
    if args.output:
        logger.info(f"\nSaving to {args.output}")
        with open(args.output, 'w') as f:
            current_chord = chords[0]
            start_time = 0.0
            
            for i in range(1, n_frames):
                if chords[i] != current_chord:
                    end_time = i * frame_duration
                    f.write(f"{start_time:.3f}\t{end_time:.3f}\t{current_chord}\n")
                    current_chord = chords[i]
                    start_time = end_time
            
            end_time = n_frames * frame_duration
            f.write(f"{start_time:.3f}\t{end_time:.3f}\t{current_chord}\n")
        logger.info("Done!")


if __name__ == '__main__':
    main()
