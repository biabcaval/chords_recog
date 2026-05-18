#!/usr/bin/env python
# encoding: utf-8
"""
Inference script for chord recognition with structure decomposition.

This script performs inference on audio files using the decomposed chord model
and returns the predicted chord sequences.

Usage:
    python infer_decomposed.py --config run_config.yaml --checkpoint model_best.pt --audio_file song.mp3
    python infer_decomposed.py --config run_config.yaml --checkpoint model_best.pt --audio_file song.mp3 --backbone chordformer
    python infer_decomposed.py --config run_config.yaml --checkpoint model_best.pt --audio_file song.mp3 --output result.lab --aggregate
"""

import argparse
import torch
import numpy as np
import librosa
from pathlib import Path
import logging

from models.btc_model_decomposed import BTC_model_decomposed, ChordFormer_model_decomposed
from utils.decomposed_inference import DecomposedChordInference, ChordMetrics
from utils.chord_decomposition import ChordDecomposer, ChordReassembler
from utils.hparams import HParams
from utils.preprocess import cqt_to_log_db


def _build_model(config, backbone='auto', checkpoint_meta=None):
    """
    Instantiate the correct decomposed model based on backbone choice.

    When backbone='auto', tries to detect from checkpoint metadata
    (training_config.backbone), falling back to 'btc'.
    Also restores model_config fields (e.g. use_head_ffn) from checkpoint
    so the model architecture matches the saved weights.
    """
    if checkpoint_meta is not None:
        tc = checkpoint_meta.get('training_config', {})
        if backbone == 'auto':
            backbone = tc.get('backbone', 'btc')
        model_cfg = tc.get('model_config', {})
        for key in ('use_head_ffn', 'head_ffn_dim'):
            if key in model_cfg:
                config.model[key] = model_cfg[key]

    if backbone == 'chordformer':
        return ChordFormer_model_decomposed(config=config)
    return BTC_model_decomposed(config=config)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChordRecognitionInference:
    """
    High-level interface for chord recognition inference.
    """
    
    def __init__(self, config_path, checkpoint_path, device='cuda', backbone='auto'):
        """
        Initialize the inference pipeline.
        
        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            backbone: 'btc', 'chordformer', or 'auto' (detect from checkpoint)
        """
        self.device = torch.device(device)
        
        # Load configuration
        self.config = HParams.load(config_path)
        
        # Load checkpoint first so we can detect backbone when auto
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Initialize model with correct backbone
        self.model = _build_model(self.config, backbone=backbone, checkpoint_meta=checkpoint)
        self.model = self.model.to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        detected = checkpoint.get('training_config', {}).get('backbone', 'btc')
        logger.info(f"Loaded checkpoint from {checkpoint_path} (backbone: {detected})")

        self.normalization = checkpoint.get('normalization', None)
        if self.normalization is not None:
            logger.info(f"Normalization: mean={self.normalization['mean']:.6f}, std={self.normalization['std']:.6f}")
        else:
            logger.info("Normalization: disabled (raw log-CQT)")
        
        # Setup inference utilities
        self.inference = DecomposedChordInference(self.model, device=self.device)
        self.decomposer = ChordDecomposer()
        self.reassembler = ChordReassembler()
        self.metrics = ChordMetrics()
        
        # Feature extraction parameters
        self.sr = self.config.mp3['song_hz']
        self.hop_length = self.config.feature['hop_length']
        self.n_bins = self.config.feature['n_bins']
        self.bins_per_octave = self.config.feature['bins_per_octave']
    
    def extract_features(self, audio_path):
        """
        Extract CQT features from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            features: CQT feature tensor (1, 1, feature_size, time_steps)
        """
        logger.info(f"Loading audio from {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr)
        logger.info(f"Audio shape: {y.shape}, Sample rate: {sr}")
        
        # Extract CQT
        logger.info("Extracting CQT features...")
        cqt = librosa.cqt(
            y,
            sr=sr,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            hop_length=self.hop_length
        )
        
        # Log magnitude in dB ref=max (ChordFormer-style); see utils.preprocess.cqt_to_log_db
        feature = cqt_to_log_db(cqt)
        if self.normalization is not None:
            feature = (feature - self.normalization['mean']) / self.normalization['std']
        
        # Normalize to expected shape
        expected_length = self.config.mp3['inst_len'] / (self.hop_length / self.sr)
        if feature.shape[1] > expected_length:
            feature = feature[:, :int(expected_length)]
        elif feature.shape[1] < expected_length:
            pad_width = ((0, 0), (0, int(expected_length) - feature.shape[1]))
            feature = np.pad(feature, pad_width, mode='constant', constant_values=0)
        
        # Convert to tensor with batch dimension
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        feature_tensor = feature_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, bins, time)
        
        logger.info(f"Feature shape: {feature_tensor.shape}")
        
        return feature_tensor
    
    def recognize_chords(self, audio_path, return_confidence=False):
        """
        Recognize chords in audio file.
        
        Args:
            audio_path: Path to audio file
            return_confidence: If True, return confidence scores
            
        Returns:
            chords: List of recognized chord labels
            confidence: Optional array of confidence scores
            timing: Optional array with frame timings (in seconds)
        """
        # Extract features
        features = self.extract_features(audio_path)
        features = features.to(self.device)
        
        # Predict
        logger.info("Running chord recognition...")
        if return_confidence:
            probabilities = self.inference.predict_batch(features, return_probabilities=True)
            predictions = self.inference.predict_batch(features, return_probabilities=False)
            
            # Decode
            chords = self.inference.decode_predictions(predictions, reshape_to_sequences=False)
            confidence = self.inference.get_confidence_scores(probabilities)
        else:
            chords = self.inference.predict_and_decode(features)
            confidence = None
        
        # Compute frame timings
        n_frames = len(chords)
        timing = np.arange(n_frames) * (self.hop_length / self.sr)
        
        return chords, confidence, timing
    
    def recognize_and_format(self, audio_path, aggregate=True):
        """
        Recognize chords and format as time-stamped chord changes.
        
        Args:
            audio_path: Path to audio file
            aggregate: If True, return only chord changes
            
        Returns:
            chord_sequence: List of (start_time, end_time, chord) tuples
        """
        chords, confidence, timing = self.recognize_chords(audio_path, return_confidence=True)
        
        if not aggregate:
            # Return all frames
            chord_sequence = []
            for i, (chord, conf, t) in enumerate(zip(chords, confidence, timing)):
                start_time = t
                end_time = timing[i + 1] if i + 1 < len(timing) else t + (self.hop_length / self.sr)
                chord_sequence.append((start_time, end_time, chord, conf))
            return chord_sequence
        
        # Aggregate: return only chord changes
        chord_sequence = []
        current_chord = chords[0]
        start_time = timing[0]
        current_confidence = confidence[0]
        
        for i in range(1, len(chords)):
            if chords[i] != current_chord:
                end_time = timing[i]
                chord_sequence.append((start_time, end_time, current_chord, current_confidence))
                
                current_chord = chords[i]
                start_time = timing[i]
                current_confidence = confidence[i]
        
        # Add final chord
        end_time = timing[-1] + (self.hop_length / self.sr)
        chord_sequence.append((start_time, end_time, current_chord, current_confidence))
        
        return chord_sequence
    
    def evaluate_on_file(self, audio_path, reference_path=None):
        """
        Evaluate chord recognition on a file (if reference available).
        
        Args:
            audio_path: Path to audio file
            reference_path: Path to reference chord file (format: start end chord)
            
        Returns:
            metrics: Dictionary with evaluation metrics
        """
        # Recognize chords
        chords, _, _ = self.recognize_chords(audio_path, return_confidence=True)
        
        if reference_path is None:
            return {'predicted_chords': chords}
        
        # Load reference
        logger.info(f"Loading reference from {reference_path}")
        reference_chords = []
        with open(reference_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    chord = parts[2]
                    reference_chords.append(chord)
        
        # Match lengths
        min_len = min(len(chords), len(reference_chords))
        chords = chords[:min_len]
        reference_chords = reference_chords[:min_len]
        
        # Compute metrics
        logger.info("Computing metrics...")
        predictions = {
            'root': np.array([self.decomposer.decompose(c)['root'] for c in chords]),
            'bass': np.array([self.decomposer.decompose(c)['bass'] for c in chords]),
            'triad': np.array([self.decomposer.decompose(c)['triad'] for c in chords]),
            'misc': np.array([self.decomposer.decompose(c)['misc'] for c in chords]),
            '7th': np.array([self.decomposer.decompose(c)['7th'] for c in chords]),
            '9th': np.array([self.decomposer.decompose(c)['9th'] for c in chords]),
            '11th': np.array([self.decomposer.decompose(c)['11th'] for c in chords]),
            '13th': np.array([self.decomposer.decompose(c)['13th'] for c in chords]),
        }
        
        targets = {
            'root': np.array([self.decomposer.decompose(c)['root'] for c in reference_chords]),
            'bass': np.array([self.decomposer.decompose(c)['bass'] for c in reference_chords]),
            'triad': np.array([self.decomposer.decompose(c)['triad'] for c in reference_chords]),
            'misc': np.array([self.decomposer.decompose(c)['misc'] for c in reference_chords]),
            '7th': np.array([self.decomposer.decompose(c)['7th'] for c in reference_chords]),
            '9th': np.array([self.decomposer.decompose(c)['9th'] for c in reference_chords]),
            '11th': np.array([self.decomposer.decompose(c)['11th'] for c in reference_chords]),
            '13th': np.array([self.decomposer.decompose(c)['13th'] for c in reference_chords]),
        }
        
        metrics = self.metrics.evaluate(predictions, targets)
        
        return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Run chord recognition inference with structure decomposition'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--audio_file', type=str, required=True,
                       help='Path to audio file for inference')
    parser.add_argument('--reference', type=str, default=None,
                       help='Path to reference chord file (optional)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run inference on')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output chord file')
    parser.add_argument('--aggregate', action='store_true', default=False,
                       help='Aggregate predictions to chord changes only')
    parser.add_argument('--backbone', type=str, default='auto',
                       choices=['auto', 'btc', 'chordformer'],
                       help='Model backbone (auto detects from checkpoint)')
    
    args = parser.parse_args()
    
    # Initialize inference
    logger.info("Initializing inference pipeline...")
    inference_engine = ChordRecognitionInference(
        args.config,
        args.checkpoint,
        device=args.device,
        backbone=args.backbone
    )
    
    # Run inference
    if args.reference:
        logger.info("Evaluating with reference...")
        metrics = inference_engine.evaluate_on_file(args.audio_file, args.reference)
        
        logger.info("\n=== Evaluation Metrics ===")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{key}: {value:.4f}")
    else:
        logger.info("Running chord recognition...")
        chord_sequence = inference_engine.recognize_and_format(args.audio_file, aggregate=args.aggregate)
        
        logger.info("\n=== Recognized Chords ===")
        for start, end, chord, conf in chord_sequence:
            logger.info(f"{start:8.2f}s - {end:8.2f}s: {chord:15s} (confidence: {conf:.3f})")
        
        # Save output if requested
        if args.output:
            logger.info(f"Saving output to {args.output}")
            with open(args.output, 'w') as f:
                for start, end, chord, conf in chord_sequence:
                    f.write(f"{start:.2f} {end:.2f} {chord} {conf:.3f}\n")


if __name__ == '__main__':
    main()
