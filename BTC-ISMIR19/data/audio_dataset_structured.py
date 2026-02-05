# encoding: utf-8
"""
Extended AudioDataset with Chord Structure Decomposition support.

This module provides enhanced dataset loading with 8-component chord decomposition.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data.audio_dataset import AudioDataset as BaseAudioDataset
from utils.chord_decomposition import ChordDecomposer, COMPONENT_NAMES, NUM_COMPONENTS, CHORD_VOCAB
import logging

logger = logging.getLogger(__name__)


class AudioDatasetStructured(BaseAudioDataset):
    """
    Extended AudioDataset that decomposes chords into 8 independent components.
    
    This class extends the base AudioDataset to support Chord Structure Decomposition,
    where each chord is decomposed into:
    1. Root: 13 classes
    2. Bass: 13 classes
    3. Triad: 7 classes
    4. Misc (Power Chord): 2 classes
    5. 7th: 4 classes
    6. 9th: 4 classes
    7. 11th: 3 classes
    8. 13th: 3 classes
    
    Example:
        dataset = AudioDatasetStructured(config, root_dir='/data', train=True)
        sample = dataset[0]
        # sample contains:
        # - 'feature': audio features
        # - 'chord': original chord labels (for compatibility)
        # - 'components': dict with 8 component arrays
    """
    
    def __init__(self, *args, decompose=True, **kwargs):
        """
        Initialize the structured dataset.
        
        Args:
            decompose: If True, decompose chords into components
            *args, **kwargs: Arguments passed to parent AudioDataset
        """
        super().__init__(*args, **kwargs)
        self.decompose = decompose
        self.decomposer = ChordDecomposer() if decompose else None
        self.component_names = COMPONENT_NAMES
    
    def __getitem__(self, idx):
        """
        Load a sample and optionally decompose its chords.
        
        Returns:
            dict with keys:
                - 'feature': audio features (log magnitude)
                - 'chord': original chord labels
                - 'components': dict mapping component names to label indices
                    (only if decompose=True)
        """
        instance_path = self.paths[idx]
        
        res = dict()
        data = torch.load(instance_path, weights_only=False)

        features = np.log(np.abs(data['feature']) + 1e-6)
        
        # Features are saved as (n_bins, timesteps), need to transpose to (timesteps, n_bins)
        if features.shape[0] != self.config.model['timestep']:
            # Transpose if needed
            features = features.T  # Now (timesteps, n_bins)
        
        # Clip/pad to expected timestep if needed
        # Expected shape: (timestep, feature_size)
        timestep = self.config.model['timestep'] if hasattr(self.config, 'model') else 108
        if features.shape[0] > timestep:
            # Take first timestep frames
            features = features[:timestep, :]
        elif features.shape[0] < timestep:
            # Pad with zeros
            pad_length = timestep - features.shape[0]
            features = np.pad(features, ((0, pad_length), (0, 0)), mode='constant')
        
        # Convert to tensor
        res['feature'] = torch.FloatTensor(features)
        
        # Handle both old and new data formats
        if 'chord' in data:
            res['chord'] = data['chord']
        elif 'original_chords' in data:
            res['chord'] = data['original_chords']
        else:
            res['chord'] = []
        
        # Decompose chords if requested
        if self.decompose and self.decomposer is not None:
            # Check if data already has decomposed chords
            if 'decomposed_chord' in data:
                # Use pre-decomposed chords
                components_indices = self._convert_decomposed_to_indices(data['decomposed_chord'])
                res['components'] = components_indices
            else:
                # Convert chord indices back to labels for decomposition
                chord_labels = self._get_chord_labels(res['chord'])
                components_indices = self.decomposer.decompose_batch(chord_labels)
                res['components'] = components_indices
        
        return res
    
    def _convert_decomposed_to_indices(self, decomposed_chords):
        """
        Convert pre-decomposed chords (dicts) to component indices.
        
        Args:
            decomposed_chords: List of dicts with component labels
            
        Returns:
            Dict mapping component names to tensors of indices
        """
        components_indices = {name: [] for name in COMPONENT_NAMES}
        
        for decomposed in decomposed_chords:
            if isinstance(decomposed, dict):
                # Use decomposer's to_indices method
                indices = self.decomposer.to_indices(decomposed)
                for component_name in COMPONENT_NAMES:
                    components_indices[component_name].append(indices.get(component_name, 0))
            else:
                # If not dict, add default indices (all zeros/N)
                for component_name in COMPONENT_NAMES:
                    components_indices[component_name].append(0)
        
        # Convert to tensors
        for component_name in COMPONENT_NAMES:
            components_indices[component_name] = torch.LongTensor(components_indices[component_name])
        
        return components_indices
    
    def _get_chord_labels(self, chord_data):
        """
        Extract chord labels from various formats.
        
        Args:
            chord_data: Either list of chord labels (strings) or indices
            
        Returns:
            List of chord label strings
        """
        if isinstance(chord_data, (list, np.ndarray)):
            if len(chord_data) > 0:
                # Check if it's already strings
                if isinstance(chord_data[0], str):
                    return chord_data
                # If indices, would need mapping (implement as needed)
                else:
                    logger.warning("Chord data appears to be indices, not labels. "
                                  "Returning as-is for decomposition.")
                    return [str(c) for c in chord_data]
        return chord_data


def _collate_fn_structured(batch):
    """
    Collate function for structured chord decomposition data.
    
    Combines batch samples and returns a dictionary with:
        - features: (batch_size, 1, feature_size, seq_len) - audio features
        - input_percentages: (batch_size,) - fraction of max length used
        - components: dict of tensors, each shape (batch_size, seq_len) - decomposed labels
        - chord_lens: (batch_size,) - number of unique chords per sample
        - boundaries: (batch_size, seq_len) - boundary indicators
    
    The labels tensor shape is (batch_size, seq_len, 8) where 8 is the number of components.
    """
    batch_size = len(batch)
    
    # Get feature dimensions from first sample
    first_feature = batch[0]['feature']
    if len(first_feature.shape) == 2:
        # Shape: (seq_len, feature_size) from __getitem__
        seq_len, feature_size = first_feature.shape
    else:
        # Unexpected shape, try to infer
        feature_size = first_feature.shape[-1]
        seq_len = first_feature.shape[0]
    
    # Find max sequence length in batch
    max_len = max(sample['feature'].shape[0] for sample in batch)
    
    # Initialize tensors
    input_percentages = torch.empty(batch_size)
    chord_lens = torch.empty(batch_size, dtype=torch.int64)
    features_list = []
    boundaries_list = []
    
    # Initialize component data - will be (batch_size, seq_len) for each
    components_data = {component: [] for component in COMPONENT_NAMES}
    
    for i in range(batch_size):
        sample = batch[i]
        feature = sample['feature']
        
        # Ensure feature is 2D (seq_len, feature_size)
        if isinstance(feature, torch.Tensor):
            feature_np = feature.numpy()
        else:
            feature_np = np.array(feature)
        
        current_len = feature_np.shape[0]
        
        # Pad features if needed
        if current_len < max_len:
            pad_length = max_len - current_len
            feature_np = np.pad(feature_np, ((0, pad_length), (0, 0)), mode='constant')
        
        features_list.append(feature_np)
        input_percentages[i] = current_len / max_len
        
        # Process chord labels for boundaries
        chord = sample.get('chord', [])
        if isinstance(chord, (list, np.ndarray)) and len(chord) > 0:
            chord_array = np.array(chord) if isinstance(chord, list) else chord
            if len(chord_array) > 1:
                # Handle multi-dimensional chord arrays
                if len(chord_array.shape) > 1:
                    # Take first column for boundary detection
                    chord_flat = chord_array[:, 0] if chord_array.shape[1] > 0 else chord_array.flatten()
                else:
                    chord_flat = chord_array
                
                # Handle string chord labels (can't use np.diff on strings)
                if chord_flat.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object
                    # Compare adjacent elements for strings
                    diff = np.array([chord_flat[j] != chord_flat[j-1] for j in range(1, len(chord_flat))], dtype=bool)
                else:
                    # Numeric labels - use standard diff
                    diff = np.diff(chord_flat).astype(bool)
            else:
                diff = np.array([], dtype=bool)
            idx = np.insert(diff, 0, True)
            chord_lens[i] = int(np.sum(idx))
            boundary = np.append([0], diff)
        else:
            chord_lens[i] = 0
            boundary = np.zeros(current_len, dtype=np.uint8)
        
        # Pad boundary if needed
        if len(boundary) < max_len:
            boundary = np.pad(boundary, (0, max_len - len(boundary)), mode='constant')
        elif len(boundary) > max_len:
            boundary = boundary[:max_len]
        
        boundaries_list.append(boundary)
        
        # Collect component data
        if 'components' in sample:
            components = sample['components']
            for component in COMPONENT_NAMES:
                comp_data = components[component]
                if isinstance(comp_data, torch.Tensor):
                    comp_array = comp_data.numpy()
                elif isinstance(comp_data, np.ndarray):
                    comp_array = comp_data
                else:
                    comp_array = np.array(comp_data)
                
                # Pad component data if needed
                if len(comp_array) < max_len:
                    comp_array = np.pad(comp_array, (0, max_len - len(comp_array)), 
                                       mode='constant', constant_values=0)
                elif len(comp_array) > max_len:
                    comp_array = comp_array[:max_len]
                
                components_data[component].append(comp_array)
        else:
            # No components, fill with zeros (N class)
            for component in COMPONENT_NAMES:
                components_data[component].append(np.zeros(max_len, dtype=np.int64))
    
    # Stack features: (batch_size, seq_len, feature_size) -> (batch_size, 1, feature_size, seq_len)
    features_array = np.stack(features_list, axis=0)  # (batch_size, seq_len, feature_size)
    features_array = features_array.transpose(0, 2, 1)  # (batch_size, feature_size, seq_len)
    features_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(1)
    
    # Stack boundaries: (batch_size, seq_len)
    boundaries_array = np.stack(boundaries_list, axis=0)
    boundaries_tensor = torch.tensor(boundaries_array, dtype=torch.uint8)
    
    # Stack components: each (batch_size, seq_len)
    components_tensor = {}
    for component in COMPONENT_NAMES:
        comp_array = np.stack(components_data[component], axis=0)
        components_tensor[component] = torch.tensor(comp_array, dtype=torch.int64)
    
    return {
        'features': features_tensor,
        'input_percentages': input_percentages,
        'components': components_tensor,
        'chord_lens': chord_lens,
        'boundaries': boundaries_tensor
    }


class AudioDataLoaderStructured(DataLoader):
    """DataLoader with structured collate function for chord decomposition."""
    
    def __init__(self, *args, **kwargs):
        super(AudioDataLoaderStructured, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_structured


def get_component_vocab_sizes():
    """
    Get vocabulary size for each component.
    
    Returns:
        dict: Mapping of component names to vocabulary sizes
    """
    return {
        'root': 13,    # N, C, C#, ..., B
        'bass': 13,    # N, C, C#, ..., B
        'triad': 7,    # N, maj, min, dim, aug, sus2, sus4
        'misc': 2,     # N, 5
        '7th': 4,      # N, 7, b7, bb7
        '9th': 4,      # N, 9, #9, b9
        '11th': 3,     # N, 11, #11
        '13th': 3      # N, 13, b13
    }
