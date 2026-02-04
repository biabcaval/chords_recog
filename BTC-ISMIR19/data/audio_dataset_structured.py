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
        res['feature'] = np.log(np.abs(data['feature']) + 1e-6)
        
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
    
    Combines batch samples and returns:
    - features: (batch_size, 1, feature_size, max_len)
    - input_percentages: (batch_size,)
    - components: dict of tensors, each shape (batch_size*time_length,)
    - chord_lens: (batch_size,)
    - boundaries: (batch_size*time_length,)
    """
    batch_size = len(batch)
    max_len = batch[0]['feature'].shape[1]
    
    input_percentages = torch.empty(batch_size)
    chord_lens = torch.empty(batch_size, dtype=torch.int64)
    features = []
    boundaries = []
    
    # Initialize component tensors
    components_data = {component: [] for component in COMPONENT_NAMES}
    
    for i in range(batch_size):
        sample = batch[i]
        feature = sample['feature']
        chord = sample['chord']
        
        # Determine boundaries
        if isinstance(chord, np.ndarray):
            diff = np.diff(chord, axis=0).astype(bool) if len(chord.shape) > 1 else np.diff(chord).astype(bool)
        else:
            # Handle list case
            chord = np.array(chord)
            diff = np.diff(chord, axis=0).astype(bool) if len(chord.shape) > 1 else np.diff(chord).astype(bool)
        
        idx = np.insert(diff, 0, True, axis=0)
        chord_lens[i] = np.sum(idx).item() if np.sum(idx).dim() > 0 else np.sum(idx)
        
        features.append(feature)
        input_percentages[i] = feature.shape[1] / max_len
        
        boundary = np.append([0], diff)
        boundaries.extend(boundary.tolist() if isinstance(boundary, np.ndarray) else boundary)
        
        # Collect component data
        if 'components' in sample:
            components = sample['components']
            for component in COMPONENT_NAMES:
                components_data[component].extend(components[component].tolist())
    
    # Convert to tensors
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
    boundaries_tensor = torch.tensor(boundaries, dtype=torch.uint8)
    
    # Convert components to tensors
    components_tensor = {}
    for component in COMPONENT_NAMES:
        components_tensor[component] = torch.tensor(
            components_data[component], 
            dtype=torch.int64
        )
    
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
