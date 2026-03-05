# encoding: utf-8
"""
Extended AudioDataset with Chord Structure Decomposition support.

This module provides enhanced dataset loading with 9-component chord decomposition.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data.audio_dataset import AudioDataset as BaseAudioDataset
from utils.chord_decomposition import ChordDecomposer, COMPONENT_NAMES, NUM_COMPONENTS, CHORD_VOCAB
from utils.mir_eval_modules import idx2voca_chord
import logging

logger = logging.getLogger(__name__)

# Global mapping from chord index to chord label (for 170-class vocabulary)
_IDX2CHORD = None

def get_idx2chord_mapping():
    """Get or create the index to chord label mapping."""
    global _IDX2CHORD
    if _IDX2CHORD is None:
        _IDX2CHORD = idx2voca_chord()
    return _IDX2CHORD


class AudioDatasetStructured(BaseAudioDataset):
    """
    Extended AudioDataset that decomposes chords into 9 independent components.
    
    This class extends the base AudioDataset to support Chord Structure Decomposition,
    where each chord is decomposed into:
    1. Root: 13 classes
    2. Bass: 13 classes
    3. Triad: 7 classes
    4. Misc (Power Chord): 2 classes
    5. 6th: 2 classes
    6. 7th: 4 classes
    7. 9th: 4 classes
    8. 11th: 3 classes
    9. 13th: 3 classes
    
    Example:
        dataset = AudioDatasetStructured(config, root_dir='/data', train=True)
        sample = dataset[0]
        # sample contains:
        # - 'feature': audio features
        # - 'chord': original chord labels (for compatibility)
        # - 'components': dict with 9 component arrays
    """
    
    def __init__(self, *args, decompose=True, normalization=None, **kwargs):
        """
        Initialize the structured dataset.
        
        Args:
            decompose: If True, decompose chords into components
            normalization: Optional dict with 'mean' and 'std' keys for
                          feature normalization, or path to a .pt file.
                          When provided, features are standardized after
                          log-magnitude transform.
            *args, **kwargs: Arguments passed to parent AudioDataset
        """
        super().__init__(*args, **kwargs)
        self.decompose = decompose
        self.decomposer = ChordDecomposer() if decompose else None
        self.component_names = COMPONENT_NAMES

        self.norm_mean = None
        self.norm_std = None
        if normalization is not None:
            if isinstance(normalization, str):
                normalization = torch.load(normalization, weights_only=False)
            self.norm_mean = float(normalization['mean'])
            self.norm_std = float(normalization['std'])
    
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

        if self.norm_mean is not None:
            features = (features - self.norm_mean) / self.norm_std
        
        # Features are saved as (n_bins, timesteps), need to transpose to (timesteps, n_bins)
        if features.shape[0] != self.config.model['timestep']:
            features = features.T  # Now (timesteps, n_bins)
        
        # Clip/pad to expected timestep
        timestep = self.config.model['timestep'] if hasattr(self.config, 'model') else 108
        if features.shape[0] > timestep:
            features = features[:timestep, :]
        elif features.shape[0] < timestep:
            pad_length = timestep - features.shape[0]
            features = np.pad(features, ((0, pad_length), (0, 0)), mode='constant')
        
        res['feature'] = torch.FloatTensor(features)
        
        # Handle chord labels - prefer original_chord_labels (full extensions)
        # Priority: original_chord_labels > original_chords > chord
        if 'original_chord_labels' in data:
            # New format with full chord labels (includes extensions like 9, 11, 13)
            res['chord'] = data['original_chord_labels']
            chord_labels = data['original_chord_labels']
        elif 'original_chords' in data:
            res['chord'] = data['original_chords']
            chord_labels = self._get_chord_labels(data['original_chords'])
        elif 'chord' in data:
            res['chord'] = data['chord']
            chord_labels = self._get_chord_labels(data['chord'])
        else:
            res['chord'] = []
            chord_labels = []
        
        # Decompose chords if requested
        if self.decompose and self.decomposer is not None and len(chord_labels) > 0:
            components_indices = self.decomposer.decompose_batch(chord_labels)
            # Convert numpy arrays to tensors
            for component_name in COMPONENT_NAMES:
                if isinstance(components_indices[component_name], np.ndarray):
                    components_indices[component_name] = torch.LongTensor(components_indices[component_name])
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
            List of chord label strings (e.g., 'C:maj', 'D:min7', 'N')
        """
        if isinstance(chord_data, (list, np.ndarray)):
            if len(chord_data) == 0:
                return chord_data
            
            # Get the first element to check type
            first_elem = chord_data[0]
            
            # Check if it's already chord label strings (e.g., 'C:maj', 'N')
            if isinstance(first_elem, str):
                # Check if the string looks like a chord label vs a numeric index
                if ':' in first_elem or first_elem in ('N', 'X'):
                    return chord_data
                # It's a numeric string like '130' - convert to int first
                try:
                    chord_data = [int(c) for c in chord_data]
                    first_elem = chord_data[0]
                except (ValueError, TypeError):
                    # Can't convert, return as-is
                    return chord_data
            
            # If indices (int or numpy int), convert to chord labels
            if isinstance(first_elem, (int, np.integer)):
                idx2chord = get_idx2chord_mapping()
                chord_labels = []
                for idx in chord_data:
                    idx_int = int(idx)
                    # Handle out of range indices
                    if idx_int in idx2chord:
                        chord_labels.append(idx2chord[idx_int])
                    else:
                        # Unknown index -> 'N'
                        chord_labels.append('N')
                return chord_labels
            
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
    
    The labels are represented as a dict of 9 tensors (one per component),
    each with shape (batch_size, seq_len).
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
        '6th': 2,      # N, 6
        '7th': 4,      # N, 7, b7, bb7
        '9th': 4,      # N, 9, #9, b9
        '11th': 3,     # N, 11, #11
        '13th': 3      # N, 13, b13
    }
