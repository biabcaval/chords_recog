# encoding: utf-8
"""
Chord Structure Decomposition module.

This module implements the decomposition of chord symbols into 8 independent components
following the Chord Structure Decomposition technique.

Components:
1. Root: 13 classes (N, C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
2. Bass: 13 classes (same as Root)
3. Triad: 7 classes (N, maj, min, dim, aug, sus2, sus4)
4. Misc (Power Chord): 2 classes (N, 5)
5. 7th: 4 classes (N, 7, b7, bb7)
6. 9th: 4 classes (N, 9, #9, b9)
7. 11th: 3 classes (N, 11, #11)
8. 13th: 3 classes (N, 13, b13)
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import torch

# Vocabulary definitions
CHORD_VOCAB = {
    'root': ['N', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
    'bass': ['N', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
    'triad': ['N', 'maj', 'min', 'dim', 'aug', 'sus2', 'sus4'],
    'misc': ['N', '5'],  # Power chord
    '7th': ['N', '7', 'b7', 'bb7'],
    '9th': ['N', '9', '#9', 'b9'],
    '11th': ['N', '11', '#11'],
    '13th': ['N', '13', 'b13']
}

# Component order (consistent with the vocab above)
COMPONENT_NAMES = ['root', 'bass', 'triad', 'misc', '7th', '9th', '11th', '13th']
NUM_COMPONENTS = len(COMPONENT_NAMES)

# Reverse mappings for easy lookup
CHORD_VOCAB_IDX = {component: {label: idx for idx, label in enumerate(labels)}
                   for component, labels in CHORD_VOCAB.items()}

# Interval definitions for chord types
INTERVAL_DEFINITIONS = {
    'maj': {'notes': [0, 4, 7], 'quality': 'major'},
    'min': {'notes': [0, 3, 7], 'quality': 'minor'},
    'dim': {'notes': [0, 3, 6], 'quality': 'diminished'},
    'aug': {'notes': [0, 4, 8], 'quality': 'augmented'},
    'sus2': {'notes': [0, 2, 7], 'quality': 'suspended2'},
    'sus4': {'notes': [0, 5, 7], 'quality': 'suspended4'},
    '5': {'notes': [0, 7], 'quality': 'power'},
}

# 7th extensions
SEVENTH_EXTENSIONS = {
    '7': 10,  # minor 7th (1 semitone below octave)
    'b7': 10,  # same as '7'
    'bb7': 9,  # diminished 7th
    '7': 11,  # major 7th (but typically just '7' = minor 7th)
}


class ChordDecomposer:
    """
    Decomposes chord labels into 8 independent components.
    
    Example:
        decomposer = ChordDecomposer()
        components = decomposer.decompose('C:maj9')
        # Returns: {'root': 'C', 'bass': 'N', 'triad': 'maj', 'misc': 'N',
        #           '7th': 'N', '9th': '9', '11th': 'N', '13th': 'N'}
    """
    
    def __init__(self):
        self.pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.vocab = CHORD_VOCAB
        self.vocab_idx = CHORD_VOCAB_IDX
    
    def decompose(self, chord_label: str) -> Dict[str, str]:
        """
        Decompose a chord label into 8 components.
        
        Args:
            chord_label: Chord label string (e.g., 'C:maj9', 'D:min7/F#')
            
        Returns:
            Dictionary with 8 keys: root, bass, triad, misc, 7th, 9th, 11th, 13th
        """
        # Initialize all components to 'N' (None/absent)
        components = {component: 'N' for component in COMPONENT_NAMES}
        
        # Handle special cases
        if chord_label == 'N' or chord_label == 'X':
            return components
        
        try:
            # Parse chord label
            root, quality, bass = self._parse_chord(chord_label)
            
            # Set root and bass
            if root is not None:
                components['root'] = root
            if bass is not None and bass != root:
                components['bass'] = bass
            
            # Decompose quality and extensions
            if quality:
                self._decompose_quality(quality, components)
        
        except Exception as e:
            print(f"Error decomposing chord '{chord_label}': {e}")
            return components
        
        return components
    
    def _parse_chord(self, label: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse chord label into (root, quality, bass).
        
        Examples:
            'C:maj9' -> ('C', 'maj9', None)
            'D:min7/F#' -> ('D', 'min7', 'F#')
            'C' -> ('C', None, None)
        """
        root = None
        quality = None
        bass = None
        
        # Find the ':' separator for quality
        colon_idx = label.find(':')
        slash_idx = label.find('/')
        
        if colon_idx == -1 and slash_idx == -1:
            # Just a root note (e.g., 'C')
            root = label
        elif colon_idx == -1:
            # Root with bass but no quality (e.g., 'C/E')
            root = label[:slash_idx]
            bass = label[slash_idx + 1:]
        elif slash_idx == -1:
            # Root with quality but no bass (e.g., 'C:maj9')
            root = label[:colon_idx]
            quality = label[colon_idx + 1:]
        else:
            # Root, quality, and bass (e.g., 'C:maj9/E')
            root = label[:colon_idx]
            quality = label[colon_idx + 1:slash_idx]
            bass = label[slash_idx + 1:]
        
        # Validate root and bass are valid pitch classes
        if root and root not in self.pitch_classes:
            root = None
        if bass and bass not in self.pitch_classes:
            bass = None
        
        return root, quality, bass
    
    def _decompose_quality(self, quality: str, components: Dict[str, str]) -> None:
        """
        Decompose quality string into triad and extensions.
        
        Examples:
            'maj9' -> triad='maj', extensions include 9th
            'min7' -> triad='min', extensions include 7th
            '5' -> triad='N', misc='5'
            '7' -> triad='maj' (implied), extensions include 7th
            'hdim7' -> triad='dim', extensions include 7th (half-diminished)
        """
        quality_lower = quality.lower()
        
        # Check for special cases
        if quality_lower == '5' or quality_lower == 'pedal':
            components['misc'] = '5'
            return
        
        # Handle half-diminished (hdim = min7b5)
        if quality_lower.startswith('hdim') or quality_lower == 'hdim7':
            components['triad'] = 'dim'
            remaining = quality_lower.replace('hdim', '', 1)
            self._extract_extensions(remaining, components)
            return
        
        # Handle power chord variations
        if quality_lower.startswith('(1,5)') or quality_lower == 'power':
            components['misc'] = '5'
            return
        
        # Extract triad type (check longer patterns first)
        triad = None
        remaining = quality_lower
        
        # Order matters: check longer patterns first to avoid partial matches
        triad_patterns = ['sus2', 'sus4', 'maj', 'min', 'dim', 'aug']
        for triad_name in triad_patterns:
            if triad_name in quality_lower:
                triad = triad_name
                # Remove triad from remaining, being careful about position
                idx = quality_lower.find(triad_name)
                remaining = quality_lower[:idx] + quality_lower[idx + len(triad_name):]
                break
        
        if triad:
            components['triad'] = triad
        else:
            # If no explicit triad but has extensions, assume major triad
            # e.g., 'C:7' means C major with dominant 7th
            if any(ext in quality_lower for ext in ['7', '9', '11', '13']):
                components['triad'] = 'maj'
                remaining = quality_lower
        
        # Extract extensions from remaining string
        self._extract_extensions(remaining, components)
    
    def _extract_extensions(self, remaining: str, components: Dict[str, str]) -> None:
        """
        Extract extension components from remaining quality string.
        
        The order of extraction is important: higher extensions first (13 before 11 before 9)
        to avoid partial matches (e.g., '13' containing '1').
        
        Examples:
            '9' -> adds 9th component
            'b7' -> adds 7th component with flat
            '13' -> adds 13th component
        """
        # Handle 13th extension FIRST (to avoid '1' matching in '13')
        if 'b13' in remaining:
            components['13th'] = 'b13'
            remaining = remaining.replace('b13', '', 1)
        elif '#13' in remaining:
            # Non-standard but handle gracefully
            components['13th'] = '13'
            remaining = remaining.replace('#13', '', 1)
        elif '13' in remaining:
            components['13th'] = '13'
            remaining = remaining.replace('13', '', 1)
        
        # Handle 11th extension SECOND (to avoid '1' matching in '11')
        if '#11' in remaining:
            components['11th'] = '#11'
            remaining = remaining.replace('#11', '', 1)
        elif 'b11' in remaining:
            # b11 is enharmonic to #9 but handle as 11
            components['11th'] = '11'
            remaining = remaining.replace('b11', '', 1)
        elif '11' in remaining:
            components['11th'] = '11'
            remaining = remaining.replace('11', '', 1)
        
        # Handle 9th extension THIRD
        if '#9' in remaining:
            components['9th'] = '#9'
            remaining = remaining.replace('#9', '', 1)
        elif 'b9' in remaining:
            components['9th'] = 'b9'
            remaining = remaining.replace('b9', '', 1)
        elif '9' in remaining:
            components['9th'] = '9'
            remaining = remaining.replace('9', '', 1)
        
        # Handle 7th extension LAST
        if 'bb7' in remaining:
            components['7th'] = 'bb7'
            remaining = remaining.replace('bb7', '', 1)
        elif 'maj7' in remaining:
            # Major 7th
            components['7th'] = '7'
            remaining = remaining.replace('maj7', '', 1)
        elif 'b7' in remaining:
            components['7th'] = 'b7'
            remaining = remaining.replace('b7', '', 1)
        elif '7' in remaining:
            components['7th'] = '7'
            remaining = remaining.replace('7', '', 1)
    
    def decompose_batch(self, chord_labels: List[str]) -> Dict[str, np.ndarray]:
        """
        Decompose a batch of chord labels into component indices.
        
        Args:
            chord_labels: List of chord label strings
            
        Returns:
            Dictionary mapping component names to arrays of indices
        """
        result = {component: [] for component in COMPONENT_NAMES}
        
        for label in chord_labels:
            components = self.decompose(label)
            for component in COMPONENT_NAMES:
                idx = self.vocab_idx[component].get(components[component], 0)  # Default to 'N' (index 0)
                result[component].append(idx)
        
        # Convert to numpy arrays
        result = {component: np.array(indices, dtype=np.int64)
                 for component, indices in result.items()}
        return result
    
    def to_indices(self, components: Dict[str, str]) -> Dict[str, int]:
        """
        Convert component strings to indices using the vocabulary.
        
        Args:
            components: Dictionary with component names as keys
            
        Returns:
            Dictionary with component names mapped to vocabulary indices
        """
        indices = {}
        for component in COMPONENT_NAMES:
            label = components.get(component, 'N')
            indices[component] = self.vocab_idx[component].get(label, 0)
        return indices


class ChordReassembler:
    """
    Reassembles chord components back into valid chord labels.
    
    Implements priority logic for musical validity:
    
    Priority Rules:
    1. If root is 'N', output is 'N' (no chord without root)
    2. If triad is 'N' AND misc is 'N', output is 'N' (need triad or power chord)
    3. If misc is '5' (power chord), use that instead of triad
    4. Extensions are only valid if there's a triad
    5. Bass is only added if different from root
    
    This ensures musically valid chord reconstructions.
    """
    
    def __init__(self):
        self.pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.vocab = CHORD_VOCAB
        self.vocab_idx = CHORD_VOCAB_IDX
    
    def reassemble(self, components: Dict[str, str]) -> str:
        """
        Reassemble chord label from decomposed components.
        
        Args:
            components: Dictionary with 8 component keys:
                - root, bass, triad, misc, 7th, 9th, 11th, 13th
            
        Returns:
            Reconstructed chord label string (e.g., 'C:maj7', 'D:min/F#', 'N')
        """
        root = components.get('root', 'N')
        triad = components.get('triad', 'N')
        misc = components.get('misc', 'N')
        bass = components.get('bass', 'N')
        
        # Priority 1: No root means no chord
        if root == 'N':
            return 'N'
        
        # Priority 2: Power chord case (misc = '5')
        if misc == '5':
            # Power chord: root:5 with optional bass
            chord = f"{root}:5"
            if bass != 'N' and bass != root:
                chord += f"/{bass}"
            return chord
        
        # Priority 3: No triad means no chord (unless power chord)
        if triad == 'N':
            return 'N'
        
        # Build chord string: root:triad
        chord = f"{root}:{triad}"
        
        # Add extensions in order: 7th, 9th, 11th, 13th
        # Extensions are added sequentially as they appear in chord notation
        ext_7th = components.get('7th', 'N')
        ext_9th = components.get('9th', 'N')
        ext_11th = components.get('11th', 'N')
        ext_13th = components.get('13th', 'N')
        
        # Add extensions in standard notation order
        if ext_7th != 'N':
            chord += ext_7th
        if ext_9th != 'N':
            chord += ext_9th
        if ext_11th != 'N':
            chord += ext_11th
        if ext_13th != 'N':
            chord += ext_13th
        
        # Add bass note if different from root
        if bass != 'N' and bass != root:
            chord += f"/{bass}"
        
        return chord
    
    def reassemble_with_confidence(self, components: Dict[str, str], 
                                   confidences: Dict[str, float]) -> Tuple[str, float]:
        """
        Reassemble chord with overall confidence score.
        
        Args:
            components: Dictionary with 8 component keys
            confidences: Dictionary with confidence scores for each component
            
        Returns:
            Tuple of (chord_label, confidence_score)
        """
        chord = self.reassemble(components)
        
        if chord == 'N':
            # For N chord, confidence is based on root or triad being N
            conf = max(confidences.get('root', 0.0), confidences.get('triad', 0.0))
        else:
            # For valid chords, use minimum confidence across active components
            active_confs = [confidences.get('root', 1.0), confidences.get('triad', 1.0)]
            
            # Add confidence for bass if present
            if components.get('bass', 'N') != 'N':
                active_confs.append(confidences.get('bass', 1.0))
            
            # Add confidence for active extensions
            for ext in ['7th', '9th', '11th', '13th']:
                if components.get(ext, 'N') != 'N':
                    active_confs.append(confidences.get(ext, 1.0))
            
            conf = min(active_confs) if active_confs else 0.0
        
        return chord, conf
    
    def reassemble_from_indices(self, indices: Dict[str, int]) -> str:
        """
        Reassemble chord label from component indices.
        
        Args:
            indices: Dictionary mapping component names to vocabulary indices
            
        Returns:
            Reconstructed chord label string
        """
        components = {}
        for component in COMPONENT_NAMES:
            idx = indices.get(component, 0)
            # Handle both int and array-like indices
            if hasattr(idx, 'item'):
                idx = idx.item()
            idx = int(idx)
            # Clamp index to valid range
            idx = max(0, min(idx, len(self.vocab[component]) - 1))
            components[component] = self.vocab[component][idx]
        return self.reassemble(components)
    
    def reassemble_batch(self, indices_batch: Dict[str, np.ndarray]) -> List[str]:
        """
        Reassemble a batch of chord labels from component indices.
        
        Args:
            indices_batch: Dictionary mapping component names to arrays of indices.
                          Each array should have shape (batch_size,) or (batch_size, seq_len)
            
        Returns:
            List of reconstructed chord label strings
        """
        # Get total number of elements
        first_component = COMPONENT_NAMES[0]
        indices_array = indices_batch[first_component]
        
        if isinstance(indices_array, np.ndarray):
            total_elements = indices_array.size
            flat_indices = {comp: indices_batch[comp].flatten() 
                           for comp in COMPONENT_NAMES}
        else:
            # Handle tensor or list
            if hasattr(indices_array, 'numpy'):
                indices_array = indices_array.numpy()
            else:
                indices_array = np.array(indices_array)
            total_elements = indices_array.size
            flat_indices = {}
            for comp in COMPONENT_NAMES:
                arr = indices_batch[comp]
                if hasattr(arr, 'numpy'):
                    arr = arr.numpy()
                else:
                    arr = np.array(arr)
                flat_indices[comp] = arr.flatten()
        
        result = []
        for i in range(total_elements):
            indices = {component: flat_indices[component][i] 
                      for component in COMPONENT_NAMES}
            chord_label = self.reassemble_from_indices(indices)
            result.append(chord_label)
        
        return result
    
    def reassemble_batch_2d(self, indices_batch: Dict[str, np.ndarray]) -> List[List[str]]:
        """
        Reassemble a 2D batch of chord labels from component indices.
        
        Args:
            indices_batch: Dictionary mapping component names to 2D arrays of indices.
                          Each array should have shape (batch_size, seq_len)
            
        Returns:
            List of lists: [[chord_labels for seq1], [chord_labels for seq2], ...]
        """
        first_component = COMPONENT_NAMES[0]
        indices_array = indices_batch[first_component]
        
        if hasattr(indices_array, 'numpy'):
            indices_array = indices_array.numpy()
        elif not isinstance(indices_array, np.ndarray):
            indices_array = np.array(indices_array)
        
        if len(indices_array.shape) == 1:
            # 1D array, return as single list
            return [self.reassemble_batch(indices_batch)]
        
        batch_size, seq_len = indices_array.shape
        result = []
        
        for b in range(batch_size):
            seq_result = []
            for t in range(seq_len):
                indices = {}
                for component in COMPONENT_NAMES:
                    arr = indices_batch[component]
                    if hasattr(arr, 'numpy'):
                        arr = arr.numpy()
                    elif not isinstance(arr, np.ndarray):
                        arr = np.array(arr)
                    indices[component] = arr[b, t]
                chord_label = self.reassemble_from_indices(indices)
                seq_result.append(chord_label)
            result.append(seq_result)
        
        return result


def get_vocab_sizes() -> Dict[str, int]:
    """
    Get vocabulary sizes for all components.
    
    Returns:
        Dictionary mapping component names to vocab sizes
    """
    return {component: len(vocab) for component, vocab in CHORD_VOCAB.items()}
