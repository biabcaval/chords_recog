# encoding: utf-8
"""
Chord Structure Decomposition module.

This module implements the decomposition of chord symbols into 9 independent components
following the Chord Structure Decomposition technique.

Components:
1. Root: 13 classes (N, C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
2. Bass: 13 classes (same as Root)
3. Triad: 7 classes (N, maj, min, dim, aug, sus2, sus4)
4. Misc (Power Chord): 2 classes (N, 5)
5. 6th: 2 classes (N, 6) - for min6, maj6 chords
6. 7th: 4 classes (N, 7, b7, bb7) - 7=maj7, b7=dom/min7, bb7=dim7
7. 9th: 4 classes (N, 9, #9, b9)
8. 11th: 3 classes (N, 11, #11)
9. 13th: 3 classes (N, 13, b13)
"""

import re
import numpy as np
from typing import Dict, Tuple, List, Optional
import torch

# Vocabulary definitions
CHORD_VOCAB = {
    'root': ['N', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
    'bass': ['N', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
    'triad': ['N', 'maj', 'min', 'dim', 'aug', 'sus2', 'sus4'],
    'misc': ['N', '5'],  # Power chord
    '6th': ['N', '6'],   # 6th extension (for min6, maj6)
    '7th': ['N', '7', 'b7', 'bb7'],  # 7=maj7, b7=dominant/min7, bb7=dim7
    '9th': ['N', '9', '#9', 'b9'],
    '11th': ['N', '11', '#11'],
    '13th': ['N', '13', 'b13']
}

# Component order (consistent with the vocab above)
COMPONENT_NAMES = ['root', 'bass', 'triad', 'misc', '6th', '7th', '9th', '11th', '13th']
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


_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_NOTE_TO_IDX = {n: i for i, n in enumerate(_NOTES)}
_FLAT_TO_SHARP = {'Cb': 'B', 'Db': 'C#', 'Eb': 'D#', 'Fb': 'E', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}

DEGREE_TO_SEMITONE = {
    '1': 0, 'b2': 1, '2': 2, 'b3': 3, '3': 4, '4': 5,
    'b5': 6, '#4': 6, '5': 7, '#5': 8, 'b6': 8, '6': 9,
    'bb7': 9, 'b7': 10, '7': 11,
}


def _parse_note(s: str) -> Tuple[str, int]:
    """Parse a note name from the start of a string, return (note, chars_consumed)."""
    if len(s) >= 2 and s[:2] in _FLAT_TO_SHARP:
        return _FLAT_TO_SHARP[s[:2]], 2
    if len(s) >= 2 and s[:2] in _NOTE_TO_IDX:
        return s[:2], 2
    if len(s) >= 1 and s[0] in _NOTE_TO_IDX:
        return s[0], 1
    return '', 0


def _shift_note(note: str, semitones: int) -> str:
    """Transpose a single note name by the given number of semitones."""
    if note in _FLAT_TO_SHARP:
        note = _FLAT_TO_SHARP[note]
    idx = _NOTE_TO_IDX.get(note)
    if idx is None:
        return note
    return _NOTES[(idx + semitones) % 12]


def transpose_chord(chord_str: str, semitones: int) -> str:
    """
    Transpose a chord label by the given number of semitones.

    Shifts root and bass notes while preserving quality and extensions.
    Handles flats by normalising to sharps before transposing.

    Examples:
        transpose_chord('A:min', 3)       -> 'C:min'
        transpose_chord('C:maj7(9)/E', 2) -> 'D:maj7(9)/F#'
        transpose_chord('N', 5)           -> 'N'
    """
    if not chord_str or chord_str in ('N', 'X'):
        return chord_str
    if semitones == 0:
        return chord_str

    colon_idx = chord_str.find(':')
    slash_idx = chord_str.rfind('/')

    if colon_idx == -1 and slash_idx == -1:
        note, consumed = _parse_note(chord_str)
        if not note:
            return chord_str
        return _shift_note(note, semitones) + chord_str[consumed:]

    if colon_idx != -1:
        root_str = chord_str[:colon_idx]
    elif slash_idx != -1:
        root_str = chord_str[:slash_idx]
    else:
        root_str = chord_str

    root_note, _ = _parse_note(root_str)
    if not root_note:
        return chord_str
    new_root = _shift_note(root_note, semitones)

    if slash_idx != -1 and slash_idx > colon_idx:
        bass_str = chord_str[slash_idx + 1:]
        bass_note, bass_consumed = _parse_note(bass_str)
        if bass_note:
            new_bass = _shift_note(bass_note, semitones)
            middle = chord_str[len(root_str):slash_idx]
            bass_suffix = bass_str[bass_consumed:]
            return new_root + middle + '/' + new_bass + bass_suffix
        else:
            middle = chord_str[len(root_str):]
            return new_root + middle
    else:
        rest = chord_str[len(root_str):]
        return new_root + rest


class ChordDecomposer:
    """
    Decomposes chord labels into 9 independent components.
    
    Example:
        decomposer = ChordDecomposer()
        components = decomposer.decompose('C:maj9')
        # Returns: {'root': 'C', 'bass': 'N', 'triad': 'maj', 'misc': 'N',
        #           '6th': 'N', '7th': '7', '9th': '9', '11th': 'N', '13th': 'N'}
    """
    
    # Mapping from flats to sharps (enharmonic equivalents)
    FLAT_TO_SHARP = {
        'Cb': 'B',
        'Db': 'C#',
        'Eb': 'D#',
        'Fb': 'E',
        'Gb': 'F#',
        'Ab': 'G#',
        'Bb': 'A#',
    }
    
    def __init__(self):
        self.pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.vocab = CHORD_VOCAB
        self.vocab_idx = CHORD_VOCAB_IDX
    
    def _normalize_pitch(self, pitch: str) -> str:
        """Convert flat notes to sharp equivalents."""
        if pitch in self.FLAT_TO_SHARP:
            return self.FLAT_TO_SHARP[pitch]
        return pitch
    
    def decompose(self, chord_label: str) -> Dict[str, str]:
        """
        Decompose a chord label into 9 components.
        
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
            elif root is not None:
                # No explicit quality means major triad (e.g., 'C' or 'F/A')
                components['triad'] = 'maj'
        
        except Exception as e:
            print(f"Error decomposing chord '{chord_label}': {e}")
            return components
        
        return components
    
    def _resolve_bass_degree(self, root: str, degree: str) -> Optional[str]:
        """Convert a scale-degree bass (e.g., '5', 'b3') to an absolute note name."""
        semitones = DEGREE_TO_SEMITONE.get(degree.lower())
        if semitones is None:
            return None
        root_idx = _NOTE_TO_IDX.get(root)
        if root_idx is None:
            return None
        return _NOTES[(root_idx + semitones) % 12]

    def _parse_chord(self, label: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse chord label into (root, quality, bass).
        
        Handles both note-name bass (e.g., /E) and scale-degree bass (e.g., /5, /b3)
        from Harte notation .lab files.
        
        Examples:
            'C:maj9'     -> ('C', 'maj9', None)
            'D:min7/F#'  -> ('D', 'min7', 'F#')
            'C:maj/5'    -> ('C', 'maj', 'G')     # degree resolved
            'A:min/b3'   -> ('A', 'min', 'C')     # degree resolved
            'C'          -> ('C', None, None)
        """
        root = None
        quality = None
        bass = None
        
        colon_idx = label.find(':')
        slash_idx = label.find('/')
        
        if colon_idx == -1 and slash_idx == -1:
            root = label
        elif colon_idx == -1:
            root = label[:slash_idx]
            bass = label[slash_idx + 1:]
        elif slash_idx == -1:
            root = label[:colon_idx]
            quality = label[colon_idx + 1:]
        else:
            root = label[:colon_idx]
            quality = label[colon_idx + 1:slash_idx]
            bass = label[slash_idx + 1:]
        
        if root:
            root = self._normalize_pitch(root)
        if bass:
            bass = self._normalize_pitch(bass)
        
        if root and root not in self.pitch_classes:
            root = None
        if bass and bass not in self.pitch_classes:
            if root and root in self.pitch_classes:
                bass = self._resolve_bass_degree(root, label[slash_idx + 1:])
            else:
                bass = None
        
        return root, quality, bass
    
    def _decompose_quality(self, quality: str, components: Dict[str, str]) -> None:
        """
        Decompose quality string into triad and extensions.

        Uses a phased architecture that correctly distinguishes shorthand
        notation (e.g., ``min7`` = minor + dominant 7th) from parenthetical
        extensions (e.g., ``min(7)`` = minor + **major** 7th added).

        Phases:
            1. Extract parenthetical content — ``(b9)``, ``(*3,9)``
            2. Process the shorthand (quality without parens)
            3. Add implied tones from shorthand (``9`` → implies 7th)
            4. Apply parenthetical extensions
            5. Apply omit (``*``) rules
        """
        paren_extensions, omit_notes = self._extract_paren_content(quality)

        shorthand = re.sub(r'\([^)]*\)', '', quality).strip()
        shorthand_lower = shorthand.lower()

        self._process_shorthand(shorthand_lower, components)
        self._add_implied_tones(shorthand, components)
        self._apply_paren_extensions(paren_extensions, components)
        self._apply_omit_rules(omit_notes, components)

    # ------------------------------------------------------------------
    # Phase 1: parenthetical content
    # ------------------------------------------------------------------

    def _extract_paren_content(self, quality: str) -> Tuple[List[str], List[str]]:
        """Split ``(...)`` groups into extensions and omit notes."""
        paren_extensions: List[str] = []
        omit_notes: List[str] = []

        for group in re.findall(r'\(([^)]*)\)', quality):
            for item in group.split(','):
                item = item.strip()
                if not item:
                    continue
                if item.startswith('*'):
                    omit_notes.append(item[1:])
                else:
                    paren_extensions.append(item)

        return paren_extensions, omit_notes

    # ------------------------------------------------------------------
    # Phase 2: shorthand processing
    # ------------------------------------------------------------------

    def _process_shorthand(self, shorthand: str, components: Dict[str, str]) -> None:
        """Process the shorthand portion of the quality (no parentheses)."""
        if shorthand in ('5', 'pedal', '1'):
            components['misc'] = '5'
            return

        if shorthand.startswith('hdim') or shorthand == 'hdim7':
            components['triad'] = 'dim'
            components['7th'] = 'b7'
            remaining = shorthand.replace('hdim', '', 1).replace('7', '', 1)
            self._extract_shorthand_extensions(remaining, components)
            return

        if 'minmaj7' in shorthand or 'minmaj' in shorthand:
            components['triad'] = 'min'
            components['7th'] = '7'
            remaining = shorthand.replace('minmaj7', '').replace('minmaj', '')
            self._extract_shorthand_extensions(remaining, components)
            return

        if shorthand == 'dim7' or shorthand.startswith('dim7'):
            components['triad'] = 'dim'
            components['7th'] = 'bb7'
            remaining = shorthand.replace('dim7', '')
            self._extract_shorthand_extensions(remaining, components)
            return

        if 'maj7' in shorthand:
            components['triad'] = 'maj'
            components['7th'] = '7'
            remaining = shorthand.replace('maj7', '')
            self._extract_shorthand_extensions(remaining, components)
            return

        if shorthand == 'power':
            components['misc'] = '5'
            return

        triad = None
        remaining = shorthand
        for triad_name in ('sus2', 'sus4', 'maj', 'min', 'dim', 'aug'):
            if triad_name in shorthand:
                triad = triad_name
                idx = shorthand.find(triad_name)
                remaining = shorthand[:idx] + shorthand[idx + len(triad_name):]
                break

        if triad:
            components['triad'] = triad
        else:
            if any(ext in shorthand for ext in ('6', '7', '9', '11', '13')):
                components['triad'] = 'maj'
                remaining = shorthand

        self._extract_shorthand_extensions(remaining, components)

    # ------------------------------------------------------------------
    # Phase 3: implied tones
    # ------------------------------------------------------------------

    def _add_implied_tones(self, shorthand: str, components: Dict[str, str]) -> None:
        """
        Add implied tones for shorthand extensions.

        Shorthand extensions imply lower extensions:
          - ``9``/``min9``/``maj9``  → implies 7th
          - ``11``/``min11``         → implies 7th + 9th
          - ``13``/``min13``         → implies 7th + 9th + 11th

        Parenthetical ``(9)`` means *add* and does **not** imply lower
        extensions — those are handled separately in ``_apply_paren_extensions``.
        """
        if components.get('7th', 'N') != 'N':
            return

        core = shorthand.lower()
        is_maj_quality = core.startswith('maj')
        implied_7th = '7' if is_maj_quality else 'b7'

        if '13' in core and components.get('13th', 'N') != 'N':
            components['7th'] = implied_7th
            if components.get('9th', 'N') == 'N':
                components['9th'] = '9'
            if components.get('11th', 'N') == 'N':
                components['11th'] = '11'
        elif '11' in core and components.get('11th', 'N') != 'N':
            components['7th'] = implied_7th
            if components.get('9th', 'N') == 'N':
                components['9th'] = '9'
        elif '9' in core and components.get('9th', 'N') != 'N':
            components['7th'] = implied_7th

    # ------------------------------------------------------------------
    # Phase 4: parenthetical extensions
    # ------------------------------------------------------------------

    def _apply_paren_extensions(self, extensions: List[str],
                                components: Dict[str, str]) -> None:
        """
        Apply parenthetical extensions to components.

        Key semantic difference from shorthand:
            ``(7)`` = *add* major 7th interval (7th = '7')
            ``(b7)`` = *add* dominant/minor 7th  (7th = 'b7')
        """
        for ext in extensions:
            low = ext.lower()

            # --- 7th variants (only set if shorthand hasn't already) ---
            if low == 'bb7':
                if components.get('7th', 'N') == 'N':
                    components['7th'] = 'bb7'
            elif low == 'b7':
                if components.get('7th', 'N') == 'N':
                    components['7th'] = 'b7'
            elif low == '7':
                if components.get('7th', 'N') == 'N':
                    components['7th'] = '7'

            # --- 9th ---
            elif low == '#9':
                components['9th'] = '#9'
            elif low == 'b9':
                components['9th'] = 'b9'
            elif low == '9':
                components['9th'] = '9'

            # --- 11th ---
            elif low == '#11':
                components['11th'] = '#11'
            elif low == '11':
                components['11th'] = '11'

            # --- 13th ---
            elif low == 'b13':
                components['13th'] = 'b13'
            elif low == '13':
                components['13th'] = '13'

            # --- 6th ---
            elif low == '6':
                components['6th'] = '6'

            # --- altered 5th — modifies triad ---
            elif low == 'b5':
                if components.get('triad') == 'min':
                    components['triad'] = 'dim'
            elif low == '#5':
                if components.get('triad') == 'maj':
                    components['triad'] = 'aug'

            # (4), (2), (b6) etc. — no vocab slot, silently ignored

    # ------------------------------------------------------------------
    # Phase 5: omit rules
    # ------------------------------------------------------------------

    def _apply_omit_rules(self, omit_notes: List[str],
                          components: Dict[str, str]) -> None:
        """
        Handle ``*`` (omit) notation.

        When the 3rd (``*3`` or ``*b3``) is omitted the triad identity is
        lost.  We approximate as a power chord (``misc='5'``).
        """
        if not omit_notes:
            return

        omit_lower = {o.lower() for o in omit_notes}
        if '3' in omit_lower or 'b3' in omit_lower:
            components['triad'] = 'N'
            components['misc'] = '5'

    # ------------------------------------------------------------------
    # Shorthand extension extraction (used by _process_shorthand)
    # ------------------------------------------------------------------

    def _extract_shorthand_extensions(self, remaining: str,
                                      components: Dict[str, str]) -> None:
        """
        Extract extensions from the leftover shorthand string.

        Extraction order (high → low) avoids partial matches
        (e.g., ``'13'`` must be consumed before ``'1'``).
        """
        if 'b13' in remaining:
            components['13th'] = 'b13'
            remaining = remaining.replace('b13', '', 1)
        elif '13' in remaining:
            components['13th'] = '13'
            remaining = remaining.replace('13', '', 1)

        if '#11' in remaining:
            components['11th'] = '#11'
            remaining = remaining.replace('#11', '', 1)
        elif '11' in remaining:
            components['11th'] = '11'
            remaining = remaining.replace('11', '', 1)

        if '#9' in remaining:
            components['9th'] = '#9'
            remaining = remaining.replace('#9', '', 1)
        elif 'b9' in remaining:
            components['9th'] = 'b9'
            remaining = remaining.replace('b9', '', 1)
        elif '9' in remaining:
            components['9th'] = '9'
            remaining = remaining.replace('9', '', 1)

        if components.get('7th', 'N') == 'N':
            if 'bb7' in remaining:
                components['7th'] = 'bb7'
                remaining = remaining.replace('bb7', '', 1)
            elif 'b7' in remaining:
                components['7th'] = 'b7'
                remaining = remaining.replace('b7', '', 1)
            elif '7' in remaining:
                components['7th'] = 'b7'
                remaining = remaining.replace('7', '', 1)

        if '6' in remaining:
            components['6th'] = '6'
            remaining = remaining.replace('6', '', 1)
    
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
            components: Dictionary with 9 component keys:
                - root, bass, triad, misc, 6th, 7th, 9th, 11th, 13th
            
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
            chord = f"{root}:5"
            paren_exts = []
            if components.get('6th', 'N') != 'N':
                paren_exts.append('6')
            if components.get('7th', 'N') != 'N':
                paren_exts.append(components['7th'])
            if components.get('9th', 'N') != 'N':
                paren_exts.append(components['9th'])
            if components.get('11th', 'N') != 'N':
                paren_exts.append(components['11th'])
            if components.get('13th', 'N') != 'N':
                paren_exts.append(components['13th'])
            if paren_exts:
                chord += f"({','.join(paren_exts)})"
            if bass != 'N' and bass != root:
                chord += f"/{bass}"
            return chord
        
        # Collect extensions before deciding on triad
        ext_6th = components.get('6th', 'N')
        ext_7th = components.get('7th', 'N')
        ext_9th = components.get('9th', 'N')
        ext_11th = components.get('11th', 'N')
        ext_13th = components.get('13th', 'N')

        has_ext = (ext_6th != 'N' or ext_7th != 'N' or ext_9th != 'N'
                   or ext_11th != 'N' or ext_13th != 'N')

        # Priority 3: No triad — if extensions are present, default to
        # major triad (mirrors decompose convention: implicit triad for
        # chords like "C:7").  Only output 'N' when nothing else is active.
        if triad == 'N':
            if has_ext or bass != 'N':
                triad = 'maj'
            else:
                return 'N'

        has_ext_7plus = ext_7th != 'N' or ext_9th != 'N' or ext_11th != 'N' or ext_13th != 'N'

        # All non-shorthand extensions are collected here so the output
        # always uses a single comma-separated parenthetical group,
        # e.g. sus4(b7,9) instead of sus4(b7)(9).
        paren_exts: list = []

        if ext_7th == 'N' and not has_ext_7plus:
            chord = f"{root}:{triad}"
            if ext_6th != 'N' and triad in ('maj', 'min'):
                chord = f"{root}:{triad}6"
        elif ext_7th == '7':
            # Major 7th interval
            if triad == 'min':
                chord = f"{root}:minmaj7"
            elif triad == 'maj':
                chord = f"{root}:maj7"
            else:
                chord = f"{root}:{triad}"
                paren_exts.append('7')
        elif ext_7th == 'b7':
            # Minor / dominant 7th interval
            if triad == 'maj':
                chord = f"{root}:7"
            elif triad == 'min':
                chord = f"{root}:min7"
            elif triad == 'dim':
                chord = f"{root}:hdim7"
            else:
                chord = f"{root}:{triad}"
                paren_exts.append('b7')
        elif ext_7th == 'bb7':
            # Diminished 7th interval
            if triad == 'dim':
                chord = f"{root}:dim7"
            else:
                chord = f"{root}:{triad}"
                paren_exts.append('bb7')
        else:
            chord = f"{root}:{triad}"

        if ext_6th != 'N' and has_ext_7plus:
            paren_exts.append('6')
        if ext_9th != 'N':
            paren_exts.append(ext_9th)
        if ext_11th != 'N':
            paren_exts.append(ext_11th)
        if ext_13th != 'N':
            paren_exts.append(ext_13th)
        if paren_exts:
            chord += f"({','.join(paren_exts)})"
        
        # Add bass note if different from root
        if bass != 'N' and bass != root:
            chord += f"/{bass}"
        
        return chord
    
    def reassemble_with_confidence(self, components: Dict[str, str], 
                                   confidences: Dict[str, float]) -> Tuple[str, float]:
        """
        Reassemble chord with overall confidence score.
        
        Args:
            components: Dictionary with 9 component keys
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
