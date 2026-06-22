# encoding: utf-8
"""
Shared per-segment chord labelling for the BEATs pipeline.

Both the offline embedding pre-extraction
(``scripts/preextract_beats_embeddings.py``) and the end-to-end fine-tuning
dataset (``data/beats_audio_dataset.py``) need to turn a ``.lab`` annotation
into per-frame chord/root/quality/bass label lists for a fixed
``inst_len``-second window, applying the pitch-shift transposition. Keeping
that logic here guarantees the embedding cache and the fine-tuning path stay
label-compatible.
"""

# These sentinel ids mirror Preprocess.generate_labels_features_voca:
#   chord 169 -> "N" (no chord), 168 -> "X" (unknown); root/bass 12 -> none;
#   quality 14 -> none. They must not be transposed.
_NO_CHORD_ID = 169
_UNKNOWN_CHORD_ID = 168
_NONE_PITCH = 12
_NONE_QUALITY = 14


def build_segment_labels(preprocessor, chord_info, inst_start_sec, shift_factor):
    """Build per-frame label lists for one window at the CQT frame rate.

    Replicates the labelling logic of
    ``Preprocess.generate_labels_features_voca`` (chord_id/root/quality/bass +
    full label string), applying the pitch-shift transposition.

    Unlike the previous inlined version, this NEVER returns ``None``: it always
    returns whatever frames it could resolve for the window. Callers that need a
    strict length (e.g. the embedding cache) should compare ``len(label_list)``
    against ``preprocessor.no_of_chord_datapoints_per_sequence`` themselves; the
    fine-tuning dataset instead resamples to the backbone patch count, which is
    robust to off-by-one boundary windows.

    Args:
        preprocessor: A ``Preprocess`` instance (provides ``time_interval``,
            ``config.mp3`` and ``Chord_class``).
        chord_info: DataFrame from ``Chord_class.get_converted_chord_full``.
        inst_start_sec: Window start in seconds.
        shift_factor: Pitch-shift in semitones to transpose labels by.

    Returns:
        Tuple ``(chord_list, root_list, quality_list, bass_list, label_list)``
        of equal length (one entry per CQT frame in the window).
    """
    time_interval = preprocessor.time_interval
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
            chord, root, quality, bass, chord_label = (
                _NO_CHORD_ID, _NONE_PITCH, _NONE_QUALITY, _NONE_PITCH, "N")

        if chord != _NO_CHORD_ID and chord != _UNKNOWN_CHORD_ID:
            chord = (chord + shift_factor * 14) % 168
        if root != _NONE_PITCH:
            root = (root + shift_factor) % 12
        if bass != _NONE_PITCH:
            bass = (bass + shift_factor) % 12
        chord_label = preprocessor.Chord_class.transpose_chord_label(chord_label, shift_factor)

        chord_list.append(int(chord))
        root_list.append(int(root))
        quality_list.append(int(quality))
        bass_list.append(int(bass))
        label_list.append(chord_label)
        cur_sec += time_interval

    return chord_list, root_list, quality_list, bass_list, label_list
