# encoding: utf-8
"""
Raw-waveform dataset for END-TO-END fine-tuning of the BEATs backbone.

Unlike :class:`data.beats_dataset.BEATsEmbeddingDataset` (which serves frozen,
pre-extracted embeddings), this dataset yields 16 kHz mono *waveforms* so the
backbone runs inside the training loop and gradients can flow into the
unfrozen top layers. Per-segment chord labels are built with the SAME logic as
the embedding cache (``utils.beats_segment_labels.build_segment_labels``) and
resampled to the backbone's (fixed) patch count, so labels stay compatible.

Design notes / trade-offs:
  * Segments are fixed ``inst_len``-second windows, so the BEATs patch count is
    constant; the caller passes ``target_patches`` (obtained from a single
    backbone forward) and every label list is resampled to it.
  * Audio is decoded lazily per song and kept in a small per-worker LRU cache.
    With ``shuffle=True`` the hit rate is modest; keep ``num_workers`` modest
    and ``audio_cache_size`` a handful of songs. This is the price of feeding
    raw audio instead of cached embeddings.
  * ``labels_only`` mode skips audio decoding entirely so class-weight
    computation can iterate the dataset cheaply.

Validation note: this module imports ``torch`` and (lazily) ``librosa``; it is
exercised on the training VM, not the local dev box.
"""

import json
import os
import logging
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sortedcontainers import SortedList

from utils.preprocess import Preprocess, FeatureTypes
from utils.beats_segment_labels import build_segment_labels
from utils.decomposition_registry import get_decomposition
from data.beats_dataset import resample_sequence

logger = logging.getLogger(__name__)

TARGET_SR = 16000  # BEATs requires 16 kHz mono input.

__all__ = ["BEATsAudioDataset", "BEATsAudioDataLoader"]


class BEATsAudioDataset(Dataset):
    """Serve (waveform, decomposed-labels) pairs for end-to-end fine-tuning.

    Args:
        config: Loaded ``HParams`` config (provides ``mp3`` timing + dataset
            directory layout used by :class:`Preprocess`).
        data_root: Dataset root passed to :class:`Preprocess`.
        dataset_names: Tuple of dataset names (e.g. ``("billboard", "queen")``).
        train: If True, use the training folds; else the held-out fold (and only
            the unshifted instances, matching the embedding pipeline).
        kfold: Which fold (0..4) is held out for validation.
        target_patches: BEATs patch count per ``inst_len`` window. Labels are
            resampled to this length so they align with backbone output.
        decomposition: ``'paper6'`` (default) or ``'full9'``.
        audio_cache_size: Number of decoded songs kept in the per-worker LRU.
        sr: Target sample rate (BEATs is fixed at 16 kHz).
    """

    def __init__(self, config, data_root, dataset_names=("billboard",), train=False,
                 kfold=4, target_patches=None, decomposition="paper6",
                 audio_cache_size=8, sr=TARGET_SR):
        super().__init__()
        if target_patches is None or target_patches <= 0:
            raise ValueError("target_patches must be a positive int (obtain it "
                             "from one backbone forward before building the dataset).")
        self.train = train
        self.sr = sr
        self.target_patches = int(target_patches)
        self.audio_cache_size = int(audio_cache_size)
        self.labels_only = False

        decomp = get_decomposition(decomposition)
        self.decomposition = decomp.scheme
        self.decomposer = decomp.ChordDecomposer()
        self.component_names = list(decomp.COMPONENT_NAMES)

        self.preprocessor = Preprocess(config, FeatureTypes.cqt, tuple(dataset_names), data_root)
        self.inst_len = float(config.mp3["inst_len"])
        self.skip_interval = float(config.mp3["skip_interval"])
        self.seg_samples = int(round(self.inst_len * sr))
        self.data_root = data_root

        self._audio_cache = OrderedDict()
        # Songs decoded up-front in the MAIN process (see prebuild_audio_cache).
        # Forked DataLoader workers inherit this read-only via copy-on-write and
        # therefore never call librosa themselves -- this both avoids the
        # librosa/numba fork deadlock and makes __getitem__ a cheap array slice.
        self._preloaded = {}      # song -> np.ndarray (full song @ sr)
        self.song_paths = {}      # song -> (lab_path, mp3_path)
        self.chord_infos = {}     # song -> DataFrame (cached lab parse)
        self.specs = []           # list of (song, start_sec)

        self._build_index(kfold)

    # ------------------------------------------------------------------
    # Index construction (song discovery + fold split + segment enumeration)
    # ------------------------------------------------------------------
    def _build_index(self, kfold):
        all_files = self.preprocessor.get_all_files()
        for entry in all_files:
            song_name = entry[0].strip()
            self.song_paths[song_name] = (entry[1], entry[2])

        song_names = SortedList(self.song_paths.keys())
        fold_map = self._load_fold_assignments(self.data_root)
        selected = self._select_songs(song_names, kfold, fold_map)

        kept = 0
        for song in selected:
            lab_path, _ = self.song_paths[song]
            try:
                chord_info = self.preprocessor.Chord_class.get_converted_chord_full(lab_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping '%s' (lab parse error: %s)", song, exc)
                continue
            if chord_info is None or len(chord_info) == 0:
                logger.warning("Skipping '%s' (empty chord_info).", song)
                continue
            self.chord_infos[song] = chord_info
            origin_len_sec = float(chord_info.iloc[-1]["end"])
            current_start = 0.0
            while current_start + self.inst_len < origin_len_sec:
                self.specs.append((song, current_start))
                current_start += self.skip_interval
                kept += 1

        logger.info("BEATsAudioDataset(%s): %d songs, %d segments (kfold=%d, target_patches=%d).",
                    "train" if self.train else "val", len(self.chord_infos), kept,
                    kfold, self.target_patches)

    @staticmethod
    def _load_fold_assignments(root_dir):
        path = os.path.join(root_dir, "fold_assignments.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Using stratified fold assignments from %s", path)
        return data.get("songs", {})

    def _select_songs(self, song_names, kfold, fold_map):
        """Return the songs belonging to the train (or val) split.

        Mirrors ``BEATsEmbeddingDataset._split_by_fold`` but at the song level
        (segments are enumerated on the fly here rather than read from disk).
        """
        total_fold = 5
        if fold_map is not None:
            selected = []
            for s in song_names:
                in_val = fold_map.get(s, -1) == kfold
                if self.train and not in_val:
                    selected.append(s)
                elif (not self.train) and in_val:
                    selected.append(s)
            return selected

        n = len(song_names)
        quotient = n // total_fold
        remainder = n % total_fold
        fold_num = [0]
        for _ in range(total_fold):
            fold_num.append(quotient)
        for i in range(remainder):
            fold_num[i + 1] += 1
        for i in range(total_fold):
            fold_num[i + 1] += fold_num[i]

        if self.train:
            selected = []
            for k in range(total_fold):
                if k != kfold:
                    selected += list(song_names[fold_num[k]:fold_num[k + 1]])
            return selected
        return list(song_names[fold_num[kfold]:fold_num[kfold + 1]])

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.specs)

    def _build_components(self, song, start_sec):
        chord_info = self.chord_infos[song]
        labels = build_segment_labels(self.preprocessor, chord_info, start_sec, 0)
        label_list = labels[4]
        label_list = resample_sequence(label_list, self.target_patches)
        components = self.decomposer.decompose_batch(label_list)
        for component in self.component_names:
            comp = components[component]
            if isinstance(comp, np.ndarray):
                comp = torch.as_tensor(comp, dtype=torch.long)
            elif not isinstance(comp, torch.Tensor):
                comp = torch.as_tensor(np.asarray(comp), dtype=torch.long)
            components[component] = comp.long()
        return components, label_list

    def prebuild_audio_cache(self, max_workers=8):
        """Decode every needed song to 16 kHz mono in the MAIN process (once).

        This must run BEFORE the DataLoader spawns workers. Decoding here (not in
        forked workers) sidesteps the librosa/numba fork deadlock, and the
        resulting arrays are shared copy-on-write with workers, so per-segment
        access becomes an in-RAM slice. Uses a thread pool because librosa's MP3
        decode is largely I/O / ffmpeg-subprocess bound.

        Args:
            max_workers: Thread pool size for parallel decoding.
        """
        songs = sorted({song for song, _ in self.specs})
        todo = [s for s in songs if s not in self._preloaded]
        if not todo:
            return
        import librosa  # main-process import only

        split = "train" if self.train else "val"
        logger.info("Pre-decoding %d %s songs to %dkHz in RAM (one-time)...",
                    len(todo), split, self.sr // 1000)
        start = time.time()
        done = 0

        def _decode(song):
            _, mp3_path = self.song_paths[song]
            wav, _ = librosa.load(mp3_path, sr=self.sr, mono=True)
            return song, np.asarray(wav, dtype=np.float32)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for song, wav in pool.map(_decode, todo):
                self._preloaded[song] = wav
                done += 1
                if done % 100 == 0 or done == len(todo):
                    gb = sum(a.nbytes for a in self._preloaded.values()) / 1e9
                    logger.info("  decoded %d/%d %s songs (%.1f GB, %.0fs elapsed)",
                                done, len(todo), split, gb, time.time() - start)

        gb = sum(a.nbytes for a in self._preloaded.values()) / 1e9
        logger.info("Pre-decode complete: %d %s songs, %.1f GB RAM, %.0fs.",
                    len(self._preloaded), split, gb, time.time() - start)

    def _song_audio(self, mp3_path):
        """Decode a full song to 16 kHz mono with a small per-worker LRU cache."""
        cached = self._audio_cache.get(mp3_path)
        if cached is not None:
            self._audio_cache.move_to_end(mp3_path)
            return cached
        import librosa  # lazy: only needed when actually serving audio
        wav, _ = librosa.load(mp3_path, sr=self.sr, mono=True)
        self._audio_cache[mp3_path] = wav
        if len(self._audio_cache) > self.audio_cache_size:
            self._audio_cache.popitem(last=False)
        return wav

    def _segment_waveform(self, song, start_sec):
        # Prefer the RAM-preloaded song (no librosa in workers); fall back to the
        # lazy per-worker LRU decode only if prebuild_audio_cache wasn't run.
        wav = self._preloaded.get(song)
        if wav is None:
            _, mp3_path = self.song_paths[song]
            wav = self._song_audio(mp3_path)
        start = int(round(start_sec * self.sr))
        seg = wav[start:start + self.seg_samples]
        out = np.zeros(self.seg_samples, dtype=np.float32)
        m = min(len(seg), self.seg_samples)
        if m > 0:
            out[:m] = seg[:m]
        return torch.from_numpy(out)

    def __getitem__(self, idx):
        song, start_sec = self.specs[idx]
        components, label_list = self._build_components(song, start_sec)
        res = {"components": components, "chord": label_list}
        if not self.labels_only:
            res["waveform"] = self._segment_waveform(song, start_sec)
        return res


def _collate_fn_beats_audio(batch):
    """Collate raw waveforms + decomposed labels for fine-tuning.

    Returns:
        dict with
          - ``waveforms``: ``(batch, samples)`` float tensor (present unless the
            dataset is in ``labels_only`` mode)
          - ``components``: dict of ``(batch, target_patches)`` long tensors
          - ``lengths``: ``(batch,)`` patch counts (constant = target_patches)
    """
    component_names = list(batch[0]["components"].keys())
    target_patches = int(batch[0]["components"][component_names[0]].shape[0])
    batch_size = len(batch)

    components = {c: torch.zeros(batch_size, target_patches, dtype=torch.long)
                 for c in component_names}
    for i, sample in enumerate(batch):
        for c in component_names:
            comp = sample["components"][c]
            m = min(int(comp.shape[0]), target_patches)
            components[c][i, :m] = comp[:m]

    out = {
        "components": components,
        "lengths": torch.full((batch_size,), target_patches, dtype=torch.long),
    }

    if "waveform" in batch[0]:
        samples = int(batch[0]["waveform"].shape[0])
        waveforms = torch.zeros(batch_size, samples, dtype=torch.float32)
        for i, sample in enumerate(batch):
            w = sample["waveform"]
            m = min(int(w.shape[0]), samples)
            waveforms[i, :m] = w[:m]
        out["waveforms"] = waveforms

    return out


class BEATsAudioDataLoader(DataLoader):
    """DataLoader using the raw-waveform collate function."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_beats_audio
