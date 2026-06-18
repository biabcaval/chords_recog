# encoding: utf-8
"""
Dataset / dataloader for training the BEATs chord decomposer on pre-extracted
embeddings.

Each ``.pt`` file (produced by ``scripts/preextract_beats_embeddings.py``)
holds the frame-level BEATs embeddings of one audio segment plus per-patch
chord labels already resampled to the embedding's temporal length. Training
therefore never touches the heavy BEATs backbone -- it only loads embeddings
and trains the lightweight multi-head classifier.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sortedcontainers import SortedList

from utils.chord_decomposition import ChordDecomposer, COMPONENT_NAMES
from data.audio_dataset_structured import get_component_vocab_sizes  # re-exported

__all__ = [
    "BEATsEmbeddingDataset",
    "BEATsDataLoader",
    "resample_sequence",
    "get_component_vocab_sizes",
]


def resample_sequence(seq, target_len):
    """Resample a 1D sequence of per-frame labels to ``target_len`` by nearest
    neighbour in time (repeat-or-subsample).

    Robust to off-by-one mismatches: any source length maps onto any target
    length without raising. Returns a plain Python list so it works for both
    string labels (chord labels) and integer labels (root/quality/bass).
    """
    src_len = len(seq)
    if src_len == 0:
        return [("N" if isinstance(seq, list) else 0)] * target_len
    if src_len == target_len:
        return list(seq)
    out = []
    for i in range(target_len):
        # Map target index to the centre of the corresponding source bin.
        src_idx = int(round((i + 0.5) * src_len / target_len - 0.5))
        src_idx = min(max(src_idx, 0), src_len - 1)
        out.append(seq[src_idx])
    return out


class BEATsEmbeddingDataset(Dataset):
    """Loads pre-extracted BEATs embeddings + decomposed chord labels.

    Two construction modes:

    1. Discovery mode (mirrors ``AudioDataset.get_paths_voca`` / kfold logic)::

           BEATsEmbeddingDataset(root_dir=..., dataset_names=(...),
                                 train=True, kfold=4, beats_tag='beats_iter3_plus',
                                 mp3_string='22050_10.0_5.0')

    2. Explicit-paths mode (handy for tests)::

           BEATsEmbeddingDataset(paths=[...])
    """

    def __init__(self, root_dir=None, dataset_names=("billboard",), train=False,
                 kfold=4, beats_tag="beats_iter3_plus", mp3_string="22050_10.0_5.0",
                 paths=None, decompose=True, max_patches=None):
        super().__init__()
        self.train = train
        self.decompose = decompose
        self.decomposer = ChordDecomposer() if decompose else None
        self.component_names = COMPONENT_NAMES
        self.max_patches = max_patches

        if paths is not None:
            self.paths = list(paths)
            self.song_names = []
        else:
            if root_dir is None:
                raise ValueError("Provide either 'paths' or 'root_dir'.")
            self.root_dir = root_dir
            self.dataset_names = dataset_names
            self.beats_tag = beats_tag
            self.mp3_string = mp3_string
            self.song_names, self.paths = self._discover_paths(kfold)

    # ------------------------------------------------------------------
    # Path discovery (mirrors data/audio_dataset.py fold handling)
    # ------------------------------------------------------------------
    def _result_dir(self, name):
        return os.path.join(self.root_dir, "result_beats", name + "_beats",
                            self.mp3_string, self.beats_tag)

    @staticmethod
    def _load_fold_assignments(root_dir):
        path = os.path.join(root_dir, "fold_assignments.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Using stratified fold assignments from {path}")
        return data.get("songs", {})

    def _split_by_fold(self, song_names, temp, kfold, fold_map=None):
        total_fold = 5
        if fold_map is not None:
            result, selected = [], []
            for s in song_names:
                in_val = fold_map.get(s, -1) == kfold
                if self.train and not in_val:
                    result += temp[s]
                    selected.append(s)
                elif (not self.train) and in_val:
                    # Validation: only the unshifted (shift 0) instances.
                    result += [inst for inst in temp[s] if "1.00_0" in inst]
                    selected.append(s)
            return selected, result

        quotient = len(song_names) // total_fold
        remainder = len(song_names) % total_fold
        fold_num = [0]
        for _ in range(total_fold):
            fold_num.append(quotient)
        for i in range(remainder):
            fold_num[i + 1] += 1
        for i in range(total_fold):
            fold_num[i + 1] += fold_num[i]

        result = []
        if self.train:
            tmp = []
            for k in range(total_fold):
                if k != kfold:
                    for i in range(fold_num[k], fold_num[k + 1]):
                        result += temp[song_names[i]]
                    tmp += song_names[fold_num[k]:fold_num[k + 1]]
            return tmp, result
        for i in range(fold_num[kfold], fold_num[kfold + 1]):
            result += [inst for inst in temp[song_names[i]] if "1.00_0" in inst]
        return song_names[fold_num[kfold]:fold_num[kfold + 1]], result

    def _discover_paths(self, kfold):
        temp = {}
        used_song_names = []
        for name in self.dataset_names:
            dataset_path = self._result_dir(name)
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(
                    f"BEATs embedding directory not found: {dataset_path}\n"
                    f"Pre-extract embeddings first with "
                    f"scripts/preextract_beats_embeddings.py for dataset '{name}'."
                )
            for song_name in os.listdir(dataset_path):
                song_dir = os.path.join(dataset_path, song_name)
                if not os.path.isdir(song_dir):
                    continue
                instance_names = os.listdir(song_dir)
                if instance_names:
                    used_song_names.append(song_name)
                temp[song_name] = [os.path.join(song_dir, n) for n in instance_names]

        song_names = SortedList(used_song_names)
        print("Total used song length : %d" % len(song_names))
        total = sum(len(temp[s]) for s in song_names)
        print("Total instances (train and valid) : %d" % total)

        fold_map = self._load_fold_assignments(self.root_dir)
        return self._split_by_fold(song_names, temp, kfold, fold_map)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.paths)

    def _get_label_list(self, data):
        if "original_chord_labels" in data:
            return list(data["original_chord_labels"])
        if "chord" in data:
            return list(data["chord"])
        return []

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx], weights_only=False)

        embedding = data["embedding"]
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(np.asarray(embedding), dtype=torch.float32)
        else:
            embedding = embedding.float()

        n_patches = embedding.shape[0]
        if self.max_patches is not None and n_patches > self.max_patches:
            embedding = embedding[: self.max_patches]
            n_patches = self.max_patches

        res = {"embedding": embedding}

        chord_labels = self._get_label_list(data)
        # Embeddings and labels should already match; guard against off-by-one.
        if len(chord_labels) != n_patches:
            chord_labels = resample_sequence(chord_labels, n_patches)
        res["chord"] = chord_labels

        if self.decompose and self.decomposer is not None:
            components = self.decomposer.decompose_batch(chord_labels)
            for component in COMPONENT_NAMES:
                comp = components[component]
                if isinstance(comp, np.ndarray):
                    comp = torch.as_tensor(comp, dtype=torch.long)
                components[component] = comp
            res["components"] = components

        return res


def _collate_fn_beats(batch):
    """Collate pre-extracted embeddings + decomposed labels.

    Returns:
        dict with
          - ``embeddings``: ``(batch, max_patches, embed_dim)`` (zero-padded)
          - ``components``: dict of ``(batch, max_patches)`` long tensors
          - ``lengths``: ``(batch,)`` true patch counts before padding
    """
    batch_size = len(batch)
    embed_dim = batch[0]["embedding"].shape[1]
    lengths = torch.tensor([b["embedding"].shape[0] for b in batch], dtype=torch.long)
    max_patches = int(lengths.max().item())

    embeddings = torch.zeros(batch_size, max_patches, embed_dim, dtype=torch.float32)
    components = {c: torch.zeros(batch_size, max_patches, dtype=torch.long)
                 for c in COMPONENT_NAMES}

    has_components = "components" in batch[0]
    for i, sample in enumerate(batch):
        emb = sample["embedding"]
        n = emb.shape[0]
        embeddings[i, :n] = emb
        if has_components:
            for c in COMPONENT_NAMES:
                comp = sample["components"][c]
                m = min(len(comp), max_patches)
                components[c][i, :m] = comp[:m]

    return {
        "embeddings": embeddings,
        "components": components,
        "lengths": lengths,
    }


class BEATsDataLoader(DataLoader):
    """DataLoader using the BEATs embedding collate function."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_beats
