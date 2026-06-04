import json
import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from utils.preprocess import Preprocess, FeatureTypes, cqt_to_log_db
import math
from multiprocessing import Pool
from sortedcontainers import SortedList

# Substring identifying the non-augmented (original) instance file.
# Files are named like "1.00_0_<idx>.pt" where 1.00 is the time-stretch factor
# and 0 is the pitch-shift offset in semitones. See scripts/preprocess_with_extensions.py.
_NON_AUG_TAG = "1.00_0"

# Valid split roles for the paper-faithful 60/20/20 rotation.
_VALID_SPLITS = (None, "train", "val", "test")


class AudioDataset(Dataset):
    def __init__(self, config, root_dir='/data/music/chord_recognition', dataset_names=('isophonic',),
                 featuretype=FeatureTypes.cqt, num_workers=20, train=False, preprocessing=False,
                 resize=None, kfold=4, split=None):
        """
        Args:
            split: One of None, 'train', 'val', 'test'.
                - None (default): legacy 80/20 mode. `train=True` returns the 4
                  folds different from `kfold` (with augmentations); `train=False`
                  returns fold `kfold` as validation (non-augmented only).
                - 'train' | 'val' | 'test': paper-faithful 60/20/20 rotation.
                  test_fold = kfold % 5, val_fold = (kfold + 1) % 5,
                  train_folds = the 3 remaining folds. Augmentations are kept
                  only in 'train'; 'val' and 'test' use the original instance
                  (filename containing "1.00_0") only.
                In paper mode the `train` flag is ignored; split alone controls
                the role of this dataset instance.
        """
        super(AudioDataset, self).__init__()

        if split not in _VALID_SPLITS:
            raise ValueError(
                f"AudioDataset: invalid split={split!r}. Expected one of {_VALID_SPLITS}."
            )

        self.config = config
        self.root_dir = root_dir
        self.dataset_names = dataset_names
        self.preprocessor = Preprocess(config, featuretype, dataset_names, self.root_dir)
        self.resize = resize
        self.train = train
        self.split = split
        self.ratio = config.experiment['data_ratio']

        # preprocessing hyperparameters
        # song_hz, n_bins, bins_per_octave, hop_length
        mp3_config = config.mp3
        feature_config = config.feature
        self.mp3_string = "%d_%.1f_%.1f" % \
                          (mp3_config['song_hz'], mp3_config['inst_len'],
                           mp3_config['skip_interval'])
        self.feature_string = "%s_%d_%d_%d" % \
                              (featuretype.value, feature_config['n_bins'], feature_config['bins_per_octave'], feature_config['hop_length'])

        if feature_config['large_voca'] == True:
            # store paths if exists
            is_preprocessed = True if os.path.exists(os.path.join(root_dir, 'result', dataset_names[0]+'_voca', self.mp3_string, self.feature_string)) else False
            if (not is_preprocessed) | preprocessing:
                midi_paths = self.preprocessor.get_all_files()

                if num_workers > 1:
                    num_path_per_process = math.ceil(len(midi_paths) / num_workers)
                    args = [midi_paths[i * num_path_per_process:(i + 1) * num_path_per_process] for i in range(num_workers)]

                    # start process
                    p = Pool(processes=num_workers)
                    p.map(self.preprocessor.generate_labels_features_voca, args)

                    p.close()
                else:
                    self.preprocessor.generate_labels_features_voca(midi_paths)

            # kfold is 5 fold index ( 0, 1, 2, 3, 4 )
            self.song_names, self.paths = self.get_paths_voca(kfold=kfold)
        else:
            # store paths if exists
            is_preprocessed = True if os.path.exists(os.path.join(root_dir, 'result', dataset_names[0], self.mp3_string, self.feature_string)) else False
            if (not is_preprocessed) | preprocessing:
                midi_paths = self.preprocessor.get_all_files()

                if num_workers > 1:
                    num_path_per_process = math.ceil(len(midi_paths) / num_workers)
                    args = [midi_paths[i * num_path_per_process:(i + 1) * num_path_per_process]
                            for i in range(num_workers)]

                    # start process
                    p = Pool(processes=num_workers)
                    p.map(self.preprocessor.generate_labels_features_new, args)

                    p.close()
                else:
                    self.preprocessor.generate_labels_features_new(midi_paths)

            # kfold is 5 fold index ( 0, 1, 2, 3, 4 )
            self.song_names, self.paths = self.get_paths(kfold=kfold)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        instance_path = self.paths[idx]

        res = dict()
        data = torch.load(instance_path)
        res['feature'] = cqt_to_log_db(data['feature'])
        res['chord'] = data['chord']
        
        # Load structured targets if available (for large_voca mode)
        if 'root' in data:
            res['root'] = data['root']
            res['quality'] = data['quality']
            res['bass'] = data['bass']
        
        return res

    @staticmethod
    def _load_fold_assignments(root_dir):
        """Load stratified fold assignments if available."""
        path = os.path.join(root_dir, "fold_assignments.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Using stratified fold assignments from {path}")
        return data.get("songs", {})

    def _resolve_split(self, kfold):
        """Map (self.split, self.train, kfold) onto the role of each fold.

        Returns:
            (train_folds, eval_fold)

            - train_folds: list of fold indices whose instances are concatenated
              (with augmentations) when assembling the dataset.
            - eval_fold: index of the single fold used as evaluation set
              (non-augmented only), or -1 if this dataset plays a train role.

        Paper-faithful 60/20/20 rotation (split in {'train','val','test'}):
            test_fold  = kfold % 5
            val_fold   = (kfold + 1) % 5
            train_folds = the 3 remaining folds

        Legacy 80/20 (split is None):
            train -> 4 folds different from `kfold`
            !train -> fold `kfold` as validation
        """
        total_fold = 5
        if self.split is None:
            # Legacy 80/20: train uses 4 folds, eval uses fold `kfold`.
            if self.train:
                return [k for k in range(total_fold) if k != kfold], -1
            return [], kfold

        # 60/20/20 rotation:
        #   test = fold kfold
        #   val  = fold (kfold + 1) % 5
        #   train = the 3 remaining folds
        test_fold = kfold % total_fold
        val_fold = (kfold + 1) % total_fold

        if self.split == "train":
            return [k for k in range(total_fold)
                    if k != test_fold and k != val_fold], -1
        if self.split == "val":
            return [], val_fold
        # split == "test"
        return [], test_fold

    def _split_by_fold(self, song_names, temp, kfold, fold_map=None):
        """Split songs into train/val[/test] by fold index.

        Behavior:
            - Paper mode (self.split in {'train','val','test'}): always uses the
              deterministic contiguous-block split derived from the lexicographic
              ordering of song_names. `fold_map` is ignored — paper-faithful
              splitting does not depend on external assignments.
            - Legacy mode (self.split is None): if `fold_map` is provided
              (loaded from fold_assignments.json), uses stratified per-song
              mapping. Otherwise falls back to the contiguous-block 80/20 split.
        """
        total_fold = 5

        # ---- Legacy stratified path (fold_map only honored when split is None) ----
        if self.split is None and fold_map is not None:
            if self.train:
                result = []
                selected = []
                for s in song_names:
                    if fold_map.get(s, -1) != kfold:
                        result += temp[s]
                        selected.append(s)
                return selected, result
            else:
                result = []
                selected = []
                for s in song_names:
                    if fold_map.get(s, -1) == kfold:
                        instances = [inst for inst in temp[s] if _NON_AUG_TAG in inst]
                        result += instances
                        selected.append(s)
                return selected, result

        # ---- Contiguous-block split (default + paper-faithful) ----
        # Build fold boundaries (5 folds; remainders go to the first folds).
        quotient = len(song_names) // total_fold
        remainder = len(song_names) % total_fold
        fold_num = [0]
        for _ in range(total_fold):
            fold_num.append(quotient)
        for i in range(remainder):
            fold_num[i + 1] += 1
        for i in range(total_fold):
            fold_num[i + 1] += fold_num[i]

        train_folds, eval_fold = self._resolve_split(kfold)

        result = []
        if eval_fold == -1:
            # Train role: concatenate the requested folds, keep augmentations.
            tmp = []
            for k in train_folds:
                for i in range(fold_num[k], fold_num[k + 1]):
                    result += temp[song_names[i]]
                tmp += song_names[fold_num[k]:fold_num[k + 1]]
            return tmp, result

        # Eval role (val or test, or legacy validation): single fold, originals only.
        for i in range(fold_num[eval_fold], fold_num[eval_fold + 1]):
            instances = temp[song_names[i]]
            instances = [inst for inst in instances if _NON_AUG_TAG in inst]
            result += instances
        return song_names[fold_num[eval_fold]:fold_num[eval_fold + 1]], result

    def get_paths(self, kfold=4):
        temp = {}
        used_song_names = list()
        for name in self.dataset_names:
            dataset_path = os.path.join(self.root_dir, "result", name, self.mp3_string, self.feature_string)
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(
                    f"Dataset directory not found: {dataset_path}\n"
                    f"Please ensure the dataset '{name}' has been preprocessed.\n"
                    f"Expected path: {dataset_path}"
                )
            song_names = os.listdir(dataset_path)
            for song_name in song_names:
                paths = []
                instance_names = os.listdir(os.path.join(dataset_path, song_name))
                if len(instance_names) > 0:
                    used_song_names.append(song_name)
                for instance_name in instance_names:
                    paths.append(os.path.join(dataset_path, song_name, instance_name))
                temp[song_name] = paths

        song_names = SortedList(used_song_names)

        print('Total used song length : %d' %len(song_names))
        tmp = []
        for i in range(len(song_names)):
            tmp += temp[song_names[i]]
        print('Total instances (train and valid) : %d' %len(tmp))

        fold_map = self._load_fold_assignments(self.root_dir)
        return self._split_by_fold(song_names, temp, kfold, fold_map)

    def get_paths_voca(self, kfold=4):
        temp = {}
        used_song_names = list()
        for name in self.dataset_names:
            dataset_path = os.path.join(self.root_dir, "result_decomposed", name+'_voca', self.mp3_string, self.feature_string)
            if not os.path.exists(dataset_path):
                dataset_path = os.path.join(self.root_dir, "result", name+'_voca', self.mp3_string, self.feature_string)
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(
                    f"Dataset directory not found: {dataset_path}\n"
                    f"Please ensure the dataset '{name}' has been preprocessed with large_voca=True.\n"
                    f"Expected path: {dataset_path}"
                )
            song_names = os.listdir(dataset_path)
            for song_name in song_names:
                paths = []
                instance_names = os.listdir(os.path.join(dataset_path, song_name))
                if len(instance_names) > 0:
                    used_song_names.append(song_name)
                for instance_name in instance_names:
                    paths.append(os.path.join(dataset_path, song_name, instance_name))
                temp[song_name] = paths

        song_names = SortedList(used_song_names)

        print('Total used song length : %d' %len(song_names))
        tmp = []
        for i in range(len(song_names)):
            tmp += temp[song_names[i]]
        print('Total instances (train and valid) : %d' %len(tmp))

        fold_map = self._load_fold_assignments(self.root_dir)
        return self._split_by_fold(song_names, temp, kfold, fold_map)

def _collate_fn(batch):
    batch_size = len(batch)
    max_len = batch[0]['feature'].shape[1]

    input_percentages = torch.empty(batch_size)  # for variable length
    chord_lens = torch.empty(batch_size, dtype=torch.int64)
    chords = []
    collapsed_chords = []
    features = []
    boundaries = []
    roots = []
    qualities = []
    basses = []
    
    # Check if structured targets are available
    has_structured = 'root' in batch[0]
    
    for i in range(batch_size):
        sample = batch[i]
        feature = sample['feature']
        chord = sample['chord']
        diff = np.diff(chord, axis=0).astype(bool)
        idx = np.insert(diff, 0, True, axis=0)
        chord_lens[i] = np.sum(idx).item(0)
        chords.extend(chord)
        features.append(feature)
        input_percentages[i] = feature.shape[1] / max_len
        collapsed_chords.extend(np.array(chord)[idx].tolist())
        boundary = np.append([0], diff)
        boundaries.extend(boundary.tolist())
        
        # Collect structured targets if available
        if has_structured:
            roots.extend(sample['root'])
            qualities.extend(sample['quality'])
            basses.extend(sample['bass'])

    features = torch.tensor(features, dtype=torch.float32).unsqueeze(1)  # batch_size*1*feature_size*max_len
    chords = torch.tensor(chords, dtype=torch.int64)  # (batch_size*time_length)
    collapsed_chords = torch.tensor(collapsed_chords, dtype=torch.int64)  # total_unique_chord_len
    boundaries = torch.tensor(boundaries, dtype=torch.uint8)  # (batch_size*time_length)
    
    if has_structured:
        roots = torch.tensor(roots, dtype=torch.int64)  # (batch_size*time_length)
        qualities = torch.tensor(qualities, dtype=torch.int64)  # (batch_size*time_length)
        basses = torch.tensor(basses, dtype=torch.int64)  # (batch_size*time_length)
        return features, input_percentages, chords, collapsed_chords, chord_lens, boundaries, roots, qualities, basses
    else:
        return features, input_percentages, chords, collapsed_chords, chord_lens, boundaries

class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class EpochSampler(Sampler):
    """Sample one random segment per song per epoch (paper-faithful).

    ChordFormer (Akram et al., 2026, IV-B) trains on a single random 1000-frame
    segment per song each epoch instead of iterating over every preprocessed
    instance. This sampler reproduces that behavior on top of a flat list of
    instance paths by grouping indices per song (via the parent directory of
    each path) and emitting one randomly-chosen index per song per epoch, then
    shuffling the resulting list.

    Use only with train datasets that contain augmented instances. Val/test
    must iterate naturally over the filtered originals.

    The sampler discovers paths through `dataset.paths` (works for AudioDataset
    and AudioDatasetStructured).
    """

    def __init__(self, dataset):
        self.dataset = dataset
        paths = self._collect_paths(dataset)

        self.song_to_indices = {}
        for idx, p in enumerate(paths):
            song_dir = os.path.basename(os.path.dirname(p))
            self.song_to_indices.setdefault(song_dir, []).append(idx)

        self.songs = list(self.song_to_indices.keys())

    @staticmethod
    def _collect_paths(dataset):
        if hasattr(dataset, "paths"):
            return list(dataset.paths)
        raise AttributeError(
            "EpochSampler expects `dataset` to expose a `.paths` list of "
            "instance file paths (AudioDataset / AudioDatasetStructured)."
        )

    def __iter__(self):
        indices = []
        for song in self.songs:
            idx = random.choice(self.song_to_indices[song])
            indices.append(idx)
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.songs)
