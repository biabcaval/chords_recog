import json
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.preprocess import Preprocess, FeatureTypes, cqt_to_log_db
import math
from multiprocessing import Pool
from sortedcontainers import SortedList

class AudioDataset(Dataset):
    def __init__(self, config, root_dir='/data/music/chord_recognition', dataset_names=('isophonic',),
                 featuretype=FeatureTypes.cqt, num_workers=20, train=False, preprocessing=False, resize=None, kfold=4):
        super(AudioDataset, self).__init__()

        self.config = config
        self.root_dir = root_dir
        self.dataset_names = dataset_names
        self.preprocessor = Preprocess(config, featuretype, dataset_names, self.root_dir)
        self.resize = resize
        self.train = train
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

    def _split_by_fold(self, song_names, temp, kfold, fold_map=None):
        """Split songs into train/val by fold index.

        If *fold_map* is provided (dict song_name -> fold_index from
        fold_assignments.json) it is used for stratified splitting.
        Otherwise falls back to the original contiguous-block method.
        """
        total_fold = 5

        if fold_map is not None:
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
                        instances = [inst for inst in temp[s] if "1.00_0" in inst]
                        result += instances
                        selected.append(s)
                return selected, result

        # Fallback: original contiguous-block split
        quotient = len(song_names) // total_fold
        remainder = len(song_names) % total_fold
        fold_num = [0]
        for i in range(total_fold):
            fold_num.append(quotient)
        for i in range(remainder):
            fold_num[i+1] += 1
        for i in range(total_fold):
            fold_num[i+1] += fold_num[i]

        result = []
        if self.train:
            tmp = []
            for k in range(total_fold):
                if k != kfold:
                    for i in range(fold_num[k], fold_num[k+1]):
                        result += temp[song_names[i]]
                    tmp += song_names[fold_num[k]:fold_num[k + 1]]
            return tmp, result
        else:
            for i in range(fold_num[kfold], fold_num[kfold+1]):
                instances = temp[song_names[i]]
                instances = [inst for inst in instances if "1.00_0" in inst]
                result += instances
            return song_names[fold_num[kfold]:fold_num[kfold+1]], result

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
