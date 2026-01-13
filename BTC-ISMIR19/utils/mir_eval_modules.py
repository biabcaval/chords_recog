import numpy as np
import librosa
import mir_eval
import torch
import os

idx2chord = ['C', 'C:min', 'C#', 'C#:min', 'D', 'D:min', 'D#', 'D#:min', 'E', 'E:min', 'F', 'F:min', 'F#',
             'F#:min', 'G', 'G:min', 'G#', 'G#:min', 'A', 'A:min', 'A#', 'A#:min', 'B', 'B:min', 'N']

root_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
quality_list = ['min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7', 'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4']


def compute_wcsr(correct_durations, total_durations):
    """
    Compute Weighted Chord Symbol Recall (WCSR) across all songs.
    
    WCSR = (Σ zi / Σ Zi) × 100
    
    Args:
        correct_durations: List of correctly predicted durations (zi) for each song
        total_durations: List of total durations (Zi) for each song
    
    Returns:
        float: WCSR score (0-100)
    """
    correct_durations = np.array(correct_durations)
    total_durations = np.array(total_durations)
    
    sum_correct = np.sum(correct_durations)
    sum_total = np.sum(total_durations)
    
    if sum_total == 0:
        return 0.0
    
    wcsr = (sum_correct / sum_total) * 100
    return wcsr


def get_song_durations(gt_path, est_path, comparison_func):
    """
    Compute correctly predicted duration (zi) and total duration (Zi) for a single song.
    
    Args:
        gt_path: Path to ground truth .lab file
        est_path: Path to estimated .lab file
        comparison_func: mir_eval comparison function (e.g., mir_eval.chord.root, mir_eval.chord.majmin)
    
    Returns:
        tuple: (correct_duration, total_duration)
    """
    (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
    ref_labels = lab_file_error_modify(ref_labels)
    (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
    
    est_intervals, est_labels = mir_eval.util.adjust_intervals(
        est_intervals, est_labels, 
        ref_intervals.min(), ref_intervals.max(), 
        mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD
    )
    
    (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels
    )
    
    durations = mir_eval.util.intervals_to_durations(intervals)
    comparisons = comparison_func(ref_labels, est_labels)
    
    # zi: sum of durations where comparison is True (correct predictions)
    correct_duration = np.sum(durations * comparisons)
    # Zi: total duration of the song
    total_duration = np.sum(durations)
    
    return correct_duration, total_duration

def idx2voca_chord():
    idx2voca_chord = {}
    idx2voca_chord[169] = 'N'
    idx2voca_chord[168] = 'X'
    for i in range(168):
        root = i // 14
        root = root_list[root]
        quality = i % 14
        quality = quality_list[quality]
        if i % 14 != 1:
            chord = root + ':' + quality
        else:
            chord = root
        idx2voca_chord[i] = chord
    return idx2voca_chord

def audio_file_to_features(audio_file, config):
    original_wav, sr = librosa.load(audio_file, sr=config.mp3['song_hz'], mono=True)
    currunt_sec_hz = 0
    while len(original_wav) > currunt_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len']:
        start_idx = int(currunt_sec_hz)
        end_idx = int(currunt_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len'])
        tmp = librosa.cqt(original_wav[start_idx:end_idx], sr=sr, n_bins=config.feature['n_bins'], bins_per_octave=config.feature['bins_per_octave'], hop_length=config.feature['hop_length'])
        if start_idx == 0:
            feature = tmp
        else:
            feature = np.concatenate((feature, tmp), axis=1)
        currunt_sec_hz = end_idx
    tmp = librosa.cqt(original_wav[currunt_sec_hz:], sr=sr, n_bins=config.feature['n_bins'], bins_per_octave=config.feature['bins_per_octave'], hop_length=config.feature['hop_length'])
    feature = np.concatenate((feature, tmp), axis=1)
    feature = np.log(np.abs(feature) + 1e-6)
    feature_per_second = config.mp3['inst_len'] / config.model['timestep']
    song_length_second = len(original_wav)/config.mp3['song_hz']
    return feature, feature_per_second, song_length_second

# Audio files with format of wav and mp3
def get_audio_paths(audio_dir):
    return [os.path.join(root, fname) for (root, dir_names, file_names) in os.walk(audio_dir, followlinks=True)
            for fname in file_names if (fname.lower().endswith('.wav') or fname.lower().endswith('.mp3'))]

class metrics():
    def __init__(self):
        super(metrics, self).__init__()
        self.score_metrics = ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']
        self.score_list_dict = dict()
        # Track correct and total durations for WCSR computation
        self.correct_durations_dict = dict()
        self.total_durations_dict = dict()
        self.wcsr_dict = dict()
        
        for i in self.score_metrics:
            self.score_list_dict[i] = list()
            self.correct_durations_dict[i] = list()
            self.total_durations_dict[i] = list()
        self.average_score = dict()

    def score(self, metric, gt_path, est_path):
        if metric == 'root':
            score = self.root_score(gt_path,est_path)
        elif metric == 'thirds':
            score = self.thirds_score(gt_path,est_path)
        elif metric == 'triads':
            score = self.triads_score(gt_path,est_path)
        elif metric == 'sevenths':
            score = self.sevenths_score(gt_path,est_path)
        elif metric == 'tetrads':
            score = self.tetrads_score(gt_path,est_path)
        elif metric == 'majmin':
            score = self.majmin_score(gt_path,est_path)
        elif metric == 'mirex':
            score = self.mirex_score(gt_path,est_path)
        else:
            raise NotImplementedError
        return score
    
    def score_with_durations(self, metric, gt_path, est_path):
        """
        Compute score and return both score and duration info for WCSR.
        
        Returns:
            tuple: (score, correct_duration, total_duration)
        """
        comparison_func = self._get_comparison_func(metric)
        correct_dur, total_dur = get_song_durations(gt_path, est_path, comparison_func)
        score = correct_dur / total_dur if total_dur > 0 else 0.0
        return score, correct_dur, total_dur
    
    def _get_comparison_func(self, metric):
        """Get the mir_eval comparison function for a given metric."""
        func_map = {
            'root': mir_eval.chord.root,
            'thirds': mir_eval.chord.thirds,
            'triads': mir_eval.chord.triads,
            'sevenths': mir_eval.chord.sevenths,
            'tetrads': mir_eval.chord.tetrads,
            'majmin': mir_eval.chord.majmin,
            'mirex': mir_eval.chord.mirex,
        }
        return func_map[metric]
    
    def compute_wcsr(self, metric):
        """Compute WCSR for a specific metric using accumulated durations."""
        return compute_wcsr(
            self.correct_durations_dict[metric],
            self.total_durations_dict[metric]
        )

    def root_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.root(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

    def thirds_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.thirds(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

    def triads_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.triads(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

    def sevenths_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.sevenths(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

    def tetrads_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.tetrads(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

    def majmin_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.majmin(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

    def mirex_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.mirex(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

def lab_file_error_modify(ref_labels):
    for i in range(len(ref_labels)):
        if ref_labels[i][-2:] == ':4':
            ref_labels[i] = ref_labels[i].replace(':4', ':sus4')
        elif ref_labels[i][-2:] == ':6':
            ref_labels[i] = ref_labels[i].replace(':6', ':maj6')
        elif ref_labels[i][-4:] == ':6/2':
            ref_labels[i] = ref_labels[i].replace(':6/2', ':maj6/2')
        elif ref_labels[i] == 'Emin/4':
            ref_labels[i] = 'E:min/4'
        elif ref_labels[i] == 'A7/3':
            ref_labels[i] = 'A:7/3'
        elif ref_labels[i] == 'Bb7/3':
            ref_labels[i] = 'Bb:7/3'
        elif ref_labels[i] == 'Bb7/5':
            ref_labels[i] = 'Bb:7/5'
        elif ref_labels[i].find(':') == -1:
            if ref_labels[i].find('min') != -1:
                ref_labels[i] = ref_labels[i][:ref_labels[i].find('min')] + ':' + ref_labels[i][ref_labels[i].find('min'):]
    return ref_labels

def root_majmin_score_calculation(valid_dataset, config, mean, std, device, model, model_type, verbose=False):
    valid_song_names = valid_dataset.song_names
    # Normalize valid song names for comparison (lowercase, replace spaces with underscores)
    valid_song_names_normalized = {name.lower().replace(' ', '_'): name for name in valid_song_names}
    paths = valid_dataset.preprocessor.get_all_files()

    metrics_ = metrics()
    song_length_list = list()
    root_majmin = ['root', 'majmin']  # Define outside try block to avoid UnboundLocalError
    
    for path in paths:
        song_name, lab_file_path, mp3_file_path, _ = path
        # Normalize song_name for comparison
        song_name_normalized = song_name.lower().replace(' ', '_')
        if song_name_normalized not in valid_song_names_normalized:
            continue
        try:
            n_timestep = config.model['timestep']
            feature, feature_per_second, song_length_second = audio_file_to_features(mp3_file_path, config)
            feature = feature.T
            feature = (feature - mean) / std
            time_unit = feature_per_second

            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
            num_instance = feature.shape[0] // n_timestep

            start_time = 0.0
            lines = []
            with torch.no_grad():
                model.eval()
                feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
                for t in range(num_instance):
                    if model_type == 'btc':
                        encoder_output, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
                        prediction, _ = model.output_layer(encoder_output)
                        prediction = prediction.squeeze()
                    elif model_type == 'btc_structured':
                        # Handle structured output model for small vocabulary
                        chunk = feature[:, n_timestep * t:n_timestep * (t + 1), :]
                        encoder_output, _ = model.self_attn_layers(chunk)
                        root_pred, quality_pred, bass_pred, _, _, _ = model.output_layer(encoder_output)
                        root_pred = root_pred.squeeze()  # [n_timestep]
                        quality_pred = quality_pred.squeeze()  # [n_timestep]
                        
                        # Convert to small vocabulary indices (25 classes)
                        # idx2chord: C, C:min, C#, C#:min, ..., B, B:min, N
                        # So for root r: major = r*2, minor = r*2+1, N = 24
                        prediction = torch.zeros_like(root_pred)
                        for idx in range(len(root_pred)):
                            r, q = root_pred[idx].item(), quality_pred[idx].item()
                            if r == 12 or q == 14:  # No chord
                                prediction[idx] = 24
                            elif q == 0:  # min
                                prediction[idx] = r * 2 + 1
                            elif q == 1:  # maj
                                prediction[idx] = r * 2
                            else:
                                # Other qualities -> map to major/minor based on quality
                                # Qualities 0,4,6,7 are minor-like; 1,3,5,8,9 are major-like
                                if q in [0, 4, 6, 7, 10, 11]:  # min, min6, min7, minmaj7, dim7, hdim7
                                    prediction[idx] = r * 2 + 1
                                else:  # maj, aug, maj6, maj7, 7, sus2, sus4, dim
                                    prediction[idx] = r * 2
                    elif model_type == 'chordformer':
                        # Handle ChordFormer model for small vocabulary
                        chunk = feature[:, n_timestep * t:n_timestep * (t + 1), :]
                        encoder_output, _ = model.conformer_encoder(chunk)
                        root_pred, quality_pred, bass_pred, _, _, _ = model.output_layer(encoder_output)
                        root_pred = root_pred.squeeze()  # [n_timestep]
                        quality_pred = quality_pred.squeeze()  # [n_timestep]
                        
                        # Convert to small vocabulary indices (25 classes)
                        prediction = torch.zeros_like(root_pred)
                        for idx in range(len(root_pred)):
                            r, q = root_pred[idx].item(), quality_pred[idx].item()
                            if r == 12 or q == 14:  # No chord
                                prediction[idx] = 24
                            elif q == 0:  # min
                                prediction[idx] = r * 2 + 1
                            elif q == 1:  # maj
                                prediction[idx] = r * 2
                            else:
                                if q in [0, 4, 6, 7, 10, 11]:  # min-like qualities
                                    prediction[idx] = r * 2 + 1
                                else:  # maj-like qualities
                                    prediction[idx] = r * 2
                    elif model_type == 'cnn' or model_type =='crnn':
                        prediction, _, _, _ = model(feature[:, n_timestep * t:n_timestep * (t + 1), :], torch.randint(config.model['num_chords'], (n_timestep,)).to(device))
                    for i in range(n_timestep):
                        if t == 0 and i == 0:
                            prev_chord = prediction[i].item()
                            continue
                        if prediction[i].item() != prev_chord:
                            lines.append(
                                '%.6f %.6f %s\n' % (
                                    start_time, time_unit * (n_timestep * t + i), idx2chord[prev_chord]))
                            start_time = time_unit * (n_timestep * t + i)
                            prev_chord = prediction[i].item()
                        if t == num_instance - 1 and i + num_pad == n_timestep:
                            if start_time != time_unit * (n_timestep * t + i):
                                lines.append(
                                    '%.6f %.6f %s\n' % (
                                        start_time, time_unit * (n_timestep * t + i), idx2chord[prev_chord]))
                            break
            pid = os.getpid()
            tmp_path = 'tmp_' + str(pid) + '.lab'
            with open(tmp_path, 'w') as f:
                for line in lines:
                    f.write(line)

            # Use score_with_durations to track WCSR components
            for m in root_majmin:
                score, correct_dur, total_dur = metrics_.score_with_durations(
                    metric=m, gt_path=lab_file_path, est_path=tmp_path
                )
                metrics_.score_list_dict[m].append(score)
                metrics_.correct_durations_dict[m].append(correct_dur)
                metrics_.total_durations_dict[m].append(total_dur)
            
            song_length_list.append(song_length_second)
            if verbose:
                for m in root_majmin:
                    print('song name %s, %s score : %.4f' % (song_name, m, metrics_.score_list_dict[m][-1]))
        except Exception as e:
            print('song name %s\' lab file error: %s' % (song_name, str(e)))

    # Check if we have any successful songs
    if len(song_length_list) == 0:
        print("WARNING: No songs were successfully processed!")
        # Return empty results
        return metrics_.score_list_dict, [], metrics_.average_score
    
    song_length_list = np.array(song_length_list)
    tmp = song_length_list / np.sum(song_length_list)
    for m in root_majmin:
        metrics_.average_score[m] = np.sum(np.multiply(metrics_.score_list_dict[m], tmp))
        # Compute WCSR for this metric
        metrics_.wcsr_dict[m] = metrics_.compute_wcsr(m)

    return metrics_.score_list_dict, song_length_list, metrics_.average_score, metrics_.wcsr_dict

def root_majmin_score_calculation_crf(valid_dataset, config, mean, std, device, pre_model, model, model_type, verbose=False):
    valid_song_names = valid_dataset.song_names
    # Normalize valid song names for comparison (lowercase, replace spaces with underscores)
    valid_song_names_normalized = {name.lower().replace(' ', '_'): name for name in valid_song_names}
    paths = valid_dataset.preprocessor.get_all_files()

    metrics_ = metrics()
    song_length_list = list()
    root_majmin = ['root', 'majmin']
    
    for path in paths:
        song_name, lab_file_path, mp3_file_path, _ = path
        # Normalize song_name for comparison
        song_name_normalized = song_name.lower().replace(' ', '_')
        if song_name_normalized not in valid_song_names_normalized:
            continue
        try:
            n_timestep = config.model['timestep']
            feature, feature_per_second, song_length_second = audio_file_to_features(mp3_file_path, config)
            feature = feature.T
            feature = (feature - mean) / std
            time_unit = feature_per_second

            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
            num_instance = feature.shape[0] // n_timestep

            start_time = 0.0
            lines = []
            with torch.no_grad():
                model.eval()
                feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
                for t in range(num_instance):
                    if (model_type == 'cnn') or (model_type == 'crnn') or (model_type == 'btc'):
                        logits = pre_model(feature[:, n_timestep * t:n_timestep * (t + 1), :], torch.randint(config.model['num_chords'], (n_timestep,)).to(device))
                        prediction, _ = model(logits, torch.randint(config.model['num_chords'], (n_timestep,)).to(device))
                    else:
                        raise NotImplementedError
                    for i in range(n_timestep):
                        if t == 0 and i == 0:
                            prev_chord = prediction[i].item()
                            continue
                        if prediction[i].item() != prev_chord:
                            lines.append(
                                '%.6f %.6f %s\n' % (
                                    start_time, time_unit * (n_timestep * t + i), idx2chord[prev_chord]))
                            start_time = time_unit * (n_timestep * t + i)
                            prev_chord = prediction[i].item()
                        if t == num_instance - 1 and i + num_pad == n_timestep:
                            if start_time != time_unit * (n_timestep * t + i):
                                lines.append(
                                    '%.6f %.6f %s\n' % (
                                        start_time, time_unit * (n_timestep * t + i), idx2chord[prev_chord]))
                            break
            pid = os.getpid()
            tmp_path = 'tmp_' + str(pid) + '.lab'
            with open(tmp_path, 'w') as f:
                for line in lines:
                    f.write(line)

            # Use score_with_durations to track WCSR components
            for m in root_majmin:
                score, correct_dur, total_dur = metrics_.score_with_durations(
                    metric=m, gt_path=lab_file_path, est_path=tmp_path
                )
                metrics_.score_list_dict[m].append(score)
                metrics_.correct_durations_dict[m].append(correct_dur)
                metrics_.total_durations_dict[m].append(total_dur)
            
            song_length_list.append(song_length_second)
            if verbose:
                for m in root_majmin:
                    print('song name %s, %s score : %.4f' % (song_name, m, metrics_.score_list_dict[m][-1]))
        except:
            print('song name %s\' lab file error' % song_name)

    song_length_list = np.array(song_length_list)
    tmp = song_length_list / np.sum(song_length_list)
    for m in root_majmin:
        metrics_.average_score[m] = np.sum(np.multiply(metrics_.score_list_dict[m], tmp))
        # Compute WCSR for this metric
        metrics_.wcsr_dict[m] = metrics_.compute_wcsr(m)

    return metrics_.score_list_dict, song_length_list, metrics_.average_score, metrics_.wcsr_dict


def large_voca_score_calculation(valid_dataset, config, mean, std, device, model, model_type, verbose=False):
    idx2voca = idx2voca_chord()
    valid_song_names = valid_dataset.song_names
    # Normalize valid song names for comparison (lowercase, replace spaces with underscores)
    valid_song_names_normalized = {name.lower().replace(' ', '_'): name for name in valid_song_names}
    paths = valid_dataset.preprocessor.get_all_files()

    metrics_ = metrics()
    song_length_list = list()
    
    for path in paths:
        song_name, lab_file_path, mp3_file_path, _ = path
        # Normalize song_name for comparison
        song_name_normalized = song_name.lower().replace(' ', '_')
        if song_name_normalized not in valid_song_names_normalized:
            continue
        try:
            n_timestep = config.model['timestep']
            feature, feature_per_second, song_length_second = audio_file_to_features(mp3_file_path, config)
            feature = feature.T
            feature = (feature - mean) / std
            time_unit = feature_per_second

            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
            num_instance = feature.shape[0] // n_timestep

            start_time = 0.0
            lines = []
            with torch.no_grad():
                model.eval()
                feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
                for t in range(num_instance):
                    if model_type == 'btc':
                        encoder_output, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
                        prediction, _ = model.output_layer(encoder_output)
                        prediction = prediction.squeeze()
                    elif model_type == 'btc_structured':
                        # Handle structured output model - use self_attn_layers and output_layer directly
                        chunk = feature[:, n_timestep * t:n_timestep * (t + 1), :]
                        encoder_output, _ = model.self_attn_layers(chunk)
                        # Get predictions from structured output layer
                        root_pred, quality_pred, bass_pred, _, _, _ = model.output_layer(encoder_output)
                        root_pred = root_pred.squeeze()  # [n_timestep]
                        quality_pred = quality_pred.squeeze()  # [n_timestep]
                        
                        # Convert structured predictions to chord indices (vectorized)
                        # root 12 = no chord (N), quality 14 = no chord, quality 15 = unknown (X)
                        prediction = root_pred * 14 + quality_pred  # Base conversion
                        # Handle no chord: root == 12 or quality == 14
                        no_chord_mask = (root_pred == 12) | (quality_pred == 14)
                        prediction[no_chord_mask] = 169
                        # Handle unknown: quality == 15
                        unknown_mask = (quality_pred == 15) & ~no_chord_mask
                        prediction[unknown_mask] = 168
                    elif model_type == 'chordformer':
                        # Handle ChordFormer model - uses conformer_encoder instead of self_attn_layers
                        chunk = feature[:, n_timestep * t:n_timestep * (t + 1), :]
                        encoder_output, _ = model.conformer_encoder(chunk)
                        # Get predictions from structured output layer
                        root_pred, quality_pred, bass_pred, _, _, _ = model.output_layer(encoder_output)
                        root_pred = root_pred.squeeze()  # [n_timestep]
                        quality_pred = quality_pred.squeeze()  # [n_timestep]
                        
                        # Convert structured predictions to chord indices (vectorized)
                        # root 12 = no chord (N), quality 14 = no chord, quality 15 = unknown (X)
                        prediction = root_pred * 14 + quality_pred  # Base conversion
                        # Handle no chord: root == 12 or quality == 14
                        no_chord_mask = (root_pred == 12) | (quality_pred == 14)
                        prediction[no_chord_mask] = 169
                        # Handle unknown: quality == 15
                        unknown_mask = (quality_pred == 15) & ~no_chord_mask
                        prediction[unknown_mask] = 168
                    elif model_type == 'cnn' or model_type =='crnn':
                        prediction, _, _, _ = model(feature[:, n_timestep * t:n_timestep * (t + 1), :], torch.randint(config.model['num_chords'], (n_timestep,)).to(device))
                    for i in range(n_timestep):
                        if t == 0 and i == 0:
                            prev_chord = prediction[i].item()
                            continue
                        if prediction[i].item() != prev_chord:
                            lines.append(
                                '%.6f %.6f %s\n' % (
                                    start_time, time_unit * (n_timestep * t + i), idx2voca[prev_chord]))
                            start_time = time_unit * (n_timestep * t + i)
                            prev_chord = prediction[i].item()
                        if t == num_instance - 1 and i + num_pad == n_timestep:
                            if start_time != time_unit * (n_timestep * t + i):
                                lines.append(
                                    '%.6f %.6f %s\n' % (
                                        start_time, time_unit * (n_timestep * t + i), idx2voca[prev_chord]))
                            break
            pid = os.getpid()
            tmp_path = 'tmp_' + str(pid) + '.lab'
            with open(tmp_path, 'w') as f:
                for line in lines:
                    f.write(line)

            # Use score_with_durations to track WCSR components
            for m in metrics_.score_metrics:
                score, correct_dur, total_dur = metrics_.score_with_durations(
                    metric=m, gt_path=lab_file_path, est_path=tmp_path
                )
                metrics_.score_list_dict[m].append(score)
                metrics_.correct_durations_dict[m].append(correct_dur)
                metrics_.total_durations_dict[m].append(total_dur)
            
            song_length_list.append(song_length_second)
            if verbose:
                for m in metrics_.score_metrics:
                    print('song name %s, %s score : %.4f' % (song_name, m, metrics_.score_list_dict[m][-1]))
        except Exception as e:
            print('song name %s\' lab file error: %s' % (song_name, str(e)))

    # Check if we have any successful songs
    if len(song_length_list) == 0:
        print("WARNING: No songs were successfully processed!")
        # Return empty results (4 values to match expected return)
        return metrics_.score_list_dict, [], metrics_.average_score, {}
    
    song_length_list = np.array(song_length_list)
    tmp = song_length_list / np.sum(song_length_list)
    for m in metrics_.score_metrics:
        metrics_.average_score[m] = np.sum(np.multiply(metrics_.score_list_dict[m], tmp))
        # Compute WCSR for this metric
        metrics_.wcsr_dict[m] = metrics_.compute_wcsr(m)

    return metrics_.score_list_dict, song_length_list, metrics_.average_score, metrics_.wcsr_dict

def large_voca_score_calculation_crf(valid_dataset, config, mean, std, device, pre_model, model, model_type, verbose=False):
    idx2voca = idx2voca_chord()
    valid_song_names = valid_dataset.song_names
    # Normalize valid song names for comparison (lowercase, replace spaces with underscores)
    valid_song_names_normalized = {name.lower().replace(' ', '_'): name for name in valid_song_names}
    paths = valid_dataset.preprocessor.get_all_files()

    metrics_ = metrics()
    song_length_list = list()
    for path in paths:
        song_name, lab_file_path, mp3_file_path, _ = path
        # Normalize song_name for comparison
        song_name_normalized = song_name.lower().replace(' ', '_')
        if song_name_normalized not in valid_song_names_normalized:
            continue
        try:
            n_timestep = config.model['timestep']
            feature, feature_per_second, song_length_second = audio_file_to_features(mp3_file_path, config)
            feature = feature.T
            feature = (feature - mean) / std
            time_unit = feature_per_second

            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
            num_instance = feature.shape[0] // n_timestep

            start_time = 0.0
            lines = []
            with torch.no_grad():
                model.eval()
                feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
                for t in range(num_instance):
                    if (model_type == 'cnn') or (model_type == 'crnn') or (model_type == 'btc'):
                        logits = pre_model(feature[:, n_timestep * t:n_timestep * (t + 1), :], torch.randint(config.model['num_chords'], (n_timestep,)).to(device))
                        prediction, _ = model(logits, torch.randint(config.model['num_chords'], (n_timestep,)).to(device))
                    else:
                        raise NotImplementedError
                    for i in range(n_timestep):
                        if t == 0 and i == 0:
                            prev_chord = prediction[i].item()
                            continue
                        if prediction[i].item() != prev_chord:
                            lines.append(
                                '%.6f %.6f %s\n' % (
                                    start_time, time_unit * (n_timestep * t + i), idx2voca[prev_chord]))
                            start_time = time_unit * (n_timestep * t + i)
                            prev_chord = prediction[i].item()
                        if t == num_instance - 1 and i + num_pad == n_timestep:
                            if start_time != time_unit * (n_timestep * t + i):
                                lines.append(
                                    '%.6f %.6f %s\n' % (
                                        start_time, time_unit * (n_timestep * t + i), idx2voca[prev_chord]))
                            break
            pid = os.getpid()
            tmp_path = 'tmp_' + str(pid) + '.lab'
            with open(tmp_path, 'w') as f:
                for line in lines:
                    f.write(line)

            # Use score_with_durations to track WCSR components
            for m in metrics_.score_metrics:
                score, correct_dur, total_dur = metrics_.score_with_durations(
                    metric=m, gt_path=lab_file_path, est_path=tmp_path
                )
                metrics_.score_list_dict[m].append(score)
                metrics_.correct_durations_dict[m].append(correct_dur)
                metrics_.total_durations_dict[m].append(total_dur)
            
            song_length_list.append(song_length_second)
            if verbose:
                for m in metrics_.score_metrics:
                    print('song name %s, %s score : %.4f' % (song_name, m, metrics_.score_list_dict[m][-1]))
        except:
            print('song name %s\' lab file error' % song_name)

    song_length_list = np.array(song_length_list)
    tmp = song_length_list / np.sum(song_length_list)
    for m in metrics_.score_metrics:
        metrics_.average_score[m] = np.sum(np.multiply(metrics_.score_list_dict[m], tmp))
        # Compute WCSR for this metric
        metrics_.wcsr_dict[m] = metrics_.compute_wcsr(m)

    return metrics_.score_list_dict, song_length_list, metrics_.average_score, metrics_.wcsr_dict
