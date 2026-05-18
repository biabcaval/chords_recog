import os
import librosa
from utils.chords import Chords
import re
from enum import Enum
import pyrubberband as pyrb
import torch
import math
import numpy as np

class FeatureTypes(Enum):
    cqt = 'cqt'


def cqt_to_log_db(cqt, ref='max', top_db=80.0, amin=1e-10):
    """Convert a (possibly complex) CQT to log-magnitude in **dB ref=max**.

    Mirrors the ChordFormer paper formulation (Tabela 1 of
    ``docs/chordformer_replication.md`` / the comparison tables):

        S_dB(t, f) = 20 * log10(|CQT(t, f)| / max(|CQT|))
        S_dB       = max(S_dB, -top_db)              # floor at ``top_db`` below the per-instance max

    This replaces the previous ``np.log(np.abs(x) + 1e-6)`` transform used
    throughout the loaders and inference paths, which was a natural-log
    of magnitude with an additive epsilon -- mathematically *not* the same
    as the ChordFormer paper's dB(ref=max) formulation.

    The two transforms are both monotonic in ``|x|`` so the *ordering* of
    bins is preserved, but their distributions differ in shape (dB has a
    hard floor at ``-top_db``; the previous form had a long left tail) and
    in scale (log10 vs ln, with a normalisation by the per-instance max).
    Because we centre/standardise the loader output via mean/std stored in
    ``normalization*.pt``, those statistics **must be recomputed** with
    :mod:`scripts/compute_normalization` after switching transforms.

    Args:
        cqt: ndarray of shape ``(n_bins, n_frames)`` (real or complex).
            Complex input is handled via ``np.abs``.
        ref: reference for the dB conversion. ``'max'`` (default) uses
            ``np.max(|cqt|)`` for each call (per-instance normalisation,
            matching ChordFormer). A positive float fixes the reference
            globally; a callable is forwarded to ``librosa.amplitude_to_db``.
        top_db: floor (in dB below the reference) for the output.
            ``80.0`` matches the librosa default and is the standard
            choice in MIR.
        amin: minimum magnitude before taking log (avoids log(0)).

    Returns:
        ndarray of the same shape as ``cqt``, dtype ``float32``, with
        values in ``[-top_db, 0]`` when ``ref='max'``.
    """
    mag = np.abs(cqt)
    ref_arg = np.max if ref == 'max' else ref
    feature = librosa.amplitude_to_db(mag, ref=ref_arg, amin=amin, top_db=top_db)
    return feature.astype(np.float32)


def shift_cqt_bins(cqt, semitones, bins_per_semitone=3):
    """Shift a CQT array along its frequency axis (axis=0) by an integer
    number of semitones, without wrap-around.

    Bins introduced at the edge by the shift are filled with zeros, since
    those frequencies were not present in the original signal.  Works for
    both real-valued (magnitude) and complex-valued CQT arrays; dtype and
    shape are preserved.

    This is the cheap, exact equivalent of ``librosa.cqt(pyrb.pitch_shift(
    audio, sr, semitones))`` *only* for CQT frontends where
    ``bins_per_octave`` is a multiple of 12 (so each semitone corresponds to
    an integer number of bins).  For chord recognition this approximation
    is musically faithful because the CQT is log-frequency by construction.

    Args:
        cqt: ndarray of shape ``(n_bins, n_frames)``.
        semitones: integer pitch shift in semitones.
        bins_per_semitone: bins per semitone of the underlying CQT
            (``bins_per_octave // 12``).  Default 3 matches the project's
            ``bins_per_octave=36`` setting.

    Returns:
        ndarray of the same shape and dtype as ``cqt``.
    """
    if semitones == 0:
        return cqt.copy()
    delta = int(semitones) * int(bins_per_semitone)
    out = np.zeros_like(cqt)
    if delta > 0:
        out[delta:, :] = cqt[:-delta, :]
    else:
        out[:delta, :] = cqt[-delta:, :]
    return out

class Preprocess():
    def __init__(self, config, feature_to_use, dataset_names, root_dir):
        self.config = config
        self.dataset_names = dataset_names
        self.root_path = root_dir + '/'

        self.time_interval = config.feature["hop_length"]/config.mp3["song_hz"]
        self.no_of_chord_datapoints_per_sequence = math.ceil(config.mp3['inst_len'] / self.time_interval)
        self.Chord_class = Chords()

        # isophonic
        self.isophonic_directory = self.root_path + 'isophonic/'

        # uspop
        self.uspop_directory = self.root_path + 'uspop/'
        self.uspop_audio_path = 'audio/'
        self.uspop_lab_path = 'annotations/uspopLabels/'
        self.uspop_index_path = 'annotations/uspopLabels.txt'

        # Generic dataset support (billboard, jaah, rwc)
        # These datasets follow a standard structure: annotations/ and audio/ directories
        self.generic_datasets = ['billboard', 'jaah', 'rwc', 'dj_avan_songbook1', 'dj_avan_songbook2', 'robbiewilliams', 'queen',
                                 'balanced_v1_train', 'balanced_v1_test', 'balanced_v1_full']

        # Datasets under a subfolder of root_path
        self._dataset_subdir = {
            'balanced_v1_train': 'personalized_datasets/balanced_v1_train',
            'balanced_v1_test': 'personalized_datasets/balanced_v1_test',
            'balanced_v1_full': 'personalized_datasets/balanced_v1_full',
        }

        for dataset_name in self.generic_datasets:
            if dataset_name in self.dataset_names:
                rel_path = self._dataset_subdir.get(dataset_name, dataset_name)
                setattr(self, f'{dataset_name}_directory', self.root_path + f'{rel_path}/')
                setattr(self, f'{dataset_name}_audio_path', 'audio/')
                setattr(self, f'{dataset_name}_lab_path', 'annotations/')

        self.feature_name = feature_to_use
        self.is_cut_last_chord = False

    def find_mp3_path(self, dirpath, word):
        for filename in os.listdir(dirpath):
            last_dir = dirpath.split("/")[-2]
            if ".mp3" in filename:
                tmp = filename.replace(".mp3", "")
                tmp = tmp.replace(last_dir, "")
                filename_lower = tmp.lower()
                filename_lower = " ".join(re.findall("[a-zA-Z]+", filename_lower))
                if word.lower().replace(" ", "") in filename_lower.replace(" ", ""):
                    return filename

    def find_audio_path_generic(self, audio_dir, lab_filename):
        """
        Generic function to find audio file matching a label file.
        Three-pass matching strategy:
          1. Exact basename match (case-insensitive, space→underscore)
          2. Prefix match (handles descriptive suffixes like jaah_018-artist_title)
          3. Numeric ID match (handles zero-padding differences like jaah_000 vs jaah_00)
        """
        lab_basename = os.path.splitext(lab_filename)[0]
        lab_normalized = lab_basename.lower().replace(' ', '_')

        candidates = [f for f in os.listdir(audio_dir)
                       if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a'))]

        for filename in candidates:
            audio_normalized = os.path.splitext(filename)[0].lower().replace(' ', '_')
            if audio_normalized == lab_normalized:
                return filename

        for filename in candidates:
            audio_normalized = os.path.splitext(filename)[0].lower().replace(' ', '_')
            if (audio_normalized.startswith(lab_normalized + '-') or
                audio_normalized.startswith(lab_normalized + '_') or
                lab_normalized.startswith(audio_normalized + '-') or
                lab_normalized.startswith(audio_normalized + '_')):
                return filename

        lab_match = re.match(r'(.+?)(\d+)$', lab_normalized)
        if lab_match:
            lab_prefix, lab_num = lab_match.group(1), int(lab_match.group(2))
            for filename in candidates:
                audio_normalized = os.path.splitext(filename)[0].lower().replace(' ', '_')
                audio_match = re.match(r'(.+?)(\d+)(.*)', audio_normalized)
                if audio_match:
                    audio_prefix = audio_match.group(1)
                    audio_num = int(audio_match.group(2))
                    if audio_prefix == lab_prefix and audio_num == lab_num:
                        return filename

        return None

    def get_all_files(self):
        res_list = []

        # isophonic
        if "isophonic" in self.dataset_names:
            for dirpath, dirnames, filenames in os.walk(self.isophonic_directory):
                if not dirnames:
                    for filename in filenames:
                        if ".lab" in filename:
                            tmp = filename.replace(".lab", "")
                            song_name = " ".join(re.findall("[a-zA-Z]+", tmp)).replace("CD", "")
                            mp3_path = self.find_mp3_path(dirpath, song_name)
                            res_list.append([song_name, os.path.join(dirpath, filename), os.path.join(dirpath, mp3_path),
                                             os.path.join(self.root_path, "result", "isophonic")])

        # uspop
        if "uspop" in self.dataset_names:
            with open(os.path.join(self.uspop_directory, self.uspop_index_path)) as f:
                uspop_lab_list = f.readlines()
            uspop_lab_list = [x.strip() for x in uspop_lab_list]

            for lab_path in uspop_lab_list:
                spl = lab_path.split('/')
                lab_artist = self.uspop_pre(spl[2])
                lab_title = self.uspop_pre(spl[4][3:-4])
                lab_path = lab_path.replace('./uspopLabels/', '')
                lab_path = os.path.join(self.uspop_directory, self.uspop_lab_path, lab_path)

                for filename in os.listdir(os.path.join(self.uspop_directory, self.uspop_audio_path)):
                    if not '.csv' in filename:
                        spl = filename.split('-')
                        mp3_artist = self.uspop_pre(spl[0])
                        mp3_title = self.uspop_pre(spl[1][:-4])

                        if lab_artist == mp3_artist and lab_title == mp3_title:
                            res_list.append([mp3_artist + mp3_title, lab_path,
                                             os.path.join(self.uspop_directory, self.uspop_audio_path, filename),
                                             os.path.join(self.root_path, "result", "uspop")])
                            break

        # Generic datasets (billboard, jaah, rwc)
        for dataset_name in self.generic_datasets:
            if dataset_name in self.dataset_names:
                dataset_dir = getattr(self, f'{dataset_name}_directory')
                audio_path = getattr(self, f'{dataset_name}_audio_path')
                lab_path = getattr(self, f'{dataset_name}_lab_path')
                
                annotations_dir = os.path.join(dataset_dir, lab_path)
                audio_dir = os.path.join(dataset_dir, audio_path)
                
                if os.path.exists(annotations_dir) and os.path.exists(audio_dir):
                    for filename in os.listdir(annotations_dir):
                        if filename.lower().endswith(('.lab', '.txt')):
                            lab_file_path = os.path.join(annotations_dir, filename)
                            audio_filename = self.find_audio_path_generic(audio_dir, filename)
                            
                            if audio_filename:
                                song_name = os.path.splitext(filename)[0]
                                res_list.append([
                                    song_name,
                                    lab_file_path,
                                    os.path.join(audio_dir, audio_filename),
                                    os.path.join(self.root_path, "result", dataset_name)
                                ])

        return res_list

    def uspop_pre(self, text):
        text = text.lower()
        text = text.replace('_', '')
        text = text.replace(' ', '')
        text = " ".join(re.findall("[a-zA-Z]+", text))
        return text

    def song_pre(self, text):
        to_remove = ["'", '`', '(', ')', ' ', '&', 'and', 'And']

        for remove in to_remove:
            text = text.replace(remove, '')

        return text

    def config_to_folder(self):
        mp3_config = self.config.mp3
        feature_config = self.config.feature
        mp3_string = "%d_%.1f_%.1f" % \
                     (mp3_config['song_hz'], mp3_config['inst_len'],
                      mp3_config['skip_interval'])
        feature_string = "%s_%d_%d_%d" % \
                         (self.feature_name.value, feature_config['n_bins'], feature_config['bins_per_octave'], feature_config['hop_length'])

        return mp3_config, feature_config, mp3_string, feature_string

    def generate_labels_features_new(self, all_list):
        pid = os.getpid()
        mp3_config, feature_config, mp3_str, feature_str = self.config_to_folder()

        i = 0  # number of songs
        j = 0  # number of impossible songs
        k = 0  # number of tried songs
        total = 0  # number of generated instances

        stretch_factors = [1.0]
        shift_factors = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

        loop_broken = False
        for song_name, lab_path, mp3_path, save_path in all_list:

            # different song initialization
            if loop_broken:
                loop_broken = False

            i += 1
            print(pid, "generating features from ...", os.path.join(mp3_path))
            if i % 10 == 0:
                print(i, ' th song')

            original_wav, sr = librosa.load(os.path.join(mp3_path), sr=mp3_config['song_hz'])

            # make result path if not exists
            # save_path, mp3_string, feature_string, song_name, aug.pt
            result_path = os.path.join(save_path, mp3_str, feature_str, song_name.strip())
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            # calculate result
            for stretch_factor in stretch_factors:
                if loop_broken:
                    loop_broken = False
                    break

                for shift_factor in shift_factors:
                    # for filename
                    idx = 0

                    chord_info = self.Chord_class.get_converted_chord(os.path.join(lab_path))

                    k += 1
                    # stretch original sound and chord info
                    x = pyrb.time_stretch(original_wav, sr, stretch_factor)
                    x = pyrb.pitch_shift(x, sr, shift_factor)
                    audio_length = x.shape[0]
                    chord_info['start'] = chord_info['start'] * 1/stretch_factor
                    chord_info['end'] = chord_info['end'] * 1/stretch_factor

                    last_sec = chord_info.iloc[-1]['end']
                    last_sec_hz = int(last_sec * mp3_config['song_hz'])
                    skip_interval_samples = int(mp3_config['skip_interval'] * mp3_config['song_hz'])

                    if audio_length + skip_interval_samples < last_sec_hz:
                        print('loaded song is too short :', song_name)
                        loop_broken = True
                        j += 1
                        break
                    elif audio_length > last_sec_hz:
                        x = x[:last_sec_hz]

                    origin_length = last_sec_hz
                    origin_length_in_sec = origin_length / mp3_config['song_hz']

                    current_start_second = 0

                    # get chord list between current_start_second and current+song_length
                    while current_start_second + mp3_config['inst_len'] < origin_length_in_sec:
                        inst_start_sec = current_start_second
                        curSec = current_start_second

                        chord_list = []
                        # extract chord per 1/self.time_interval
                        while curSec < inst_start_sec + mp3_config['inst_len']:
                            try:
                                available_chords = chord_info.loc[(chord_info['start'] <= curSec) & (
                                        chord_info['end'] > curSec + self.time_interval)].copy()
                                if len(available_chords) == 0:
                                    available_chords = chord_info.loc[((chord_info['start'] >= curSec) & (
                                            chord_info['start'] <= curSec + self.time_interval)) | (
                                                                              (chord_info['end'] >= curSec) & (
                                                                              chord_info['end'] <= curSec + self.time_interval))].copy()
                                if len(available_chords) == 1:
                                    chord = available_chords['chord_id'].iloc[0]
                                elif len(available_chords) > 1:
                                    max_starts = available_chords.apply(lambda row: max(row['start'], curSec),
                                                                        axis=1)
                                    available_chords['max_start'] = max_starts
                                    min_ends = available_chords.apply(
                                        lambda row: min(row.end, curSec + self.time_interval), axis=1)
                                    available_chords['min_end'] = min_ends
                                    chords_lengths = available_chords['min_end'] - available_chords['max_start']
                                    available_chords['chord_length'] = chords_lengths
                                    chord = available_chords.loc[available_chords['chord_length'].idxmax(), 'chord_id']                                
                                else:
                                    chord = 24
                            except Exception as e:
                                chord = 24
                                print(e)
                                print(pid, "no chord")
                                raise RuntimeError()
                            finally:
                                # convert chord by shift factor
                                if chord != 24:
                                    chord += shift_factor * 2
                                    chord = chord % 24

                                chord_list.append(chord)
                                curSec += self.time_interval

                        if len(chord_list) == self.no_of_chord_datapoints_per_sequence:
                            # Check if file already exists before processing
                            etc = '%.1f_%.1f' % (
                                current_start_second, current_start_second + mp3_config['inst_len'])
                            aug = '%.2f_%i' % (stretch_factor, shift_factor)
                            filename = aug + "_" + str(idx) + ".pt"
                            output_file = os.path.join(result_path, filename)
                            
                            if os.path.exists(output_file):
                                print(f"{pid} Skipping existing file: {filename}")
                                idx += 1
                                total += 1
                            else:
                                try:
                                    sequence_start_time = current_start_second
                                    sequence_end_time = current_start_second + mp3_config['inst_len']

                                    start_index = int(sequence_start_time * mp3_config['song_hz'])
                                    end_index = int(sequence_end_time * mp3_config['song_hz'])

                                    song_seq = x[start_index:end_index]

                                    if self.feature_name == FeatureTypes.cqt:
                                        # print(pid, "make feature")
                                        feature = librosa.cqt(song_seq, sr=sr, n_bins=feature_config['n_bins'],
                                                              bins_per_octave=feature_config['bins_per_octave'],
                                                              hop_length=feature_config['hop_length'])
                                    else:
                                        raise NotImplementedError

                                    if feature.shape[1] > self.no_of_chord_datapoints_per_sequence:
                                        feature = feature[:, :self.no_of_chord_datapoints_per_sequence]

                                    if feature.shape[1] != self.no_of_chord_datapoints_per_sequence:
                                        print('loaded features length is too short :', song_name)
                                        loop_broken = True
                                        j += 1
                                        break

                                    result = {
                                        'feature': feature,
                                        'chord': chord_list,
                                        'etc': etc
                                    }

                                    # save_path, mp3_string, feature_string, song_name, aug.pt
                                    torch.save(result, output_file)
                                    idx += 1
                                    total += 1
                                except Exception as e:
                                    print(e)
                                    print(pid, "feature error")
                                    raise RuntimeError()
                        else:
                            print("invalid number of chord datapoints in sequence :", len(chord_list))
                        current_start_second += mp3_config['skip_interval']
        print(pid, "total instances: %d" % total)

    def generate_labels_features_voca(self, all_list):
        pid = os.getpid()
        mp3_config, feature_config, mp3_str, feature_str = self.config_to_folder()

        i = 0  # number of songs
        j = 0  # number of impossible songs
        k = 0  # number of tried songs
        total = 0  # number of generated instances
        stretch_factors = [1.0]
        shift_factors = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

        loop_broken = False
        for song_name, lab_path, mp3_path, save_path in all_list:
            save_path = save_path + '_voca'

            # different song initialization
            if loop_broken:
                loop_broken = False

            i += 1
            print(pid, "generating features from ...", os.path.join(mp3_path))
            if i % 10 == 0:
                print(i, ' th song')

            original_wav, sr = librosa.load(os.path.join(mp3_path), sr=mp3_config['song_hz'])

            # save_path, mp3_string, feature_string, song_name, aug.pt
            result_path = os.path.join(save_path, mp3_str, feature_str, song_name.strip())
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            # calculate result
            for stretch_factor in stretch_factors:
                if loop_broken:
                    loop_broken = False
                    break

                for shift_factor in shift_factors:
                    # for filename
                    idx = 0

                    try:
                        chord_info = self.Chord_class.get_converted_chord_full(os.path.join(lab_path))
                    except Exception as e:
                        print(e)
                        print(pid, " chord lab file error : %s" % song_name)
                        loop_broken = True
                        j += 1
                        break

                    k += 1
                    # stretch original sound and chord info
                    x = pyrb.time_stretch(original_wav, sr, stretch_factor)
                    x = pyrb.pitch_shift(x, sr, shift_factor)
                    audio_length = x.shape[0]
                    chord_info['start'] = chord_info['start'] * 1/stretch_factor
                    chord_info['end'] = chord_info['end'] * 1/stretch_factor

                    last_sec = chord_info.iloc[-1]['end']
                    last_sec_hz = int(last_sec * mp3_config['song_hz'])
                    skip_interval_samples = int(mp3_config['skip_interval'] * mp3_config['song_hz'])

                    if audio_length + skip_interval_samples < last_sec_hz:
                        print('loaded song is too short :', song_name)
                        loop_broken = True
                        j += 1
                        break
                    elif audio_length > last_sec_hz:
                        x = x[:last_sec_hz]

                    origin_length = last_sec_hz
                    origin_length_in_sec = origin_length / mp3_config['song_hz']

                    current_start_second = 0

                    # get chord list between current_start_second and current+song_length
                    while current_start_second + mp3_config['inst_len'] < origin_length_in_sec:
                        inst_start_sec = current_start_second
                        curSec = current_start_second

                        chord_list = []
                        root_list = []
                        quality_list = []
                        bass_list = []
                        chord_label_list = []
                        
                        # extract chord per 1/self.time_interval
                        while curSec < inst_start_sec + mp3_config['inst_len']:
                            try:
                                available_chords = chord_info.loc[(chord_info['start'] <= curSec) & (chord_info['end'] > curSec + self.time_interval)].copy()
                                if len(available_chords) == 0:
                                    available_chords = chord_info.loc[((chord_info['start'] >= curSec) & (chord_info['start'] <= curSec + self.time_interval)) | ((chord_info['end'] >= curSec) & (chord_info['end'] <= curSec + self.time_interval))].copy()

                                if len(available_chords) == 1:
                                    chord = available_chords['chord_id'].iloc[0]
                                    root = available_chords['root'].iloc[0]
                                    quality = available_chords['quality'].iloc[0]
                                    bass = available_chords['bass'].iloc[0]
                                    chord_label = available_chords['chord_label'].iloc[0]
                                elif len(available_chords) > 1:
                                    max_starts = available_chords.apply(lambda row: max(row['start'], curSec),axis=1)
                                    available_chords['max_start'] = max_starts
                                    min_ends = available_chords.apply(lambda row: min(row.end, curSec + self.time_interval), axis=1)
                                    available_chords['min_end'] = min_ends
                                    chords_lengths = available_chords['min_end'] - available_chords['max_start']
                                    available_chords['chord_length'] = chords_lengths
                                    max_idx = available_chords['chord_length'].idxmax()
                                    chord = available_chords.loc[max_idx, 'chord_id']
                                    root = available_chords.loc[max_idx, 'root']
                                    quality = available_chords.loc[max_idx, 'quality']
                                    bass = available_chords.loc[max_idx, 'bass']
                                    chord_label = available_chords.loc[max_idx, 'chord_label']
                                else:
                                    chord = 169
                                    root = 12  # No chord
                                    quality = 14  # No chord quality
                                    bass = 12  # No bass
                                    chord_label = 'N'
                            except Exception as e:
                                chord = 169
                                root = 12
                                quality = 14
                                bass = 12
                                chord_label = 'N'
                                print(e)
                                print(pid, "no chord")
                                raise RuntimeError()
                            finally:
                                # convert chord by shift factor
                                if chord != 169 and chord != 168:
                                    chord += shift_factor * 14
                                    chord = chord % 168
                                
                                # Apply pitch shifting to root and bass (not quality)
                                if root != 12:  # If not "no chord"
                                    root = (root + shift_factor) % 12
                                if bass != 12:  # If not "no bass"
                                    bass = (bass + shift_factor) % 12

                                # Transpose the full chord label string
                                chord_label = self.Chord_class.transpose_chord_label(
                                    chord_label, shift_factor)

                                chord_list.append(chord)
                                root_list.append(root)
                                quality_list.append(quality)
                                bass_list.append(bass)
                                chord_label_list.append(chord_label)
                                curSec += self.time_interval

                        if len(chord_list) == self.no_of_chord_datapoints_per_sequence:
                            # Check if file already exists before processing
                            etc = '%.1f_%.1f' % (
                                current_start_second, current_start_second + mp3_config['inst_len'])
                            aug = '%.2f_%i' % (stretch_factor, shift_factor)
                            filename = aug + "_" + str(idx) + ".pt"
                            output_file = os.path.join(result_path, filename)
                            
                            if os.path.exists(output_file):
                                print(f"{pid} Skipping existing file: {filename}")
                                idx += 1
                                total += 1
                            else:
                                try:
                                    sequence_start_time = current_start_second
                                    sequence_end_time = current_start_second + mp3_config['inst_len']

                                    start_index = int(sequence_start_time * mp3_config['song_hz'])
                                    end_index = int(sequence_end_time * mp3_config['song_hz'])

                                    song_seq = x[start_index:end_index]

                                    if self.feature_name == FeatureTypes.cqt:
                                        feature = librosa.cqt(song_seq, sr=sr, n_bins=feature_config['n_bins'],
                                                              bins_per_octave=feature_config['bins_per_octave'],
                                                              hop_length=feature_config['hop_length'])
                                    else:
                                        raise NotImplementedError

                                    if feature.shape[1] > self.no_of_chord_datapoints_per_sequence:
                                        feature = feature[:, :self.no_of_chord_datapoints_per_sequence]

                                    if feature.shape[1] != self.no_of_chord_datapoints_per_sequence:
                                        print('loaded features length is too short :', song_name)
                                        loop_broken = True
                                        j += 1
                                        break

                                    result = {
                                        'feature': feature,
                                        'chord': chord_list,
                                        'root': root_list,
                                        'quality': quality_list,
                                        'bass': bass_list,
                                        'original_chord_labels': chord_label_list,
                                        'etc': etc
                                    }

                                    # save_path, mp3_string, feature_string, song_name, aug.pt
                                    torch.save(result, output_file)
                                    idx += 1
                                    total += 1
                                except Exception as e:
                                    print(e)
                                    print(pid, "feature error")
                                    raise RuntimeError()
                        else:
                            print("invalid number of chord datapoints in sequence :", len(chord_list))
                        current_start_second += mp3_config['skip_interval']
        print(pid, "total instances: %d" % total)

    def generate_labels_features_voca_cqtroll(self, all_list):
        """CQT-first variant of :meth:`generate_labels_features_voca`.

        Computes the CQT *once* per song on the original audio, then derives
        each pitch-shifted version by rolling bins along the frequency axis
        (requires ``bins_per_octave`` to be a multiple of 12).  Chord labels
        and component fields are transposed exactly as in the pyrb pipeline.

        Output ``.pt`` schema and filenames are byte-identical to the pyrb
        variant ('feature', 'chord', 'root', 'quality', 'bass',
        'original_chord_labels', 'etc'; filename ``'%.2f_%i_%i.pt'``), so
        downstream loaders ([data/audio_dataset.py](data/audio_dataset.py))
        do not need any change -- only the *root* of the result tree differs
        (the caller is expected to pass save_paths under e.g.
        ``result_cqtroll/`` instead of ``result/``).

        Time-stretch is **not** supported by this method (only
        ``stretch_factor == 1.0``); use :meth:`generate_labels_features_voca`
        for time-stretch augmentation.
        """
        pid = os.getpid()
        mp3_config, feature_config, mp3_str, feature_str = self.config_to_folder()

        if feature_config['bins_per_octave'] % 12 != 0:
            raise ValueError(
                "generate_labels_features_voca_cqtroll requires "
                f"bins_per_octave to be a multiple of 12 "
                f"(got {feature_config['bins_per_octave']})"
            )
        bins_per_semitone = feature_config['bins_per_octave'] // 12
        hop = feature_config['hop_length']

        i = 0  # number of songs
        j = 0  # number of impossible songs
        k = 0  # number of tried songs
        total = 0  # number of generated instances
        stretch_factors = [1.0]
        shift_factors = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

        for st in stretch_factors:
            if st != 1.0:
                raise ValueError(
                    "generate_labels_features_voca_cqtroll only supports "
                    f"stretch_factor=1.0 (got {st}); use the pyrb variant "
                    "for time-stretch augmentation."
                )

        for song_name, lab_path, mp3_path, save_path in all_list:
            save_path = save_path + '_voca'

            i += 1
            print(pid, "generating features (cqtroll) from ...", os.path.join(mp3_path))
            if i % 10 == 0:
                print(i, ' th song')

            try:
                original_wav, sr = librosa.load(os.path.join(mp3_path), sr=mp3_config['song_hz'])
            except Exception as e:
                print(e)
                print(pid, " load error : %s" % song_name)
                j += 1
                continue

            try:
                chord_info = self.Chord_class.get_converted_chord_full(os.path.join(lab_path))
            except Exception as e:
                print(e)
                print(pid, " chord lab file error : %s" % song_name)
                j += 1
                continue

            # save_path, mp3_string, feature_string, song_name, aug.pt
            result_path = os.path.join(save_path, mp3_str, feature_str, song_name.strip())
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            # Truncate audio to label end (mirrors pyrb path; stretch is fixed at 1.0)
            audio_length = original_wav.shape[0]
            last_sec = chord_info.iloc[-1]['end']
            last_sec_hz = int(last_sec * mp3_config['song_hz'])
            skip_interval_samples = int(mp3_config['skip_interval'] * mp3_config['song_hz'])

            if audio_length + skip_interval_samples < last_sec_hz:
                print('loaded song is too short :', song_name)
                j += 1
                continue
            elif audio_length > last_sec_hz:
                x = original_wav[:last_sec_hz]
            else:
                x = original_wav

            origin_length = x.shape[0]
            origin_length_in_sec = origin_length / mp3_config['song_hz']

            if self.feature_name != FeatureTypes.cqt:
                raise NotImplementedError

            # Single CQT on the (truncated) original audio -- the only
            # expensive step of this pipeline.  Each pitch-shifted version
            # is then a free np.roll on the frequency axis.
            try:
                full_cqt = librosa.cqt(
                    x, sr=sr,
                    n_bins=feature_config['n_bins'],
                    bins_per_octave=feature_config['bins_per_octave'],
                    hop_length=hop,
                )
            except Exception as e:
                print(e)
                print(pid, "full-song cqt error")
                raise RuntimeError()

            # Mirror pyrb structure: stretch outer, shift next, window inner.
            # ``idx`` is reset per (stretch, shift) pair so filenames match
            # the pyrb output exactly.
            song_failed = False
            for stretch_factor in stretch_factors:
                if song_failed:
                    break

                for shift_factor in shift_factors:
                    if song_failed:
                        break
                    idx = 0
                    k += 1
                    current_start_second = 0

                    # get chord list between current_start_second and current+song_length
                    while current_start_second + mp3_config['inst_len'] < origin_length_in_sec:
                        inst_start_sec = current_start_second
                        curSec = current_start_second

                        chord_list = []
                        root_list = []
                        quality_list = []
                        bass_list = []
                        chord_label_list = []

                        # extract chord per 1/self.time_interval
                        while curSec < inst_start_sec + mp3_config['inst_len']:
                            try:
                                available_chords = chord_info.loc[(chord_info['start'] <= curSec) & (chord_info['end'] > curSec + self.time_interval)].copy()
                                if len(available_chords) == 0:
                                    available_chords = chord_info.loc[((chord_info['start'] >= curSec) & (chord_info['start'] <= curSec + self.time_interval)) | ((chord_info['end'] >= curSec) & (chord_info['end'] <= curSec + self.time_interval))].copy()

                                if len(available_chords) == 1:
                                    chord = available_chords['chord_id'].iloc[0]
                                    root = available_chords['root'].iloc[0]
                                    quality = available_chords['quality'].iloc[0]
                                    bass = available_chords['bass'].iloc[0]
                                    chord_label = available_chords['chord_label'].iloc[0]
                                elif len(available_chords) > 1:
                                    max_starts = available_chords.apply(lambda row: max(row['start'], curSec), axis=1)
                                    available_chords['max_start'] = max_starts
                                    min_ends = available_chords.apply(lambda row: min(row.end, curSec + self.time_interval), axis=1)
                                    available_chords['min_end'] = min_ends
                                    chords_lengths = available_chords['min_end'] - available_chords['max_start']
                                    available_chords['chord_length'] = chords_lengths
                                    max_idx = available_chords['chord_length'].idxmax()
                                    chord = available_chords.loc[max_idx, 'chord_id']
                                    root = available_chords.loc[max_idx, 'root']
                                    quality = available_chords.loc[max_idx, 'quality']
                                    bass = available_chords.loc[max_idx, 'bass']
                                    chord_label = available_chords.loc[max_idx, 'chord_label']
                                else:
                                    chord = 169
                                    root = 12  # No chord
                                    quality = 14  # No chord quality
                                    bass = 12  # No bass
                                    chord_label = 'N'
                            except Exception as e:
                                chord = 169
                                root = 12
                                quality = 14
                                bass = 12
                                chord_label = 'N'
                                print(e)
                                print(pid, "no chord")
                                raise RuntimeError()
                            finally:
                                # convert chord by shift factor
                                if chord != 169 and chord != 168:
                                    chord += shift_factor * 14
                                    chord = chord % 168

                                # Apply pitch shifting to root and bass (not quality)
                                if root != 12:  # If not "no chord"
                                    root = (root + shift_factor) % 12
                                if bass != 12:  # If not "no bass"
                                    bass = (bass + shift_factor) % 12

                                # Transpose the full chord label string
                                chord_label = self.Chord_class.transpose_chord_label(
                                    chord_label, shift_factor)

                                chord_list.append(chord)
                                root_list.append(root)
                                quality_list.append(quality)
                                bass_list.append(bass)
                                chord_label_list.append(chord_label)
                                curSec += self.time_interval

                        if len(chord_list) == self.no_of_chord_datapoints_per_sequence:
                            etc = '%.1f_%.1f' % (
                                current_start_second, current_start_second + mp3_config['inst_len'])
                            aug = '%.2f_%i' % (stretch_factor, shift_factor)
                            filename = aug + "_" + str(idx) + ".pt"
                            output_file = os.path.join(result_path, filename)

                            if os.path.exists(output_file):
                                print(f"{pid} Skipping existing file: {filename}")
                                idx += 1
                                total += 1
                            else:
                                try:
                                    sequence_start_time = current_start_second

                                    # Frame index in the precomputed full CQT.
                                    # librosa.cqt centers frame n at sample n*hop,
                                    # so this aligns frame 0 of the segment with
                                    # the audio sample int(sequence_start_time*sr).
                                    start_frame = int(sequence_start_time * mp3_config['song_hz']) // hop
                                    end_frame = start_frame + self.no_of_chord_datapoints_per_sequence

                                    seg_cqt = full_cqt[:, start_frame:end_frame]

                                    if seg_cqt.shape[1] > self.no_of_chord_datapoints_per_sequence:
                                        seg_cqt = seg_cqt[:, :self.no_of_chord_datapoints_per_sequence]

                                    if seg_cqt.shape[1] != self.no_of_chord_datapoints_per_sequence:
                                        print('loaded features length is too short :', song_name)
                                        song_failed = True
                                        j += 1
                                        break

                                    feature = shift_cqt_bins(seg_cqt, int(shift_factor), bins_per_semitone)

                                    result = {
                                        'feature': feature,
                                        'chord': chord_list,
                                        'root': root_list,
                                        'quality': quality_list,
                                        'bass': bass_list,
                                        'original_chord_labels': chord_label_list,
                                        'etc': etc
                                    }

                                    # save_path, mp3_string, feature_string, song_name, aug.pt
                                    torch.save(result, output_file)
                                    idx += 1
                                    total += 1
                                except Exception as e:
                                    print(e)
                                    print(pid, "feature error")
                                    raise RuntimeError()
                        else:
                            print("invalid number of chord datapoints in sequence :", len(chord_list))
                        current_start_second += mp3_config['skip_interval']
        print(pid, "total instances: %d" % total)