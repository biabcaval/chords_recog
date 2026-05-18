#!/usr/bin/env python
# encoding: utf-8
"""
CQT-first variant of ``preprocess_decomposed.py``.

This script is a *parallel* variant of
[scripts/preprocess_decomposed.py](preprocess_decomposed.py) that uses the
``Preprocess.generate_labels_features_voca_cqtroll`` method.  Instead of
calling ``pyrubberband.pitch_shift`` 12 times per song (and computing 12
CQTs), it computes the CQT *once* and synthesizes each pitch-shifted
version by rolling the frequency axis of the CQT.  Output ``.pt`` files
have an identical schema and filename pattern to the pyrb pipeline.

The pyrb pipeline at
[scripts/preprocess_decomposed.py](preprocess_decomposed.py) is left
untouched; both can coexist by writing to different result trees.

Default output layout
---------------------

    <root_dir>/result_cqtroll/<dataset>_voca/<mp3_str>/<feature_str>/<song>/...
    <root_dir>/result_decomposed_cqtroll/<dataset>_voca/<mp3_str>/<feature_str>/<song>/...

The training loader at [data/audio_dataset.py](../data/audio_dataset.py)
expects the decomposed tree at ``<root_dir>/result_decomposed/...``.  To
train on the CQT-roll output without modifying the loader, point the
loader's path to the new tree (recommended: rename or symlink after
preprocessing):

    mv <root_dir>/result_decomposed <root_dir>/result_decomposed.pyrb_backup
    ln -s <root_dir>/result_decomposed_cqtroll <root_dir>/result_decomposed

The intermediate / output subdirectory names can be customised with
``--intermediate_subdir`` and ``--output_subdir`` if you prefer a
different naming convention.
"""

import os
import sys
import argparse
from multiprocessing import Pool

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocess import Preprocess, FeatureTypes
from utils.hparams import HParams

from scripts.preprocess_decomposed import decompose_preprocessed_data


def _substitute_result_root(save_path, intermediate_subdir):
    """Rewrite the trailing ``.../result/<dataset>`` segment of a save_path
    coming from :meth:`Preprocess.get_all_files` to use a different
    intermediate subdirectory name.

    ``Preprocess.get_all_files`` always returns paths of the form
    ``os.path.join(self.root_path, "result", dataset_name)``; we keep
    everything except the literal ``"result"`` token.
    """
    head, dataset = os.path.split(save_path)
    head_root, result_token = os.path.split(head)
    if result_token != 'result':
        return save_path
    return os.path.join(head_root, intermediate_subdir, dataset)


def preprocess_datasets_decomposed_cqtroll(
    config_path,
    root_dir,
    dataset_names,
    num_workers=1,
    force=False,
    log_file=None,
    intermediate_subdir='result_cqtroll',
    output_subdir='result_decomposed_cqtroll',
):
    """Two-step preprocess pipeline using the CQT-roll augmentation method.

    Step 1: ``Preprocess.generate_labels_features_voca_cqtroll`` writes
        full-vocab ``.pt`` files (with structured fields) into
        ``<root_dir>/<intermediate_subdir>/<dataset>_voca/...``.

    Step 2: ``decompose_preprocessed_data`` (reused as-is from the pyrb
        script) converts those into 9-component decomposed ``.pt`` files
        under ``<root_dir>/<output_subdir>/<dataset>_voca/...``.
    """

    print("\n" + "=" * 70)
    print("STEP 1: Standard Preprocessing (170 chords) -- CQT-roll variant")
    print(f"        intermediate subdir: {intermediate_subdir}")
    print("=" * 70)

    config = HParams.load(config_path)
    config.feature['large_voca'] = True
    config.model['num_chords'] = 170

    preprocessor = Preprocess(
        config=config,
        feature_to_use=FeatureTypes.cqt,
        dataset_names=dataset_names,
        root_dir=root_dir,
    )

    print("\nScanning for annotation and audio files...")
    all_files = preprocessor.get_all_files()

    if len(all_files) == 0:
        print("ERROR: No files found!")
        return False

    print(f"Found {len(all_files)} songs to process")

    # Rewrite the save_path 4th element so files land in the cqtroll subdir
    # instead of the default 'result/' tree.
    all_files = [
        (song_name, lab_path, mp3_path,
         _substitute_result_root(save_path, intermediate_subdir))
        for song_name, lab_path, mp3_path, save_path in all_files
    ]

    dataset_counts = {}
    for song_name, lab_path, mp3_path, save_path in all_files:
        dataset_name = os.path.basename(save_path)
        dataset_counts.setdefault(dataset_name, []).append((song_name, lab_path, mp3_path))

    print("\nDataset breakdown:")
    for dataset_name, songs in dataset_counts.items():
        print(f"  {dataset_name}: {len(songs)} songs")

    mp3_config = config.mp3
    feature_config = config.feature
    mp3_string = "%d_%.1f_%.1f" % (
        mp3_config['song_hz'], mp3_config['inst_len'], mp3_config['skip_interval']
    )
    feature_string = "%s_%d_%d_%d" % (
        FeatureTypes.cqt.value, feature_config['n_bins'],
        feature_config['bins_per_octave'], feature_config['hop_length']
    )

    intermediate_dirs = []

    if not force:
        print("\nChecking for existing preprocessed data...")
        needs_processing = []
        for dataset_name in dataset_names:
            result_path = os.path.join(
                root_dir, intermediate_subdir, dataset_name + '_voca',
                mp3_string, feature_string,
            )
            intermediate_dirs.append((dataset_name, result_path))

            if os.path.exists(result_path):
                print(f"  {dataset_name}: Preprocessed data exists")
            else:
                print(f"  {dataset_name}: Need to preprocess")
                needs_processing.append(dataset_name)

        if not needs_processing:
            print("\nAll datasets already preprocessed!")
            all_files = []
        else:
            all_files = [
                f for f in all_files
                if any(dataset in f[3] for dataset in needs_processing)
            ]
    else:
        for dataset_name in dataset_names:
            result_path = os.path.join(
                root_dir, intermediate_subdir, dataset_name + '_voca',
                mp3_string, feature_string,
            )
            intermediate_dirs.append((dataset_name, result_path))

    if all_files:
        print("\n" + "=" * 60)
        print(f"Starting CQT-roll preprocessing ({len(all_files)} songs)...")
        print("=" * 60)

        # Validate cqtroll preconditions once, up-front, so the user gets
        # an immediate error rather than waiting for the first worker to
        # raise inside the Pool.
        preprocessor._validate_cqtroll_config()

        total_written = 0
        total_skipped = 0
        failed_songs = 0

        if num_workers > 1:
            print(f"Using {num_workers} parallel workers")
            with Pool(processes=num_workers) as p:
                iterator = p.imap_unordered(
                    preprocessor.process_one_song_cqtroll,
                    all_files,
                )
                with tqdm(total=len(all_files), desc="Step 1 (cqtroll)",
                          unit="song", smoothing=0.05, dynamic_ncols=True) as pbar:
                    for result in iterator:
                        total_written += result.get('instances_written', 0)
                        total_skipped += result.get('instances_skipped', 0)
                        if result.get('song_failed'):
                            failed_songs += 1
                        pbar.set_postfix(
                            written=total_written,
                            skipped=total_skipped,
                            failed=failed_songs,
                        )
                        pbar.update(1)
        else:
            print("Using single worker (sequential processing)")
            with tqdm(total=len(all_files), desc="Step 1 (cqtroll)",
                      unit="song", smoothing=0.05, dynamic_ncols=True) as pbar:
                for song in all_files:
                    result = preprocessor.process_one_song_cqtroll(song)
                    total_written += result.get('instances_written', 0)
                    total_skipped += result.get('instances_skipped', 0)
                    if result.get('song_failed'):
                        failed_songs += 1
                    pbar.set_postfix(
                        written=total_written,
                        skipped=total_skipped,
                        failed=failed_songs,
                    )
                    pbar.update(1)

        print(
            f"\nStep 1 done: {total_written} instances written, "
            f"{total_skipped} skipped (already existed), "
            f"{failed_songs} songs failed"
        )

    print("\n" + "=" * 70)
    print("STEP 2: Converting to Decomposed Format (9 components)")
    print(f"        output subdir: {output_subdir}")
    print("=" * 70)

    for dataset_name, intermediate_dir in intermediate_dirs:
        if os.path.exists(intermediate_dir):
            output_dir = os.path.join(
                root_dir, output_subdir, dataset_name + '_voca',
                mp3_string, feature_string,
            )

            print(f"\nConverting {dataset_name}...")
            print(f"  Input:  {intermediate_dir}")
            print(f"  Output: {output_dir}")

            ds_log = None
            if log_file:
                from pathlib import Path
                ds_log = str(Path(log_file).with_suffix('')) + f'_{dataset_name}.csv'
            decompose_preprocessed_data(
                intermediate_dir,
                output_dir,
                force=force,
                log_file=ds_log,
            )
        else:
            print(f"\nSkipping {dataset_name} (not found at {intermediate_dir})")

    print("\n" + "=" * 70)
    print("Complete! CQT-roll decomposed data ready for training.")
    print("=" * 70)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess datasets for chord recognition with decomposed '
                    'structure, using the CQT-first augmentation pipeline '
                    '(np.roll on the frequency axis instead of pyrb.pitch_shift).'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='run_config.yaml',
        help='Path to config YAML file (default: run_config.yaml)'
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        default=None,
        help='Root directory containing datasets (default: from config)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['billboard', 'dj_avan_songbook1', 'dj_avan_songbook2'],
        help='Dataset names to preprocess'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of parallel workers (default: 8)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force preprocessing even if data already exists'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='Write a CSV log mapping each unique chord label to its '
             '9 decomposed components (one file per dataset)'
    )
    parser.add_argument(
        '--intermediate_subdir',
        type=str,
        default='result_cqtroll',
        help='Subdir under <root_dir> for the intermediate (Step 1) output '
             '(default: result_cqtroll). Use "result" to drop-in replace the '
             'pyrb output (DESTRUCTIVE: will be skipped/merged via the '
             'existing skip-existing logic).'
    )
    parser.add_argument(
        '--output_subdir',
        type=str,
        default='result_decomposed_cqtroll',
        help='Subdir under <root_dir> for the final decomposed (Step 2) '
             'output (default: result_decomposed_cqtroll). Use '
             '"result_decomposed" to drop-in replace the pyrb output.'
    )

    args = parser.parse_args()

    config = HParams.load(args.config)
    root_dir = args.root_dir if args.root_dir else config.path['root_path']

    success = preprocess_datasets_decomposed_cqtroll(
        config_path=args.config,
        root_dir=root_dir,
        dataset_names=args.datasets,
        num_workers=args.num_workers,
        force=args.force,
        log_file=args.log_file,
        intermediate_subdir=args.intermediate_subdir,
        output_subdir=args.output_subdir,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
