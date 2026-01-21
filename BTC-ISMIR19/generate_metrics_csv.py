#!/usr/bin/env python
"""
Script to generate metrics CSVs for chord recognition experiments.
Generates separate CSVs for RWC and Djavan test sets.
"""

import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import mir_eval
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


def parse_lab_file(filepath):
    """Parse a .lab file and return intervals and labels."""
    intervals = []
    labels = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1])
                chord = ' '.join(parts[2:])
                intervals.append([start, end])
                labels.append(chord)
    return np.array(intervals), labels


def normalize_filename(filename):
    """Normalize filename for matching: lowercase, replace spaces with underscores, remove special chars."""
    normalized = filename.lower()
    normalized = normalized.replace(' ', '_')
    normalized = normalized.replace('-', '_')
    return normalized


def build_gt_filename_map(ground_truth_dir):
    """Build a mapping from normalized filenames to actual ground truth filenames."""
    gt_files = glob.glob(os.path.join(ground_truth_dir, "*.lab"))
    gt_map = {}
    for gt_file in gt_files:
        basename = os.path.basename(gt_file)
        normalized = normalize_filename(basename)
        gt_map[normalized] = gt_file
    return gt_map


def calculate_metrics_for_inference(inference_dir, ground_truth_dir):
    """Calculate all metrics for a single inference directory."""
    inference_files = glob.glob(os.path.join(inference_dir, "*.lab"))
    
    if not inference_files:
        return None
    
    gt_filename_map = build_gt_filename_map(ground_truth_dir)
    all_scores = defaultdict(list)
    
    for inf_file in inference_files:
        track_id = os.path.basename(inf_file)
        normalized_track_id = normalize_filename(track_id)
        
        gt_file = os.path.join(ground_truth_dir, track_id)
        if not os.path.exists(gt_file):
            gt_file = gt_filename_map.get(normalized_track_id)
        
        if gt_file is None or not os.path.exists(gt_file):
            continue
        
        try:
            ref_intervals, ref_labels = parse_lab_file(gt_file)
            est_intervals, est_labels = parse_lab_file(inf_file)
            
            if len(ref_intervals) == 0 or len(est_intervals) == 0:
                continue
            
            scores = mir_eval.chord.evaluate(ref_intervals, ref_labels, 
                                            est_intervals, est_labels)
            
            for metric, value in scores.items():
                all_scores[metric].append(value)
            
            all_scores['num_predictions'].append(len(est_labels))
            all_scores['num_ground_truth'].append(len(ref_labels))
            all_scores['duration_seconds'].append(ref_intervals[-1][1])
            
            est_changes = len(est_labels) / (est_intervals[-1][1] / 60) if est_intervals[-1][1] > 0 else 0
            ref_changes = len(ref_labels) / (ref_intervals[-1][1] / 60) if ref_intervals[-1][1] > 0 else 0
            all_scores['pred_changes_per_min'].append(est_changes)
            all_scores['gt_changes_per_min'].append(ref_changes)
            
        except Exception as e:
            print(f"Error processing {track_id}: {str(e)}")
            continue
    
    if not all_scores:
        return None
    
    mean_metrics = {key: np.mean(values) for key, values in all_scores.items()}
    std_metrics = {f"{key}_std": np.std(values) for key, values in all_scores.items()}
    
    wcsr_metrics = ['root', 'majmin', 'thirds', 'triads', 'tetrads', 'sevenths', 'mirex']
    durations = np.array(all_scores['duration_seconds'])
    total_duration = np.sum(durations)
    
    wcsr_results = {}
    for metric in wcsr_metrics:
        if metric in all_scores:
            scores = np.array(all_scores[metric])
            correct_duration = np.sum(scores * durations)
            wcsr = (correct_duration / total_duration) * 100 if total_duration > 0 else 0
            wcsr_results[f'{metric}_wcsr'] = wcsr
    
    combined_metrics = {**mean_metrics, **std_metrics, **wcsr_results}
    combined_metrics['num_tracks'] = len(all_scores['root'])
    
    return combined_metrics


def get_test_dataset(inference_dir_name):
    """Determine test dataset based on folder name."""
    if "_test_Rw" in inference_dir_name:
        return "rwc"
    elif "_test_Dj" in inference_dir_name:
        return "djavan"
    elif "_test_Ja" in inference_dir_name:
        return "jaah"
    elif "_test_Bi" in inference_dir_name:
        return "billboard"
    elif "_test_Qu" in inference_dir_name:
        return "queen"
    elif "_test_Ro" in inference_dir_name:
        return "robbiewilliams"
    else:
        return "unknown"


def get_ground_truth_dir(inference_dir_name):
    """Determine ground truth directory based on test dataset in folder name."""
    base_gt_path = "/home/daniel.melo/datasets"
    
    if "_test_Rw" in inference_dir_name:
        return os.path.join(base_gt_path, "rwc/annotations")
    elif "_test_Dj" in inference_dir_name:
        return os.path.join(base_gt_path, "dj_avan/annotations")
    elif "_test_Ja" in inference_dir_name:
        return os.path.join(base_gt_path, "jaah/annotations")
    elif "_test_Bi" in inference_dir_name:
        return os.path.join(base_gt_path, "billboard/annotations")
    elif "_test_Qu" in inference_dir_name:
        return os.path.join(base_gt_path, "queen/annotations")
    elif "_test_Ro" in inference_dir_name:
        return os.path.join(base_gt_path, "robbiewilliams/annotations")
    else:
        print(f"  Warning: Could not determine test dataset from '{inference_dir_name}'")
        return None


def generate_metrics_tables(inference_base_dirs, output_prefix, target_test_set=None):
    """Generate detailed and summary comparison tables for inference directories.
    
    Args:
        inference_base_dirs: List of base directories containing inference results
        output_prefix: Prefix for output CSV files
        target_test_set: If specified, only process experiments with this test set
    """
    inference_dirs = []
    for base_dir in inference_base_dirs:
        if os.path.exists(base_dir):
            dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) 
                   if os.path.isdir(d)]
            inference_dirs.extend(dirs)
    
    inference_dirs = sorted(inference_dirs)
    
    if not inference_dirs:
        print(f"No inference directories found")
        return None, None
    
    print(f"Found {len(inference_dirs)} inference directories")
    print("Calculating metrics...\n")
    
    results = {}
    for inf_dir in inference_dirs:
        dir_name = os.path.basename(inf_dir)
        
        # Check if this directory matches target test set
        test_set = get_test_dataset(dir_name)
        if target_test_set and test_set != target_test_set:
            continue
        
        exp_name = dir_name
        if exp_name.startswith('inference_'):
            exp_name = exp_name.replace('inference_', '', 1)
        elif exp_name.startswith('inferences_'):
            exp_name = exp_name.replace('inferences_', '', 1)
        
        ground_truth_dir = get_ground_truth_dir(dir_name)
        if ground_truth_dir is None:
            continue
            
        print(f"Processing: {dir_name}")
        print(f"  Test set: {test_set}")
        print(f"  Using ground truth: {ground_truth_dir}")
        
        metrics = calculate_metrics_for_inference(inf_dir, ground_truth_dir)
        
        if metrics:
            results[exp_name] = metrics
            print(f"  ✓ Processed {metrics['num_tracks']} tracks")
        else:
            print(f"  ✗ No valid tracks found")
    
    if not results:
        print("No results to display")
        return None, None
    
    df_raw = pd.DataFrame(results).T
    
    # DETAILED TABLE
    metric_order = [
        'root', 'majmin', 'thirds', 'triads', 'tetrads', 'sevenths',
        'root_wcsr', 'majmin_wcsr', 'thirds_wcsr', 'triads_wcsr', 'tetrads_wcsr', 'sevenths_wcsr', 'mirex_wcsr',
        'overseg', 'underseg', 'seg',
        'num_tracks', 'duration_seconds',
        'num_predictions', 'num_ground_truth',
        'pred_changes_per_min', 'gt_changes_per_min'
    ]
    
    available_metrics = [m for m in metric_order if m in df_raw.columns]
    
    all_columns = []
    for metric in available_metrics:
        all_columns.append(metric)
        if f"{metric}_std" in df_raw.columns:
            all_columns.append(f"{metric}_std")
    
    df_detailed = df_raw[all_columns].copy()
    
    column_names_detailed = {
        'root': 'Root Accuracy (Mean)',
        'root_std': 'Root Accuracy (Std)',
        'majmin': 'Maj/Min Accuracy (Mean)',
        'majmin_std': 'Maj/Min Accuracy (Std)',
        'thirds': 'Thirds Accuracy (Mean)',
        'thirds_std': 'Thirds Accuracy (Std)',
        'triads': 'Triads Accuracy (Mean)',
        'triads_std': 'Triads Accuracy (Std)',
        'tetrads': 'Tetrads Accuracy (Mean)',
        'tetrads_std': 'Tetrads Accuracy (Std)',
        'sevenths': 'Sevenths Accuracy (Mean)',
        'sevenths_std': 'Sevenths Accuracy (Std)',
        'root_wcsr': 'Root WCSR (%)',
        'majmin_wcsr': 'Maj/Min WCSR (%)',
        'thirds_wcsr': 'Thirds WCSR (%)',
        'triads_wcsr': 'Triads WCSR (%)',
        'tetrads_wcsr': 'Tetrads WCSR (%)',
        'sevenths_wcsr': 'Sevenths WCSR (%)',
        'mirex_wcsr': 'MIREX WCSR (%)',
        'overseg': 'Over-segmentation (Mean)',
        'overseg_std': 'Over-segmentation (Std)',
        'underseg': 'Under-segmentation (Mean)',
        'underseg_std': 'Under-segmentation (Std)',
        'seg': 'Segmentation (Mean)',
        'seg_std': 'Segmentation (Std)',
        'num_tracks': 'Number of Tracks',
        'duration_seconds': 'Avg Duration (s) (Mean)',
        'duration_seconds_std': 'Avg Duration (s) (Std)',
        'num_predictions': 'Avg Predictions (Mean)',
        'num_predictions_std': 'Avg Predictions (Std)',
        'num_ground_truth': 'Avg Ground Truth (Mean)',
        'num_ground_truth_std': 'Avg Ground Truth (Std)',
        'pred_changes_per_min': 'Pred Changes/min (Mean)',
        'pred_changes_per_min_std': 'Pred Changes/min (Std)',
        'gt_changes_per_min': 'GT Changes/min (Mean)',
        'gt_changes_per_min_std': 'GT Changes/min (Std)'
    }
    
    df_detailed = df_detailed.rename(columns=column_names_detailed)
    
    # Format numeric columns for detailed table
    for col in df_detailed.columns:
        if 'Number of Tracks' in col:
            df_detailed[col] = df_detailed[col].apply(lambda x: f"{x:.0f}")
        elif 'WCSR' in col:
            df_detailed[col] = df_detailed[col].apply(lambda x: f"{x:.2f}")
        elif 'Accuracy' in col or 'segmentation' in col.lower() or 'Segmentation' in col:
            df_detailed[col] = df_detailed[col].apply(lambda x: f"{x:.4f}")
        elif 'Changes' in col or 'Duration' in col or 'Predictions' in col or 'Ground Truth' in col:
            df_detailed[col] = df_detailed[col].apply(lambda x: f"{x:.2f}")
    
    # SUMMARY TABLE
    summary_base_metrics = ['root', 'thirds', 'majmin', 'triads', 'sevenths', 'tetrads', 'mirex']
    summary_wcsr_metrics = ['root_wcsr', 'thirds_wcsr', 'majmin_wcsr', 'triads_wcsr', 'sevenths_wcsr', 'tetrads_wcsr', 'mirex_wcsr']
    
    summary_columns = []
    for metric in summary_base_metrics:
        if metric in df_raw.columns:
            summary_columns.append(metric)
            if f"{metric}_std" in df_raw.columns:
                summary_columns.append(f"{metric}_std")
    
    for metric in summary_wcsr_metrics:
        if metric in df_raw.columns:
            summary_columns.append(metric)
    
    df_summary = df_raw[summary_columns].copy()
    
    column_names_summary = {
        'root': 'Root (Mean)',
        'root_std': 'Root (Std)',
        'thirds': 'Thirds (Mean)',
        'thirds_std': 'Thirds (Std)',
        'majmin': 'Maj/Min (Mean)',
        'majmin_std': 'Maj/Min (Std)',
        'triads': 'Triads (Mean)',
        'triads_std': 'Triads (Std)',
        'sevenths': 'Sevenths (Mean)',
        'sevenths_std': 'Sevenths (Std)',
        'tetrads': 'Tetrads (Mean)',
        'tetrads_std': 'Tetrads (Std)',
        'mirex': 'MIREX (Mean)',
        'mirex_std': 'MIREX (Std)',
        'root_wcsr': 'Root WCSR',
        'thirds_wcsr': 'Thirds WCSR',
        'majmin_wcsr': 'Maj/Min WCSR',
        'triads_wcsr': 'Triads WCSR',
        'sevenths_wcsr': 'Sevenths WCSR',
        'tetrads_wcsr': 'Tetrads WCSR',
        'mirex_wcsr': 'MIREX WCSR'
    }
    
    df_summary = df_summary.rename(columns=column_names_summary)
    
    # Format numeric columns for summary table
    for col in df_summary.columns:
        if 'WCSR' in col:
            df_summary[col] = df_summary[col].apply(lambda x: f"{x:.2f}%")
        else:
            df_summary[col] = df_summary[col].apply(lambda x: f"{x:.4f}")
    
    return df_detailed, df_summary


def main():
    """Main function to generate all metrics CSVs."""
    
    # Base directories
    inferences_rwc = "/home/daniel.melo/BTC_ORIGINAL/BTC-ISMIR19/inferences"
    inferences_dja = "/home/daniel.melo/BTC_ORIGINAL/BTC-ISMIR19/inferences_dja"
    output_base = "/home/daniel.melo/BTC_ORIGINAL/BTC-ISMIR19/metrics_results"
    
    os.makedirs(output_base, exist_ok=True)
    
    # ===== RWC Test Set =====
    print("=" * 80)
    print("GENERATING METRICS FOR RWC TEST SET")
    print("=" * 80)
    
    detailed_rwc, summary_rwc = generate_metrics_tables(
        inference_base_dirs=[inferences_rwc],
        output_prefix="rwc",
        target_test_set="rwc"
    )
    
    if detailed_rwc is not None and summary_rwc is not None:
        detailed_path = os.path.join(output_base, "metrics_detailed_rwc.csv")
        summary_path = os.path.join(output_base, "metrics_summary_rwc.csv")
        
        detailed_rwc.to_csv(detailed_path)
        summary_rwc.to_csv(summary_path)
        
        print(f"\n✓ RWC Detailed metrics saved to: {detailed_path}")
        print(f"✓ RWC Summary metrics saved to: {summary_path}")
        print("\nRWC Summary Table:")
        print(summary_rwc.to_string())
    
    # ===== Djavan Test Set =====
    print("\n" + "=" * 80)
    print("GENERATING METRICS FOR DJAVAN TEST SET")
    print("=" * 80)
    
    detailed_dja, summary_dja = generate_metrics_tables(
        inference_base_dirs=[inferences_dja],
        output_prefix="djavan",
        target_test_set="djavan"
    )
    
    if detailed_dja is not None and summary_dja is not None:
        detailed_path = os.path.join(output_base, "metrics_detailed_djavan.csv")
        summary_path = os.path.join(output_base, "metrics_summary_djavan.csv")
        
        detailed_dja.to_csv(detailed_path)
        summary_dja.to_csv(summary_path)
        
        print(f"\n✓ Djavan Detailed metrics saved to: {detailed_path}")
        print(f"✓ Djavan Summary metrics saved to: {summary_path}")
        print("\nDjavan Summary Table:")
        print(summary_dja.to_string())
    
    # ===== Combined metrics (all test sets) =====
    print("\n" + "=" * 80)
    print("GENERATING COMBINED METRICS (ALL TEST SETS)")
    print("=" * 80)
    
    detailed_all, summary_all = generate_metrics_tables(
        inference_base_dirs=[inferences_rwc, inferences_dja],
        output_prefix="all",
        target_test_set=None  # Include all
    )
    
    if detailed_all is not None and summary_all is not None:
        detailed_path = os.path.join(output_base, "metrics_detailed_all.csv")
        summary_path = os.path.join(output_base, "metrics_summary_all.csv")
        
        detailed_all.to_csv(detailed_path)
        summary_all.to_csv(summary_path)
        
        print(f"\n✓ All Detailed metrics saved to: {detailed_path}")
        print(f"✓ All Summary metrics saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("METRICS GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll CSV files saved to: {output_base}")


if __name__ == "__main__":
    main()
