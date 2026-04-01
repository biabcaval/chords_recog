#!/usr/bin/env python
"""
generate_detailed_metrics.py
============================

Avaliacao detalhada frame-a-frame de reconhecimento de acordes, calculando
Accuracy, Precision, Recall, F1 e Matriz de Confusao ao nivel do acorde
completo (full chord).

Os chord labels sao canonicalizados via decompose + reassemble (bemois ->
sustenidos, bass normalizado, extensoes padronizadas) para garantir
comparacao justa entre GT e inferencia.

Metodologia
-----------
1. Le pares de arquivos .lab (inferencia vs ground truth), fazendo match
   por nome de arquivo (case-insensitive, espacos e hifens normalizados).
2. Amostra ambos os .lab numa grade temporal fixa (default 100 fps = 10 ms).
3. Canonicaliza cada chord label via ChordDecomposer + ChordReassembler.
4. Agrega os frames de todas as tracks e calcula metricas via scikit-learn.

Saidas geradas
--------------
CSVs:
  {prefix}_per_track.csv                    - accuracy por track
  {prefix}_full_chord_per_class.csv         - P / R / F1 / support por acorde
  {prefix}_full_chord_confusion_matrix.csv  - matriz de confusao
  {prefix}_summary.csv                      - accuracy + P/R/F1 macro/micro/weighted

PNGs (desativavel com --no_plots):
  {prefix}_full_chord_confusion_matrix.png  - heatmap da matriz de confusao
  {prefix}_full_chord_f1_bar.png            - grafico de barras F1 por acorde

Uso
---
    python generate_detailed_metrics.py \\
        --inference_dir ./inferences_decomposed/inference_chordformer_test_Dj1 \\
        --gt_dir /home/daniel.melo/datasets/dj_avan_songbook1/annotations \\
        --output_dir ./detailed_metrics \\
        --prefix BiQuRoRwJaDj2_testDj1
"""

import os
import argparse
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from utils.chord_decomposition import ChordDecomposer, ChordReassembler

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────────────────────────────
# .lab parsing
# ───────────────────────────────────────────────────────────────────────

_DECOMPOSER = ChordDecomposer()
_REASSEMBLER = ChordReassembler()


def parse_lab_file(filepath):
    """Le um arquivo .lab e retorna (intervals, labels)."""
    intervals, labels = [], []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                intervals.append([float(parts[0]), float(parts[1])])
                labels.append(" ".join(parts[2:]))
    return np.array(intervals) if intervals else np.empty((0, 2)), labels


def normalize_filename(filename):
    """Normaliza nome de arquivo para matching flexivel."""
    return filename.lower().replace(" ", "_").replace("-", "_")


def build_gt_filename_map(gt_dir):
    """Constroi mapa {nome_normalizado: caminho_completo} para todos os .lab no diretorio GT."""
    gt_map = {}
    for f in glob.glob(os.path.join(gt_dir, "*.lab")):
        gt_map[normalize_filename(os.path.basename(f))] = f
    return gt_map


# ───────────────────────────────────────────────────────────────────────
# Canonicalization
# ───────────────────────────────────────────────────────────────────────

def canonicalize_chord(label):
    """Canonicaliza um chord label via decompose + reassemble."""
    comp = _DECOMPOSER.decompose(label)
    return _REASSEMBLER.reassemble(comp)


# ───────────────────────────────────────────────────────────────────────
# Frame-level sampling
# ───────────────────────────────────────────────────────────────────────

def sample_labels_at_frames(intervals, labels, fps):
    """Amostra chord labels numa grade temporal fixa."""
    if len(intervals) == 0:
        return np.array([]), []
    end_time = intervals[-1, 1]
    n_frames = int(np.ceil(end_time * fps))
    if n_frames == 0:
        return np.array([]), []
    times = np.arange(n_frames) / fps
    sampled = []
    idx = 0
    for t in times:
        while idx < len(intervals) - 1 and t >= intervals[idx, 1]:
            idx += 1
        if intervals[idx, 0] <= t < intervals[idx, 1]:
            sampled.append(labels[idx])
        else:
            sampled.append("N")
    return times, sampled


# ───────────────────────────────────────────────────────────────────────
# Metrics computation
# ───────────────────────────────────────────────────────────────────────

def compute_frame_metrics(inference_dir, gt_dir, fps=100):
    """Coleta chord labels canonicalizados frame-a-frame de todas as tracks casadas."""
    inference_files = glob.glob(os.path.join(inference_dir, "*.lab"))
    if not inference_files:
        return None

    gt_map = build_gt_filename_map(gt_dir)

    all_ref_chords, all_est_chords = [], []
    track_stats = []

    for inf_file in sorted(inference_files):
        track_id = os.path.basename(inf_file)
        norm_id = normalize_filename(track_id)

        gt_file = os.path.join(gt_dir, track_id)
        if not os.path.exists(gt_file):
            gt_file = gt_map.get(norm_id)
        if gt_file is None or not os.path.exists(gt_file):
            print(f"  [skip] no GT for {track_id}")
            continue

        try:
            ref_intervals, ref_labels = parse_lab_file(gt_file)
            est_intervals, est_labels = parse_lab_file(inf_file)
            if len(ref_intervals) == 0 or len(est_intervals) == 0:
                continue

            common_end = min(ref_intervals[-1, 1], est_intervals[-1, 1])
            if common_end <= 0:
                continue

            _, ref_sampled = sample_labels_at_frames(ref_intervals, ref_labels, fps)
            _, est_sampled = sample_labels_at_frames(est_intervals, est_labels, fps)

            n = min(len(ref_sampled), len(est_sampled))
            ref_sampled = ref_sampled[:n]
            est_sampled = est_sampled[:n]

            ref_canon = [canonicalize_chord(l) for l in ref_sampled]
            est_canon = [canonicalize_chord(l) for l in est_sampled]

            all_ref_chords.extend(ref_canon)
            all_est_chords.extend(est_canon)

            chord_acc = accuracy_score(ref_canon, est_canon)
            track_stats.append({
                "track": os.path.splitext(track_id)[0],
                "frames": n,
                "duration_s": n / fps,
                "chord_accuracy": chord_acc,
            })

        except Exception as e:
            print(f"  [error] {track_id}: {e}")
            continue

    if not track_stats:
        return None

    return all_ref_chords, all_est_chords, track_stats


def classification_metrics(y_true, y_pred, class_order=None):
    """Calcula accuracy + Precision/Recall/F1 por classe e agregados."""
    labels = class_order
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    present = sorted(set(y_true) | set(y_pred))
    labels = [l for l in labels if l in present]

    acc = accuracy_score(y_true, y_pred)

    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    df = pd.DataFrame({
        "class": labels,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": sup.astype(int),
    }).set_index("class")

    def _avg(avg_type):
        pa, ra, fa, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=avg_type, zero_division=0
        )
        return {"precision": pa, "recall": ra, "f1": fa}

    return acc, df, _avg("macro"), _avg("micro"), _avg("weighted")


def build_confusion(y_true, y_pred, class_order=None):
    """Constroi uma matriz de confusao como DataFrame rotulado."""
    labels = class_order
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    present = sorted(set(y_true) | set(y_pred))
    labels = [l for l in labels if l in present]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


# ───────────────────────────────────────────────────────────────────────
# Plotting
# ───────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm_df, title, output_path, figsize=None, annot=True):
    """Salva um heatmap da matriz de confusao como PNG."""
    n = len(cm_df)
    if figsize is None:
        side = max(6, n * 0.55)
        figsize = (side, side)
    fig, ax = plt.subplots(figsize=figsize)

    fmt = "d" if cm_df.values.max() < 1e6 else ".1e"
    annot_kws = {"size": max(6, 10 - n // 8)}

    sns.heatmap(
        cm_df, annot=annot and n <= 40, fmt=fmt, cmap="Blues",
        linewidths=0.4, ax=ax, annot_kws=annot_kws,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.tick_params(axis="both", labelsize=max(6, 10 - n // 10))
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_per_class_f1(df_per_class, title, output_path):
    """Grafico de barras horizontal do F1 por classe."""
    df_sorted = df_per_class.sort_values("f1", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df_sorted) * 0.35)))
    colors = plt.cm.RdYlGn(df_sorted["f1"].values)
    ax.barh(df_sorted.index, df_sorted["f1"], color=colors, edgecolor="gray", linewidth=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 Score", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax.text(row["f1"] + 0.01, i, f"{row['f1']:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Frame-level Accuracy / P / R / F1 and Confusion Matrix "
                    "for full chord recognition."
    )
    p.add_argument("--inference_dir", required=True, help="Dir with predicted .lab files")
    p.add_argument("--gt_dir", required=True, help="Dir with ground-truth .lab files")
    p.add_argument("--output_dir", default="./detailed_metrics", help="Output directory")
    p.add_argument("--prefix", default="detailed", help="Filename prefix for outputs")
    p.add_argument("--fps", type=int, default=100,
                   help="Sampling rate in frames per second (default 100 = 10 ms)")
    p.add_argument("--no_plots", action="store_true", help="Skip PNG generation")
    return p.parse_args()


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not os.path.isdir(args.inference_dir):
        raise FileNotFoundError(f"Inference dir not found: {args.inference_dir}")
    if not os.path.isdir(args.gt_dir):
        raise FileNotFoundError(f"GT dir not found: {args.gt_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"  Inference dir : {args.inference_dir}")
    print(f"  Ground-truth  : {args.gt_dir}")
    print(f"  FPS           : {args.fps}")
    print("=" * 70)

    result = compute_frame_metrics(args.inference_dir, args.gt_dir, fps=args.fps)
    if result is None:
        print("\nNo tracks matched.")
        return

    ref_chords, est_chords, track_stats = result
    total_frames = len(ref_chords)
    n_tracks = len(track_stats)

    print(f"\nMatched {n_tracks} tracks  |  {total_frames:,} frames "
          f"({total_frames / args.fps:.1f} s)\n")

    # ── Per-track stats ──────────────────────────────────────────────
    df_tracks = pd.DataFrame(track_stats).set_index("track")
    track_path = os.path.join(args.output_dir, f"{args.prefix}_per_track.csv")
    df_tracks.to_csv(track_path)
    print(f"  Per-track stats     : {track_path}")

    # ── Full chord metrics ───────────────────────────────────────────
    fc_acc, fc_cls, fc_macro, fc_micro, fc_wtd = classification_metrics(
        ref_chords, est_chords
    )

    fc_cls_path = os.path.join(
        args.output_dir, f"{args.prefix}_full_chord_per_class.csv"
    )
    fc_cls.to_csv(fc_cls_path)
    print(f"  Per-class metrics   : {fc_cls_path}")

    fc_cm = build_confusion(ref_chords, est_chords)
    fc_cm_path = os.path.join(
        args.output_dir, f"{args.prefix}_full_chord_confusion_matrix.csv"
    )
    fc_cm.to_csv(fc_cm_path)
    print(f"  Confusion matrix    : {fc_cm_path}")

    if not args.no_plots:
        plot_confusion_matrix(
            fc_cm, "Confusion Matrix — Full Chord",
            os.path.join(
                args.output_dir,
                f"{args.prefix}_full_chord_confusion_matrix.png",
            ),
        )
        plot_per_class_f1(
            fc_cls, "Per-class F1 — Full Chord",
            os.path.join(
                args.output_dir, f"{args.prefix}_full_chord_f1_bar.png"
            ),
        )

    # ── Summary ──────────────────────────────────────────────────────
    summary = {
        "num_tracks": n_tracks,
        "total_frames": total_frames,
        "total_duration_s": total_frames / args.fps,
        "chord_accuracy": fc_acc,
        "chord_f1_macro": fc_macro["f1"],
        "chord_f1_micro": fc_micro["f1"],
        "chord_f1_weighted": fc_wtd["f1"],
        "chord_precision_macro": fc_macro["precision"],
        "chord_recall_macro": fc_macro["recall"],
        "chord_precision_micro": fc_micro["precision"],
        "chord_recall_micro": fc_micro["recall"],
        "chord_precision_weighted": fc_wtd["precision"],
        "chord_recall_weighted": fc_wtd["recall"],
        "chord_num_classes": len(fc_cls),
    }

    df_summary = pd.DataFrame([summary])
    summary_path = os.path.join(args.output_dir, f"{args.prefix}_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"  Summary             : {summary_path}")

    # ── Console output ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  FULL CHORD ({len(fc_cls)} distinct classes)")
    print(f"    Accuracy : {fc_acc:.4f}")
    print(f"    Macro    : P={fc_macro['precision']:.4f}  "
          f"R={fc_macro['recall']:.4f}  F1={fc_macro['f1']:.4f}")
    print(f"    Weighted : P={fc_wtd['precision']:.4f}  "
          f"R={fc_wtd['recall']:.4f}  F1={fc_wtd['f1']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
