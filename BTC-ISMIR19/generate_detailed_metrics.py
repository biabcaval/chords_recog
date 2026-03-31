#!/usr/bin/env python
"""
generate_detailed_metrics.py
============================

Avaliacao detalhada frame-a-frame de reconhecimento de acordes, calculando
Accuracy, Precision, Recall, F1 e Matrizes de Confusao para cada um dos
9 componentes do ChordDecomposer (mesmos do pipeline de treino/inferencia):

  * **root**  (13 classes)  – nota fundamental
  * **bass**  (13 classes)  – nota do baixo
  * **triad** (7 classes)   – tríade (maj, min, dim, aug, sus2, sus4)
  * **misc**  (2 classes)   – power chord
  * **6th**   (2 classes)   – extensão de 6ª
  * **7th**   (4 classes)   – tipo de 7ª (maj7, dom7, dim7)
  * **9th**   (4 classes)   – extensão de 9ª
  * **11th**  (3 classes)   – extensão de 11ª
  * **13th**  (3 classes)   – extensão de 13ª

Adicionalmente, calcula uma **quality derivada** (31 classes) que combina
triad + extensões numa classe de shorthand canônica (e.g. "min7", "hdim7").

Metodologia
-----------
1. Lê pares de arquivos .lab (inferência vs ground truth), fazendo match
   por nome de arquivo (case-insensitive, espaços e hifens normalizados).
2. Amostra ambos os .lab numa grade temporal fixa (default 100 fps = 10 ms).
3. Decompõe cada chord label nos 9 componentes via ChordDecomposer
   (mesmo usado no treino), usando exatamente o CHORD_VOCAB do modelo.
4. Agrega os frames de todas as tracks e calcula métricas via scikit-learn.

Saídas geradas
--------------
CSVs (por componente, 9x):
  {prefix}_{comp}_per_class.csv           – P / R / F1 / support por classe
  {prefix}_{comp}_confusion_matrix.csv    – matriz de confusão

CSVs (quality derivada, 1x):
  {prefix}_quality_per_class.csv          – P / R / F1 / support
  {prefix}_quality_confusion_matrix.csv   – matriz de confusão

Resumo:
  {prefix}_per_track.csv                  – accuracy por componente por track
  {prefix}_summary.csv                    – accuracy + P/R/F1 para tudo

PNGs (desativável com --no_plots):
  {prefix}_{comp}_confusion_matrix.png    – heatmap por componente
  {prefix}_{comp}_f1_bar.png              – barras F1 por componente
  {prefix}_quality_confusion_matrix.png   – heatmap quality derivada
  {prefix}_quality_f1_bar.png             – barras F1 quality derivada
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
from utils.chord_decomposition import (
    ChordDecomposer, ChordReassembler, CHORD_VOCAB, COMPONENT_NAMES,
)

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────────

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_TO_SHARP = {
    "Cb": "B", "Db": "C#", "Eb": "D#", "Fb": "E",
    "Gb": "F#", "Ab": "G#", "Bb": "A#",
}

# Mapeamento de shorthands → quality canônica.
# Cobre todos os shorthands definidos em utils/chords.py e
# utils/chord_decomposition.py. Shorthands não listadas aqui são mantidas
# como estão; string vazia → "maj".
QUALITY_CANON = {
    # --- tríades básicas ---
    "maj": "maj",
    "min": "min",
    "dim": "dim",
    "aug": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
    "sus": "sus4",

    # --- power chords ---
    "5": "5",
    "1": "5",
    "pedal": "5",
    "power": "5",

    # --- acordes com 6ª ---
    "maj6": "maj6",
    "min6": "min6",

    # --- acordes com 7ª ---
    "7": "7",
    "maj7": "maj7",
    "min7": "min7",
    "dim7": "dim7",
    "hdim7": "hdim7",
    "hdim": "hdim7",
    "minmaj7": "minmaj7",
    "minmaj": "minmaj7",
    "aug7": "aug7",

    # --- 7ª + suspensão ---
    "7sus2": "7sus2",
    "7sus4": "7sus4",

    # --- acordes com 9ª ---
    "9": "9",
    "maj9": "maj9",
    "min9": "min9",
    "9sus4": "9sus4",

    # --- acordes com 11ª ---
    "11": "11",
    "min11": "min11",
    "maj11": "maj11",

    # --- acordes com 13ª ---
    "13": "13",
    "maj13": "maj13",
    "min13": "min13",

    # --- sem quality explícita → major ---
    "": "maj",
}

# Ordem fixa das classes de root para a matriz de confusão (N primeiro, depois cromático)
ROOT_ORDER = ["N"] + _NOTE_NAMES

# Ordem fixa das classes de quality para a matriz de confusão.
# Agrupa por família (tríades, power, 6ª, 7ª, 9ª, 11ª, 13ª).
QUALITY_ORDER = [
    "N",
    # tríades
    "maj", "min", "dim", "aug", "sus2", "sus4",
    # power
    "5",
    # 6ª
    "maj6", "min6",
    # 7ª
    "7", "maj7", "min7", "dim7", "hdim7", "minmaj7", "aug7",
    # 7ª + suspensão
    "7sus2", "7sus4",
    # 9ª
    "9", "maj9", "min9", "9sus4",
    # 11ª
    "11", "min11", "maj11",
    # 13ª
    "13", "maj13", "min13",
]

# ───────────────────────────────────────────────────────────────────────
# .lab parsing (shared logic with generate_metrics_csv.py)
# ───────────────────────────────────────────────────────────────────────

def parse_lab_file(filepath):
    """Lê um arquivo .lab e retorna (intervals, labels).

    Formato esperado de cada linha: ``start_time  end_time  chord_label``

    Returns
    -------
    intervals : np.ndarray, shape (N, 2)
        Pares [início, fim] em segundos.
    labels : list[str]
        Chord labels correspondentes (ex: "A:min7", "N").
    """
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
    """Normaliza nome de arquivo para matching flexível (lowercase, _ no lugar de espaço/hífen)."""
    return filename.lower().replace(" ", "_").replace("-", "_")


def build_gt_filename_map(gt_dir):
    """Constrói mapa {nome_normalizado → caminho_completo} para todos os .lab no diretório GT."""
    gt_map = {}
    for f in glob.glob(os.path.join(gt_dir, "*.lab")):
        gt_map[normalize_filename(os.path.basename(f))] = f
    return gt_map


# ───────────────────────────────────────────────────────────────────────
# Chord label → (root, quality)  via ChordDecomposer
# ───────────────────────────────────────────────────────────────────────

_DECOMPOSER = ChordDecomposer()
_REASSEMBLER = ChordReassembler()


def _components_to_quality(comp):
    """Mapeia os componentes decompostos para uma quality canônica.

    Usa triad, misc, 6th e 7th (+ presença de 9th/11th/13th) para
    determinar a classe de quality, seguindo a mesma hierarquia de
    shorthands dos datasets padrão (Harte notation).

    Exemplos de mapeamento via componentes::

        triad=dim,  7th=b7         → "hdim7"
        triad=aug,  7th=b7         → "aug7"
        triad=sus4, 7th=b7         → "7sus4"
        triad=maj,  7th=b7, 9th=9  → "9"
        triad=min,  7th=b7, 9th=9  → "min9"
        misc=5                      → "5"
    """
    if comp.get("root", "N") == "N":
        return "N"

    misc = comp.get("misc", "N")
    if misc == "5":
        return "5"

    triad = comp.get("triad", "N")
    if triad == "N":
        return "N"

    ext_6 = comp.get("6th", "N")
    ext_7 = comp.get("7th", "N")
    has_7 = ext_7 != "N"
    has_9 = comp.get("9th", "N") != "N"
    has_11 = comp.get("11th", "N") != "N"
    has_13 = comp.get("13th", "N") != "N"

    # --- 13ª (requer 7ª) ---
    if has_13 and has_7:
        if triad == "maj" and ext_7 == "7":
            return "maj13"
        if triad == "min" and ext_7 == "b7":
            return "min13"
        if triad == "maj" and ext_7 == "b7":
            return "13"

    # --- 11ª (requer 7ª) ---
    if has_11 and has_7:
        if triad == "maj" and ext_7 == "7":
            return "maj11"
        if triad == "min" and ext_7 == "b7":
            return "min11"
        if triad == "maj" and ext_7 == "b7":
            return "11"

    # --- 9ª (requer 7ª) ---
    if has_9 and has_7:
        if triad == "sus4" and ext_7 == "b7":
            return "9sus4"
        if triad == "maj" and ext_7 == "7":
            return "maj9"
        if triad == "min" and ext_7 == "b7":
            return "min9"
        if triad == "maj" and ext_7 == "b7":
            return "9"

    # --- 7ª ---
    if has_7:
        if triad == "sus2":
            return "7sus2"
        if triad == "sus4":
            return "7sus4"
        if ext_7 == "7":
            if triad == "min":
                return "minmaj7"
            return "maj7"
        if ext_7 == "b7":
            if triad == "dim":
                return "hdim7"
            if triad == "aug":
                return "aug7"
            if triad == "min":
                return "min7"
            return "7"
        if ext_7 == "bb7":
            if triad == "dim":
                return "dim7"

    # --- 6ª (sem 7ª) ---
    if ext_6 != "N":
        if triad == "min":
            return "min6"
        return "maj6"

    # --- tríade pura ---
    return triad


def chord_to_root_quality(label):
    """Decompõe um chord label em (root, quality) usando o ChordDecomposer.

    Delega a análise completa do shorthand (parênteses, omissões,
    extensões implícitas, alterações de 5ª, etc.) ao ChordDecomposer
    e depois mapeia os componentes para uma classe de quality canônica.

    Exemplos::

        "A:min7"      → ("A",  "min7")
        "Bb:maj"      → ("A#", "maj")
        "N"           → ("N",  "N")
        "G"           → ("G",  "maj")
        "C:min7(b5)"  → ("C",  "hdim7")   # corretamente identificado
        "C:sus4(b7)"  → ("C",  "7sus4")   # extensão capturada
        "C:maj(#5)"   → ("C",  "aug")     # alteração de 5ª reconhecida
    """
    comp = _DECOMPOSER.decompose(label)
    root = comp["root"]
    quality = _components_to_quality(comp)
    return root, quality


# ───────────────────────────────────────────────────────────────────────
# Frame-level sampling
# ───────────────────────────────────────────────────────────────────────

def sample_labels_at_frames(intervals, labels, fps):
    """Amostra chord labels numa grade temporal fixa.

    Para cada instante t = 0, 1/fps, 2/fps, ... até o fim da última anotação,
    retorna o chord label ativo naquele instante. Frames fora de qualquer
    intervalo recebem "N" (no chord).

    Parameters
    ----------
    intervals : np.ndarray, shape (N, 2)
    labels : list[str]
    fps : int
        Frames por segundo (100 = resolução de 10 ms).

    Returns
    -------
    times : np.ndarray
    sampled : list[str]
    """
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

def _decompose_labels(sampled_labels):
    """Decompõe uma lista de chord labels em componentes + quality + chord canônico.

    Cada label passa por decompose → reassemble para produzir uma forma
    canônica (bemóis → sustenidos, bass normalizado, sem redundâncias).

    Returns
    -------
    components : dict[str, list[str]]
        Mapa {component_name: [label_por_frame]} para cada um dos 9 componentes.
    qualities : list[str]
        Quality derivada (via _components_to_quality) por frame.
    canonical : list[str]
        Chord label canônico (via decompose+reassemble) por frame.
    """
    components = {comp: [] for comp in COMPONENT_NAMES}
    qualities = []
    canonical = []
    for label in sampled_labels:
        comp = _DECOMPOSER.decompose(label)
        for name in COMPONENT_NAMES:
            components[name].append(comp[name])
        qualities.append(_components_to_quality(comp))
        canonical.append(_REASSEMBLER.reassemble(comp))
    return components, qualities, canonical


def compute_frame_metrics(inference_dir, gt_dir, fps=100):
    """Coleta os 9 componentes + quality derivada frame-a-frame de todas as tracks.

    Para cada par (inferência, GT) com mesmo nome de arquivo:
      1. Parseia ambos os .lab
      2. Amostra na resolução ``fps``
      3. Trunca ao menor comprimento comum
      4. Decompõe cada frame nos 9 componentes do ChordDecomposer
         (root, bass, triad, misc, 6th, 7th, 9th, 11th, 13th)
         + quality derivada

    Returns
    -------
    tuple ou None
        (all_ref_comp, all_est_comp, all_ref_quals, all_est_quals,
         all_ref_chords, all_est_chords, track_stats)
        onde all_ref_comp / all_est_comp são dicts {component: list[str]}
        e all_ref_chords / all_est_chords são os chord labels originais.
        Retorna None se nenhuma track foi casada.
    """
    inference_files = glob.glob(os.path.join(inference_dir, "*.lab"))
    if not inference_files:
        return None

    gt_map = build_gt_filename_map(gt_dir)

    all_ref_comp = {comp: [] for comp in COMPONENT_NAMES}
    all_est_comp = {comp: [] for comp in COMPONENT_NAMES}
    all_ref_quals, all_est_quals = [], []
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

            ref_comp, ref_q, ref_canon = _decompose_labels(ref_sampled)
            est_comp, est_q, est_canon = _decompose_labels(est_sampled)

            for comp in COMPONENT_NAMES:
                all_ref_comp[comp].extend(ref_comp[comp])
                all_est_comp[comp].extend(est_comp[comp])
            all_ref_quals.extend(ref_q)
            all_est_quals.extend(est_q)
            all_ref_chords.extend(ref_canon)
            all_est_chords.extend(est_canon)

            stat = {
                "track": os.path.splitext(track_id)[0],
                "frames": n,
                "duration_s": n / fps,
            }
            for comp in COMPONENT_NAMES:
                stat[f"{comp}_accuracy"] = accuracy_score(
                    ref_comp[comp], est_comp[comp]
                )
            stat["quality_accuracy"] = accuracy_score(ref_q, est_q)
            track_stats.append(stat)

        except Exception as e:
            print(f"  [error] {track_id}: {e}")
            continue

    if not track_stats:
        return None

    return (all_ref_comp, all_est_comp, all_ref_quals, all_est_quals,
            all_ref_chords, all_est_chords, track_stats)


def classification_metrics(y_true, y_pred, class_order=None):
    """Calcula accuracy + Precision/Recall/F1 por classe e agregados.

    Parameters
    ----------
    y_true, y_pred : list[str]
        Labels de referência e estimados.
    class_order : list[str], optional
        Ordem das classes (filtra apenas as presentes nos dados).

    Returns
    -------
    accuracy : float
    df_per_class : pd.DataFrame
        Colunas: precision, recall, f1, support — indexado por classe.
    macro, micro, weighted : dict
        Cada um com chaves "precision", "recall", "f1".
    """
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
    """Constrói uma matriz de confusão como DataFrame rotulado (linhas=GT, colunas=pred)."""
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
    """Salva um heatmap da matriz de confusão como PNG (anotações desativadas se > 40 classes)."""
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
    """Gráfico de barras horizontal do F1 por classe, colorido de vermelho (0) a verde (1)."""
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
        description="Frame-level Accuracy / P / R / F1 and Confusion Matrices "
                    "for chord recognition, at Root and Quality granularity."
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

    (all_ref_comp, all_est_comp, ref_quals, est_quals,
     ref_chords, est_chords, track_stats) = result
    total_frames = len(all_ref_comp["root"])
    n_tracks = len(track_stats)

    print(f"\nMatched {n_tracks} tracks  |  {total_frames:,} frames "
          f"({total_frames / args.fps:.1f} s)\n")

    # ── Per-track stats ──────────────────────────────────────────────
    df_tracks = pd.DataFrame(track_stats).set_index("track")
    track_path = os.path.join(args.output_dir, f"{args.prefix}_per_track.csv")
    df_tracks.to_csv(track_path)
    print(f"  Per-track stats     : {track_path}")

    # ── Per-component metrics (9 components from CHORD_VOCAB) ────────
    comp_results = {}

    for comp in COMPONENT_NAMES:
        class_order = CHORD_VOCAB[comp]
        acc, cls_df, macro, micro, wtd = classification_metrics(
            all_ref_comp[comp], all_est_comp[comp], class_order=class_order
        )
        comp_results[comp] = {
            "acc": acc, "macro": macro, "micro": micro, "weighted": wtd,
        }

        cls_path = os.path.join(
            args.output_dir, f"{args.prefix}_{comp}_per_class.csv"
        )
        cls_df.to_csv(cls_path)
        print(f"  {comp:>5s} per-class    : {cls_path}")

        cm_df = build_confusion(
            all_ref_comp[comp], all_est_comp[comp], class_order=class_order
        )
        cm_path = os.path.join(
            args.output_dir, f"{args.prefix}_{comp}_confusion_matrix.csv"
        )
        cm_df.to_csv(cm_path)

        if not args.no_plots:
            plot_confusion_matrix(
                cm_df, f"Confusion Matrix — {comp}",
                os.path.join(
                    args.output_dir,
                    f"{args.prefix}_{comp}_confusion_matrix.png",
                ),
            )
            plot_per_class_f1(
                cls_df, f"Per-class F1 — {comp}",
                os.path.join(
                    args.output_dir, f"{args.prefix}_{comp}_f1_bar.png"
                ),
            )

    # ── Quality derivada (retrocompatível) ───────────────────────────
    qual_acc, qual_cls, qual_macro, qual_micro, qual_wtd = classification_metrics(
        ref_quals, est_quals, class_order=QUALITY_ORDER
    )
    qual_cls_path = os.path.join(
        args.output_dir, f"{args.prefix}_quality_per_class.csv"
    )
    qual_cls.to_csv(qual_cls_path)
    print(f"  quality per-class   : {qual_cls_path}")

    qual_cm = build_confusion(ref_quals, est_quals, class_order=QUALITY_ORDER)
    qual_cm_path = os.path.join(
        args.output_dir, f"{args.prefix}_quality_confusion_matrix.csv"
    )
    qual_cm.to_csv(qual_cm_path)

    if not args.no_plots:
        plot_confusion_matrix(
            qual_cm, "Confusion Matrix — Quality (derived)",
            os.path.join(
                args.output_dir,
                f"{args.prefix}_quality_confusion_matrix.png",
            ),
        )
        plot_per_class_f1(
            qual_cls, "Per-class F1 — Quality (derived)",
            os.path.join(
                args.output_dir, f"{args.prefix}_quality_f1_bar.png"
            ),
        )

    # ── Full chord analysis (acorde completo, sem PNG) ─────────────
    fc_acc, fc_cls, fc_macro, fc_micro, fc_wtd = classification_metrics(
        ref_chords, est_chords
    )
    fc_cls_path = os.path.join(
        args.output_dir, f"{args.prefix}_full_chord_per_class.csv"
    )
    fc_cls.to_csv(fc_cls_path)
    print(f"  full chord per-class: {fc_cls_path}")

    fc_cm = build_confusion(ref_chords, est_chords)
    fc_cm_path = os.path.join(
        args.output_dir, f"{args.prefix}_full_chord_confusion_matrix.csv"
    )
    fc_cm.to_csv(fc_cm_path)
    print(f"  full chord conf mat : {fc_cm_path}")

    # ── Summary ──────────────────────────────────────────────────────
    summary = {
        "num_tracks": n_tracks,
        "total_frames": total_frames,
        "total_duration_s": total_frames / args.fps,
    }
    for comp in COMPONENT_NAMES:
        cr = comp_results[comp]
        summary[f"{comp}_accuracy"] = cr["acc"]
        summary[f"{comp}_f1_macro"] = cr["macro"]["f1"]
        summary[f"{comp}_f1_weighted"] = cr["weighted"]["f1"]
        summary[f"{comp}_precision_macro"] = cr["macro"]["precision"]
        summary[f"{comp}_recall_macro"] = cr["macro"]["recall"]
        summary[f"{comp}_precision_weighted"] = cr["weighted"]["precision"]
        summary[f"{comp}_recall_weighted"] = cr["weighted"]["recall"]

    summary["quality_accuracy"] = qual_acc
    summary["quality_f1_macro"] = qual_macro["f1"]
    summary["quality_f1_weighted"] = qual_wtd["f1"]
    summary["quality_precision_macro"] = qual_macro["precision"]
    summary["quality_recall_macro"] = qual_macro["recall"]
    summary["quality_precision_weighted"] = qual_wtd["precision"]
    summary["quality_recall_weighted"] = qual_wtd["recall"]

    summary["full_chord_accuracy"] = fc_acc
    summary["full_chord_f1_macro"] = fc_macro["f1"]
    summary["full_chord_f1_weighted"] = fc_wtd["f1"]
    summary["full_chord_precision_weighted"] = fc_wtd["precision"]
    summary["full_chord_recall_weighted"] = fc_wtd["recall"]
    summary["full_chord_num_classes"] = len(fc_cls)

    df_summary = pd.DataFrame([summary])
    summary_path = os.path.join(args.output_dir, f"{args.prefix}_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"  Summary             : {summary_path}")

    # ── Console output ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    for comp in COMPONENT_NAMES:
        cr = comp_results[comp]
        print(f"  {comp.upper()}")
        print(f"    Accuracy : {cr['acc']:.4f}")
        print(f"    Macro    : P={cr['macro']['precision']:.4f}  "
              f"R={cr['macro']['recall']:.4f}  F1={cr['macro']['f1']:.4f}")
        print(f"    Weighted : P={cr['weighted']['precision']:.4f}  "
              f"R={cr['weighted']['recall']:.4f}  "
              f"F1={cr['weighted']['f1']:.4f}")
        print()

    print("  QUALITY (derived)")
    print(f"    Accuracy : {qual_acc:.4f}")
    print(f"    Macro    : P={qual_macro['precision']:.4f}  "
          f"R={qual_macro['recall']:.4f}  F1={qual_macro['f1']:.4f}")
    print(f"    Weighted : P={qual_wtd['precision']:.4f}  "
          f"R={qual_wtd['recall']:.4f}  F1={qual_wtd['f1']:.4f}")
    print()
    print(f"  FULL CHORD ({len(fc_cls)} distinct classes)")
    print(f"    Accuracy : {fc_acc:.4f}")
    print(f"    Macro    : P={fc_macro['precision']:.4f}  "
          f"R={fc_macro['recall']:.4f}  F1={fc_macro['f1']:.4f}")
    print(f"    Weighted : P={fc_wtd['precision']:.4f}  "
          f"R={fc_wtd['recall']:.4f}  F1={fc_wtd['f1']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
