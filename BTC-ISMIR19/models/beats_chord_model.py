# encoding: utf-8
"""
BEATs frontend for chord recognition with structure decomposition.

This module wires Microsoft's frozen BEATs audio backbone into the project's
9-component decomposed chord-recognition pipeline. It mirrors the multi-head
output style of :mod:`models.btc_model_decomposed` so the same losses,
inference utilities and metrics apply.

The backbone (~90M params) is not on PyPI. It must be cloned from
``microsoft/unilm`` (the ``beats`` subdirectory) and a checkpoint such as
``BEATs_iter3_plus_AS2M.pt`` provided separately. Because training happens on
*pre-extracted embeddings* (see ``scripts/preextract_beats_embeddings.py``),
the heads and the full :class:`BEATsChordDecomposer` can be constructed
WITHOUT the checkpoint, which keeps unit/smoke tests and embedding-based
training free of the heavy backbone dependency.
"""

import os
import sys
import importlib

import torch
import torch.nn as nn

from models.btc_model_decomposed import MultiTaskLoss
from utils.chord_decomposition import COMPONENT_NAMES, CHORD_VOCAB

# Native embedding dimension of BEATs_iter3_plus_AS2M (12-layer, 768-wide ViT).
BEATS_EMBED_DIM = 768


def load_beats_backbone(checkpoint_path, source_path=None, freeze=True,
                        device="cpu"):
    """Load the frozen BEATs backbone from a cloned source tree + checkpoint.

    Args:
        checkpoint_path: Path to a BEATs ``.pt`` checkpoint (e.g.
            ``BEATs_iter3_plus_AS2M.pt``). The file must contain ``cfg`` and
            ``model`` keys, as published by Microsoft.
        source_path: Path to the cloned ``unilm/beats`` directory that holds
            ``BEATs.py``. Required because BEATs is not pip-installable. When
            ``None``, the import is attempted from the current ``sys.path``.
        freeze: If True, set ``requires_grad=False`` on all backbone params and
            put it in eval mode (recommended for our small dataset).
        device: Device to map the loaded weights onto.

    Returns:
        The loaded ``BEATs`` module.

    Raises:
        FileNotFoundError: If ``checkpoint_path`` / ``source_path`` are missing.
        ImportError: If the BEATs source cannot be imported, with an actionable
            message describing how to obtain it.
    """
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "BEATs checkpoint not found at "
            f"'{checkpoint_path}'. Download e.g. 'BEATs_iter3_plus_AS2M.pt' "
            "from the Microsoft UniLM release and pass its path via "
            "--beats_checkpoint."
        )

    if source_path is not None:
        if not os.path.isdir(source_path):
            raise FileNotFoundError(
                f"BEATs source directory not found at '{source_path}'. Clone it "
                "with `git clone https://github.com/microsoft/unilm.git` and "
                "point --beats_source at the 'unilm/beats' subdirectory."
            )
        if source_path not in sys.path:
            sys.path.insert(0, source_path)

    try:
        beats_module = importlib.import_module("BEATs")
        BEATs = beats_module.BEATs
        BEATsConfig = beats_module.BEATsConfig
    except ImportError as exc:
        raise ImportError(
            "Could not import the BEATs backbone. BEATs is not on PyPI; clone "
            "`https://github.com/microsoft/unilm.git` and pass the path to its "
            "'beats' subdirectory via --beats_source (the directory containing "
            f"BEATs.py). Original error: {exc}"
        ) from exc

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = BEATsConfig(checkpoint["cfg"])
    backbone = BEATs(cfg)
    backbone.load_state_dict(checkpoint["model"])
    backbone = backbone.to(device)

    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()

    return backbone


@torch.no_grad()
def extract_beats_embeddings(backbone, waveform, device="cpu"):
    """Extract frame-level BEATs embeddings for a single mono waveform.

    Args:
        backbone: A loaded BEATs module (frozen).
        waveform: 1D float tensor of 16kHz mono audio.
        device: Device to run extraction on.

    Returns:
        Tensor of shape ``(n_patches, BEATS_EMBED_DIM)`` (time preserved, no
        mean-pool). The patch rate (~6.25 patches/s for the default config)
        is derived from the audio length downstream rather than hardcoded.
    """
    if waveform.dim() == 1:
        audio = waveform.unsqueeze(0)
    else:
        audio = waveform
    audio = audio.to(device)
    padding_mask = torch.zeros(audio.shape[0], audio.shape[1], dtype=torch.bool, device=device)
    embeddings = backbone.extract_features(audio, padding_mask=padding_mask)[0]
    return embeddings.squeeze(0).cpu()


class LinearClassifier(nn.Module):
    """Single linear projection applied per BEATs patch (paper baseline)."""

    def __init__(self, input_dim=BEATS_EMBED_DIM, num_classes=170, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: (batch, n_patches, input_dim) -> (batch, n_patches, num_classes)
        return self.classifier(self.dropout(x))


class MLPClassifier(nn.Module):
    """Two-layer MLP head with ReLU + dropout applied per BEATs patch."""

    def __init__(self, input_dim=BEATS_EMBED_DIM, hidden_dim=256, num_classes=170,
                 dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (batch, n_patches, input_dim) -> (batch, n_patches, num_classes)
        return self.classifier(x)


class BEATsChordDecomposer(nn.Module):
    """Multi-head chord decomposer on top of BEATs frame-level embeddings.

    Produces one logits tensor per component, shape
    ``(batch, n_patches, vocab_size)``. The lightweight heads are the only
    trainable part; the BEATs backbone is frozen (and usually not even loaded,
    since training consumes pre-extracted embeddings).

    Args:
        input_dim: Embedding width fed to the heads (768 for BEATs).
        head_type: ``'linear'`` or ``'mlp'``.
        hidden_dim: Hidden width for the MLP heads.
        dropout: Dropout probability inside each head.
        class_weights: Optional dict mapping component -> class-weight tensor.
        component_weights: Optional dict of per-component loss weights.
        focal_gamma: Focal-loss focusing parameter (0 = standard CE).
        backbone: Optional pre-loaded frozen BEATs module. When provided,
            ``forward`` accepts raw waveforms ``(batch, samples)`` and extracts
            embeddings internally (for end-to-end inference). When ``None``,
            ``forward`` expects pre-extracted embeddings ``(batch, n_patches,
            input_dim)``.
        probs_out: If True, ``forward`` returns the raw logits dict.
    """

    def __init__(self, input_dim=BEATS_EMBED_DIM, head_type="linear",
                 hidden_dim=256, dropout=0.1, class_weights=None,
                 component_weights=None, focal_gamma=0.0, backbone=None,
                 probs_out=False):
        super().__init__()
        self.input_dim = input_dim
        self.head_type = head_type
        self.probs_out = probs_out
        self.backbone = backbone
        self.component_names = COMPONENT_NAMES
        self.vocab_sizes = {c: len(CHORD_VOCAB[c]) for c in COMPONENT_NAMES}
        self.last_shared_features = None

        self.heads = nn.ModuleDict({
            component: self._build_head(head_type, input_dim, hidden_dim,
                                        self.vocab_sizes[component], dropout)
            for component in COMPONENT_NAMES
        })

        self.criterion = MultiTaskLoss(
            vocab_sizes=self.vocab_sizes,
            class_weights=class_weights,
            component_weights=component_weights,
            focal_gamma=focal_gamma,
        )

    @staticmethod
    def _build_head(head_type, input_dim, hidden_dim, vocab_size, dropout):
        if head_type == "mlp":
            return MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim,
                                 num_classes=vocab_size, dropout=dropout)
        if head_type == "linear":
            return LinearClassifier(input_dim=input_dim, num_classes=vocab_size,
                                    dropout=dropout)
        raise ValueError(f"Unknown head_type '{head_type}'. Use 'linear' or 'mlp'.")

    def _embed(self, x):
        """Return frame-level embeddings from waveform or pass through."""
        if self.backbone is not None and x.dim() == 2:
            padding_mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool,
                                       device=x.device)
            with torch.no_grad():
                return self.backbone.extract_features(x, padding_mask=padding_mask)[0]
        return x

    def forward(self, x, labels=None):
        """Forward pass.

        Args:
            x: Either pre-extracted embeddings ``(batch, n_patches, input_dim)``
               or, when a backbone is attached, raw waveforms ``(batch, samples)``.
            labels: Optional dict mapping component -> target indices
               ``(batch, n_patches)``.

        Returns:
            If ``probs_out``: the logits dict.
            Otherwise: ``(predictions, loss, None, component_losses)`` matching
            the decomposed model interface (the 3rd slot is attention weights,
            unused here).
        """
        embeddings = self._embed(x)
        self.last_shared_features = embeddings

        logits = {component: self.heads[component](embeddings)
                  for component in self.component_names}

        if self.probs_out:
            return logits

        predictions = self.get_predictions(logits)

        loss = None
        component_losses = None
        if labels is not None:
            loss, component_losses = self.criterion(logits, labels)

        return predictions, loss, None, component_losses

    def get_logits(self, x):
        embeddings = self._embed(x)
        return {component: self.heads[component](embeddings)
                for component in self.component_names}

    def get_predictions(self, logits):
        return {component: torch.argmax(logits[component], dim=-1)
                for component in self.component_names}

    def predict_probabilities(self, x):
        logits = self.get_logits(x)
        return {component: torch.softmax(logits[component], dim=-1)
                for component in self.component_names}
