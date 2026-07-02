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
from utils.decomposition_registry import get_decomposition

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


def set_beats_trainable(backbone, unfreeze_last_n, unfreeze_final_norm=True):
    """Unfreeze the last ``unfreeze_last_n`` BEATs encoder layers for fine-tuning.

    Freezes every backbone parameter first, then re-enables gradients on the top
    ``unfreeze_last_n`` transformer layers (and, optionally, the encoder's final
    LayerNorm). All other layers stay frozen, so the bulk of the ~90M-param
    backbone is preserved while only the highest-level representations adapt.

    Args:
        backbone: A loaded BEATs module.
        unfreeze_last_n: Number of top encoder layers to make trainable. ``0``
            keeps the whole backbone frozen.
        unfreeze_final_norm: Also unfreeze ``encoder.layer_norm`` when present.

    Returns:
        dict with ``n_total_layers``, ``unfrozen_layers`` (indices) and
        ``n_trainable_params`` (for logging).

    Raises:
        AttributeError: If the encoder's layer list can't be located, with an
            actionable message (the BEATs internals must expose
            ``encoder.layers``).
    """
    for param in backbone.parameters():
        param.requires_grad = False

    encoder = getattr(backbone, "encoder", None)
    layers = getattr(encoder, "layers", None) if encoder is not None else None
    if layers is None:
        raise AttributeError(
            "Could not locate 'backbone.encoder.layers'; cannot select BEATs "
            "layers to unfreeze. Inspect the loaded backbone's module tree and "
            "adjust set_beats_trainable() to match its attribute names."
        )

    n_total = len(layers)
    n = max(0, min(int(unfreeze_last_n), n_total))
    unfrozen_layers = []
    for i in range(n_total - n, n_total):
        for param in layers[i].parameters():
            param.requires_grad = True
        unfrozen_layers.append(i)

    if unfreeze_final_norm and encoder is not None:
        final_norm = getattr(encoder, "layer_norm", None)
        if final_norm is not None:
            for param in final_norm.parameters():
                param.requires_grad = True

    n_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    return {
        "n_total_layers": n_total,
        "unfrozen_layers": unfrozen_layers,
        "n_trainable_params": n_trainable,
    }


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


class FFNBlock(nn.Module):
    """Transformer-style position-wise feed-forward block.

    Applies the canonical FFN pattern with a pre-LayerNorm and a residual
    connection: ``x + Dropout(Linear(Dropout(GELU(Linear(LN(x))))))``. The inner
    Linear expands the width by ``expansion`` (paper/Transformer default 4) and
    the second projects back to ``dim``, so blocks preserve their input/output
    width and can be stacked with residuals.

    Args:
        dim: Feature width (kept constant across the block).
        expansion: Hidden-width multiplier (``d_ff = dim * expansion``).
        dropout: Dropout probability applied after the activation and after the
            output projection (two independent draws, standard FFN regularisation).
    """

    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm residual keeps the width == dim so blocks are stackable.
        return x + self.ff(self.norm(x))


class DeepSharedTrunk(nn.Module):
    """Stack of ``num_blocks`` shared :class:`FFNBlock`s feeding all heads.

    Placed between the (frozen) BEATs embeddings and the lightweight per-
    component heads, this deepens the trainable FFN capacity ONCE and shares it
    across every head, rather than duplicating a deep stack per head. Output is a
    LayerNorm'd tensor with the same width as the input, so downstream linear
    heads stay simple and existing shared-feature consumers keep the same shape.

    Args:
        dim: Feature width (matches the BEATs embedding dim, e.g. 768).
        num_blocks: Number of stacked FFN blocks (``>= 1``).
        expansion: Hidden-width multiplier passed to each :class:`FFNBlock`.
        dropout: Dropout probability inside each block.
    """

    def __init__(self, dim, num_blocks=2, expansion=4, dropout=0.1):
        super().__init__()
        self.blocks = nn.Sequential(*[
            FFNBlock(dim, expansion=expansion, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.out_norm(self.blocks(x))


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
        dropout: Dropout probability inside each head AND inside the shared trunk.
        head_layers: Number of shared :class:`FFNBlock`s inserted between the
            embeddings and the per-component heads. ``0`` (default) disables the
            trunk entirely (identity), reproducing the previous shallow-head
            behaviour; ``>= 1`` deepens the trainable FFN capacity once and
            shares it across all heads.
        ffn_expansion: Hidden-width multiplier inside each trunk FFN block
            (``d_ff = input_dim * ffn_expansion``; Transformer default 4). Only
            used when ``head_layers >= 1``.
        class_weights: Optional dict mapping component -> class-weight tensor.
        component_weights: Optional dict of per-component loss weights.
        focal_gamma: Focal-loss focusing parameter (0 = standard CE).
        backbone: Optional pre-loaded BEATs module. When provided,
            ``forward`` accepts raw waveforms ``(batch, samples)`` and extracts
            embeddings internally (for end-to-end inference / fine-tuning). When
            ``None``, ``forward`` expects pre-extracted embeddings ``(batch,
            n_patches, input_dim)``.
        backbone_trainable: If True, gradients flow through the backbone during
            ``forward`` (use with :func:`set_beats_trainable` to fine-tune the
            top layers). If False, the backbone runs under ``torch.no_grad()``
            (frozen feature extractor). Has no effect when ``backbone`` is None.
        probs_out: If True, ``forward`` returns the raw logits dict.
        decomposition: Chord-decomposition scheme. ``'paper6'`` (default) uses
            the ChordFormer paper's 6 heads (root_triad, bass, 7th, 9th, 11th,
            13th); ``'full9'`` uses the project's 9-head decomposition.
    """

    def __init__(self, input_dim=BEATS_EMBED_DIM, head_type="linear",
                 hidden_dim=256, dropout=0.1, class_weights=None,
                 component_weights=None, focal_gamma=0.0, backbone=None,
                 backbone_trainable=False, probs_out=False, decomposition="paper6",
                 head_layers=0, ffn_expansion=4):
        super().__init__()
        self.input_dim = input_dim
        self.head_type = head_type
        self.head_layers = int(head_layers)
        self.ffn_expansion = int(ffn_expansion)
        self.probs_out = probs_out
        self.backbone = backbone
        self.backbone_trainable = backbone_trainable
        decomp = get_decomposition(decomposition)
        self.decomposition = decomp.scheme
        self.component_names = list(decomp.COMPONENT_NAMES)
        self.vocab_sizes = {c: len(decomp.CHORD_VOCAB[c]) for c in self.component_names}
        self.last_shared_features = None

        # Optional deep shared FFN trunk between embeddings and heads. With
        # head_layers=0 it is an identity, so the model matches the previous
        # shallow-head behaviour exactly (and old checkpoints stay loadable).
        if self.head_layers > 0:
            self.trunk = DeepSharedTrunk(input_dim, num_blocks=self.head_layers,
                                         expansion=self.ffn_expansion, dropout=dropout)
        else:
            self.trunk = nn.Identity()

        self.heads = nn.ModuleDict({
            component: self._build_head(head_type, input_dim, hidden_dim,
                                        self.vocab_sizes[component], dropout)
            for component in self.component_names
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
        """Return frame-level embeddings from waveform or pass through.

        When a backbone is attached and the input is a raw waveform
        ``(batch, samples)``, run it through BEATs. If ``backbone_trainable`` is
        set, keep the autograd graph so gradients reach the unfrozen top layers;
        otherwise wrap extraction in ``torch.no_grad()`` (frozen extractor).
        """
        if self.backbone is not None and x.dim() == 2:
            padding_mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool,
                                       device=x.device)
            if self.backbone_trainable:
                return self.backbone.extract_features(x, padding_mask=padding_mask)[0]
            with torch.no_grad():
                return self.backbone.extract_features(x, padding_mask=padding_mask)[0]
        return x

    def _features(self, x):
        """Embed (if needed) then run the shared FFN trunk.

        Returns the per-patch shared features fed to every head. With an
        identity trunk (``head_layers=0``) this is just the embeddings.
        """
        return self.trunk(self._embed(x))

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
        features = self._features(x)
        self.last_shared_features = features

        logits = {component: self.heads[component](features)
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
        features = self._features(x)
        return {component: self.heads[component](features)
                for component in self.component_names}

    def get_predictions(self, logits):
        return {component: torch.argmax(logits[component], dim=-1)
                for component in self.component_names}

    def predict_probabilities(self, x):
        logits = self.get_logits(x)
        return {component: torch.softmax(logits[component], dim=-1)
                for component in self.component_names}
