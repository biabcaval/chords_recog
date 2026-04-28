"""
CRF-based sequence decoders for the ChordMax decomposed pipeline.

Two modes are available:

1. **root_triad** (``HarmonicCRF``): Joint CRF over root x triad
   (13 x 7 = 91 tags).  Extensions resolved via argmax.

2. **full** (``FullChordCRF``): CRF over the full observed chord
   vocabulary (~2000 tags built from training data).  All 9 components
   are captured in the transition matrix.

Both are designed as post-processing stages on top of a frozen
ChordFormer: the 9-head logits are combined into an observation
potential, and a learnable transition matrix captures plausible
harmonic progressions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from models.crf_model import CRF
from utils.chord_decomposition import COMPONENT_NAMES, CHORD_VOCAB

CRF_MODE_CHOICES = ['root_triad', 'full']

# Components whose predictions come from the CRF (joint decoding) - root_triad mode
CRF_COMPONENTS = ['root', 'triad']

# Components resolved independently via argmax (root_triad mode only)
ARGMAX_COMPONENTS = [c for c in COMPONENT_NAMES if c not in CRF_COMPONENTS]


class HarmonicCRF(nn.Module):
    """
    CRF-based sequence decoder for the ChordMax decomposed pipeline.

    Combines the root and triad head logits into a joint observation
    potential over 91 tags (13 roots x 7 triads), then uses a learnable
    transition matrix + Viterbi to decode temporally coherent sequences.

    The remaining 7 heads (bass, misc, 6th, 7th, 9th, 11th, 13th) are
    resolved independently via argmax, since their temporal coherence
    depends primarily on root/triad identity rather than on their own
    sequential patterns.

    Args:
        n_roots: Number of root classes (default: 13, matching CHORD_VOCAB['root'])
        n_triads: Number of triad classes (default: 7, matching CHORD_VOCAB['triad'])
    """

    def __init__(self, n_roots: int = 13, n_triads: int = 7):
        super().__init__()
        self.n_roots = n_roots
        self.n_triads = n_triads
        self.n_joint_tags = n_roots * n_triads

        self.crf = CRF(num_tags=self.n_joint_tags)

    def encode_joint_tag(self, root_idx: torch.Tensor, triad_idx: torch.Tensor) -> torch.Tensor:
        """Encode separate root and triad indices into a single joint tag.

        joint_tag = root * n_triads + triad

        Args:
            root_idx: (B, T) root class indices
            triad_idx: (B, T) triad class indices

        Returns:
            (B, T) joint tag indices in [0, n_joint_tags)
        """
        return root_idx * self.n_triads + triad_idx

    def decode_joint_tag(self, joint_tag: torch.Tensor):
        """Decode a joint tag back into root and triad indices.

        Args:
            joint_tag: (B, T) joint tag indices

        Returns:
            root_idx: (B, T) root class indices
            triad_idx: (B, T) triad class indices
        """
        root_idx = joint_tag // self.n_triads
        triad_idx = joint_tag % self.n_triads
        return root_idx, triad_idx

    def compute_observation_potential(self, logits: dict) -> torch.Tensor:
        """Compute the observation potential for the CRF from head logits.

        For each frame t, the score of joint tag (r, q) is:
            phi(r, q, t) = log P(root=r | t) + log P(triad=q | t)

        This follows the decomposed observation model where the score
        of a chord is the sum of log-probabilities from independent heads.

        Future: To incorporate additional heads (e.g. bass, 7th), add their
        log-prob contributions here. For example:
            phi(r, q, t) += log P(7th=best_7th | t)
        This would require either marginalizing or taking the max over
        the extension values for each (r, q) combination.

        Args:
            logits: Dict mapping component names to logit tensors.
                    Must contain 'root' (B, T, 13) and 'triad' (B, T, 7).

        Returns:
            observation_potential: (B, T, n_joint_tags) scores for the CRF
        """
        root_logits = logits['root']    # (B, T, 13)
        triad_logits = logits['triad']  # (B, T, 7)

        log_p_root = F.log_softmax(root_logits, dim=-1)    # (B, T, 13)
        log_p_triad = F.log_softmax(triad_logits, dim=-1)  # (B, T, 7)

        # Outer sum: for each frame, combine every root with every triad.
        # joint[b, t, r, q] = log P(root=r) + log P(triad=q)
        joint = log_p_root.unsqueeze(-1) + log_p_triad.unsqueeze(-2)  # (B, T, 13, 7)

        B, T = root_logits.shape[:2]
        observation_potential = joint.reshape(B, T, self.n_joint_tags)  # (B, T, 91)

        return observation_potential

    def forward(self, logits: dict) -> dict:
        """Decode the best chord sequence using Viterbi.

        Combines root + triad into joint observation potential, runs
        Viterbi to find the globally optimal tag sequence, then splits
        back into root and triad. Other components use argmax.

        Args:
            logits: Dict mapping component names to logit tensors (B, T, C)

        Returns:
            predictions: Dict mapping component names to predicted indices (B, T)
        """
        obs = self.compute_observation_potential(logits)  # (B, T, 91)

        # Viterbi decoding for root x triad
        joint_tags = self.crf(obs)  # (B, T)

        root_pred, triad_pred = self.decode_joint_tag(joint_tags)

        predictions = {
            'root': root_pred,
            'triad': triad_pred,
        }

        # Remaining components: independent argmax (not part of CRF)
        for comp in ARGMAX_COMPONENTS:
            if comp in logits:
                predictions[comp] = torch.argmax(logits[comp], dim=-1)

        return predictions

    def loss(self, logits: dict, labels: dict) -> torch.Tensor:
        """Compute the CRF negative log-likelihood loss.

        Args:
            logits: Dict mapping component names to logit tensors (B, T, C)
            labels: Dict mapping component names to target indices (B, T)

        Returns:
            Scalar NLL loss
        """
        obs = self.compute_observation_potential(logits)  # (B, T, 91)

        # Combine GT root and triad labels into joint tags
        root_labels = labels['root']    # (B, T)
        triad_labels = labels['triad']  # (B, T)
        joint_tags = self.encode_joint_tag(root_labels, triad_labels)  # (B, T)

        return self.crf.loss(obs, joint_tags)


class FullChordCRF(nn.Module):
    """CRF over the full observed chord vocabulary.

    Instead of only root x triad (91 tags), this CRF operates on every
    unique chord that appears in the training data (~2000 tags).  The
    observation potential for each tag is the sum of log-probabilities
    from all 9 decomposed heads, indexed by a pre-computed component
    matrix.

    Args:
        chord_vocab: Ordered list of chord label strings.
        component_matrix: ``(N_vocab, 9)`` int64 tensor.  Row *i* holds
            the 9 component-class indices for ``chord_vocab[i]``.
        chord_to_idx: Mapping from label string to vocab index.
    """

    def __init__(
        self,
        chord_vocab: List[str],
        component_matrix: torch.Tensor,
        chord_to_idx: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.chord_vocab = list(chord_vocab)
        self.n_tags = len(chord_vocab)
        self.chord_to_idx = chord_to_idx or {l: i for i, l in enumerate(chord_vocab)}

        self.register_buffer('component_matrix', component_matrix.long())
        self.crf = CRF(num_tags=self.n_tags)

        lookup, strides = self._build_lookup_tensor()
        self.register_buffer('lookup_tensor', lookup)
        self.register_buffer('lookup_strides', strides)

    def _build_lookup_tensor(self):
        """Build a vectorized mixed-radix lookup: component tuple → vocab index.

        Encodes each 9-component combination as a single integer using
        mixed-radix strides, then builds a flat lookup tensor. Replaces the
        O(B×T) Python dict loop in ``labels_to_tags`` with a single
        vectorized ``index_select`` operation.

        Returns:
            lookup: ``(total_combinations,)`` long tensor.
            strides: ``(9,)`` long tensor of per-component strides.
        """
        sizes = [len(CHORD_VOCAB[c]) for c in COMPONENT_NAMES]
        strides = torch.ones(len(sizes), dtype=torch.long)
        for i in range(len(sizes) - 2, -1, -1):
            strides[i] = strides[i + 1] * sizes[i + 1]

        fallback = self.chord_to_idx.get('N', 0)
        total = int(strides[0].item() * sizes[0])
        lookup = torch.full((total,), fallback, dtype=torch.long)

        key_ints = (self.component_matrix * strides).sum(-1)  # (N_vocab,)
        for i in range(self.n_tags):
            lookup[key_ints[i].item()] = i

        return lookup, strides

    def labels_to_tags(self, labels: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert per-frame 9-component index labels to vocab tag indices.

        Fully vectorized via a pre-built mixed-radix lookup tensor — no
        Python loops over batch or time dimensions.

        Args:
            labels: Dict mapping component names to ``(B, T)`` index tensors.

        Returns:
            ``(B, T)`` tensor of vocab tag indices.
        """
        stacked = torch.stack([labels[c] for c in COMPONENT_NAMES], dim=-1)  # (B, T, 9)
        key_ints = (stacked * self.lookup_strides.to(stacked.device)).sum(-1)  # (B, T)
        return self.lookup_tensor.to(stacked.device)[key_ints]  # (B, T)

    def compute_observation_potential(self, logits: dict) -> torch.Tensor:
        """Sum log-probs from all 9 heads for every vocab entry.

        Args:
            logits: Dict mapping component names to ``(B, T, C_i)`` tensors.

        Returns:
            ``(B, T, N_vocab)`` observation scores.
        """
        first_key = COMPONENT_NAMES[0]
        B, T = logits[first_key].shape[:2]
        device = logits[first_key].device

        obs = torch.zeros(B, T, self.n_tags, device=device)

        for j, comp in enumerate(COMPONENT_NAMES):
            log_p = F.log_softmax(logits[comp], dim=-1)           # (B, T, C_j)
            indices = self.component_matrix[:, j].to(device)       # (N_vocab,)
            obs = obs + log_p[:, :, indices]                       # (B, T, N_vocab)

        return obs

    def forward(self, logits: dict) -> dict:
        """Viterbi decode the best chord sequence.

        Returns:
            Dict mapping each component name to ``(B, T)`` predicted indices.
        """
        obs = self.compute_observation_potential(logits)
        joint_tags = self.crf(obs)  # (B, T)

        predictions = {}
        for j, comp in enumerate(COMPONENT_NAMES):
            comp_col = self.component_matrix[:, j].to(joint_tags.device)
            predictions[comp] = comp_col[joint_tags]  # (B, T)

        return predictions

    def loss(self, logits: dict, labels: dict) -> torch.Tensor:
        """CRF negative log-likelihood.

        Args:
            logits: Dict of ``(B, T, C_i)`` logit tensors.
            labels: Dict of ``(B, T)`` ground-truth component indices.

        Returns:
            Scalar NLL loss.
        """
        obs = self.compute_observation_potential(logits)
        tags = self.labels_to_tags(labels)
        return self.crf.loss(obs, tags)
