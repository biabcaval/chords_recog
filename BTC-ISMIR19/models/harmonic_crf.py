"""
HarmonicCRF: Conditional Random Field for chord sequence decoding.

This module implements a CRF that operates on the joint root x triad space
(13 roots x 7 triads = 91 tags) to enforce temporal coherence in chord
predictions from the ChordMax decomposed pipeline.

It is designed to work as a post-processing stage on top of the frozen
ChordFormer model: the 9-head logits are combined into an observation
potential, and the CRF's learnable transition matrix captures which
harmonic progressions are plausible.

This is separate from the legacy BTC CRF (models/crf_model.py) which
operates on the monolithic 170-class vocabulary.

Architecture:
    ChordFormer (frozen) -> 9 head logits
        -> HarmonicCRF:
            1. Compute observation potential from root + triad log-probs
            2. Viterbi decoding with learned transitions
            3. Split joint tags back into root + triad
            4. Extensions (bass, misc, 6th, 7th, 9th, 11th, 13th) via argmax

Extensibility (future improvements):
    - Add other heads (bass, 7th) to the observation potential
    - Expand to root x triad x 7th (91 x 4 = 364 tags)
    - Use fixed penalty transitions instead of learned (for comparison)
    - Estimate transitions from GT statistics (no training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.crf_model import CRF
from utils.chord_decomposition import COMPONENT_NAMES


# Components whose predictions come from the CRF (joint decoding)
CRF_COMPONENTS = ['root', 'triad']

# Components resolved independently via argmax (not part of CRF)
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
