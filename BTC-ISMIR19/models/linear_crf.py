"""Linear-chain CRF with a fixed, non-trainable transition matrix.

This implements the "linear CRF, lambda = 30" decoding stage described in the
ChordFormer paper (Tabela 5):

    T[i, j] = lambda  if i == j
              0       otherwise

so the CRF only rewards staying in the same tag (smoothing), without learning
any harmonic prior. ``start_transitions`` and ``stop_transitions`` are zero
buffers (no preference for any start/end tag).

The class exposes the same ``forward`` / ``loss`` API as
``models.crf_model.CRF`` so it can be dropped in wherever the trainable CRF is
used. The internal forward / Viterbi implementations are simplified versions
of the chunked ones in ``crf_model.py``; with a diagonal transition matrix
the chunked trick is unnecessary, but we keep the algorithm shape identical
for predictable behaviour.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class LinearCRF(nn.Module):
    """Linear-chain CRF with diagonal, fixed transitions (``lambda * I``).

    Args:
        num_tags: Number of CRF tags.
        lam: Diagonal value of the transition matrix (self-transition bonus).
            Defaults to ``30.0`` to match the ChordFormer paper.
        chunk_size: Kept for API parity with ``CRF``; unused here because the
            diagonal matrix never needs chunking.
    """

    def __init__(self, num_tags, lam=30.0, chunk_size=256):
        super(LinearCRF, self).__init__()

        self.num_tags = int(num_tags)
        self.lam = float(lam)
        self.chunk_size = int(chunk_size)

        transitions = torch.eye(self.num_tags) * self.lam
        start_transitions = torch.zeros(self.num_tags)
        stop_transitions = torch.zeros(self.num_tags)

        # Buffers (move with .to(device) but are NOT trainable parameters).
        self.register_buffer('transitions', transitions)
        self.register_buffer('start_transitions', start_transitions)
        self.register_buffer('stop_transitions', stop_transitions)

    # ------------------------------------------------------------------
    # Public API (mirrors models.crf_model.CRF)
    # ------------------------------------------------------------------
    def forward(self, feats):
        if feats.dim() != 3:
            raise ValueError("feats must be 3-d got {}-d".format(feats.shape))
        return self._viterbi(feats)

    def loss(self, feats, tags):
        """Negative log-likelihood under the fixed-transition CRF.

        Even though the transitions are not trainable, this still produces a
        meaningful loss for the emission scores ``feats`` (the model can be
        trained end-to-end so that emissions are calibrated to the fixed
        smoothing prior).
        """
        if feats.dim() != 3:
            raise ValueError("feats must be 3-d got {}-d".format(feats.shape))
        if tags.dim() != 2:
            raise ValueError("tags must be 2-d but got {}-d".format(tags.shape))
        if feats.shape[:2] != tags.shape:
            raise ValueError('First two dimensions of feats and tags must match')

        sequence_score = self._sequence_score(feats, tags)
        partition_function = self._partition_function(feats)
        log_probability = sequence_score - partition_function
        return -log_probability.mean()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _sequence_score(self, feats, tags):
        feat_score = feats.gather(2, tags.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

        # Transition contribution: only "stay" transitions earn ``lam``; any
        # change of tag contributes 0. So the transition score is simply
        # ``lam * #consecutive equal pairs``.
        equal_pairs = (tags[:, 1:] == tags[:, :-1]).to(feats.dtype)
        trans_score = equal_pairs.sum(dim=-1) * self.lam

        start_score = self.start_transitions[tags[:, 0]]
        stop_score = self.stop_transitions[tags[:, -1]]
        return feat_score + start_score + trans_score + stop_score

    def _partition_function(self, feats):
        _, seq_size, num_tags = feats.shape
        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))

        a = feats[:, 0] + self.start_transitions.unsqueeze(0)  # (B, N)
        transitions = self.transitions.unsqueeze(0)            # (1, N, N)

        for i in range(1, seq_size):
            feat = feats[:, i]
            # (B, N, 1) + (1, N, N) -> (B, N, N); logsumexp over "from" dim.
            chunk = a.unsqueeze(-1) + transitions
            a = torch.logsumexp(chunk, dim=1) + feat

        return torch.logsumexp(a + self.stop_transitions.unsqueeze(0), dim=1)

    def _viterbi(self, feats):
        batch_size, seq_size, num_tags = feats.shape
        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))

        v = feats[:, 0] + self.start_transitions.unsqueeze(0)
        transitions = self.transitions.unsqueeze(0)

        paths = torch.empty(seq_size - 1, batch_size, num_tags,
                            dtype=torch.long, device=feats.device)

        for i in range(1, seq_size):
            chunk = v.unsqueeze(-1) + transitions  # (B, N, N)
            v_max, argmax = chunk.max(1)           # max over "from" dim
            paths[i - 1] = argmax
            v = v_max + feats[:, i]

        v, tag = (v + self.stop_transitions.unsqueeze(0)).max(1, keepdim=True)

        tags = [tag]
        for i in range(seq_size - 2, -1, -1):
            tag = paths[i].gather(1, tag)
            tags.append(tag)

        tags.reverse()
        return torch.cat(tags, dim=1)
