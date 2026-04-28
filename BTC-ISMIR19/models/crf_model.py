from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class CRF(nn.Module):
    """
    Implements Conditional Random Fields that can be trained via
    backpropagation.

    The forward algorithm (partition function) and Viterbi decoder both
    process the tag dimension in chunks of ``chunk_size`` to avoid
    materialising the full ``(B, N, N)`` transition tensor at each step.
    This makes the ``full`` CRF mode (N ≈ 2000) feasible without OOM.

    Args:
        num_tags: Number of CRF tags.
        chunk_size: Number of destination tags processed per chunk in the
            forward and Viterbi passes.  Smaller values use less peak
            memory at the cost of slightly more kernel launches.
            Defaults to 256.
    """

    def __init__(self, num_tags, chunk_size=256):
        super(CRF, self).__init__()

        self.num_tags = num_tags
        self.chunk_size = chunk_size
        self.transitions = nn.Parameter(torch.Tensor(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.stop_transitions = nn.Parameter(torch.randn(num_tags))

        nn.init.xavier_normal_(self.transitions)

    def forward(self, feats):
        # Shape checks
        if len(feats.shape) != 3:
            raise ValueError("feats must be 3-d got {}-d".format(feats.shape))

        return self._viterbi(feats)

    def loss(self, feats, tags):
        """
        Computes negative log likelihood between features and tags.
        Essentially difference between individual sequence scores and
        sum of all possible sequence scores (partition function)
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
            tags: Target tag indices [batch size, sequence length]. Should be between
                    0 and num_tags
        Returns:
            Negative log likelihood [a scalar]
        """
        # Shape checks
        if len(feats.shape) != 3:
            raise ValueError("feats must be 3-d got {}-d".format(feats.shape))

        if len(tags.shape) != 2:
            raise ValueError('tags must be 2-d but got {}-d'.format(tags.shape))

        if feats.shape[:2] != tags.shape:
            raise ValueError('First two dimensions of feats and tags must match')

        sequence_score = self._sequence_score(feats, tags)
        partition_function = self._partition_function(feats)
        log_probability = sequence_score - partition_function

        # -ve of l()
        # Average across batch
        return -log_probability.mean()

    def _sequence_score(self, feats, tags):
        """
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
            tags: Target tag indices [batch size, sequence length]. Should be between
                    0 and num_tags
        Returns: Sequence score of shape [batch size]
        """

        # Compute feature scores
        feat_score = feats.gather(2, tags.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

        # Compute transition scores
        # Unfold to get [from, to] tag index pairs
        tags_pairs = tags.unfold(1, 2, 1)

        # Use advanced indexing to pull out required transition scores
        indices = tags_pairs.permute(2, 0, 1).chunk(2)
        trans_score = self.transitions[indices].squeeze(0).sum(dim=-1)

        # Compute start and stop scores
        start_score = self.start_transitions[tags[:, 0]]
        stop_score = self.stop_transitions[tags[:, -1]]

        return feat_score + start_score + trans_score + stop_score

    def _partition_function(self, feats):
        """
        Computes the partition function for the CRF using the forward algorithm.

        The transition step is computed in chunks of ``self.chunk_size``
        destination tags to avoid allocating the full ``(B, N, N)`` tensor,
        keeping peak memory at ``O(B × N × chunk_size)`` instead of
        ``O(B × N²)``.

        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns:
            Total scores of shape [batch size]
        """
        _, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))

        a = feats[:, 0] + self.start_transitions.unsqueeze(0)  # (B, N)
        transitions = self.transitions.unsqueeze(0)             # (1, N, N)

        for i in range(1, seq_size):
            feat = feats[:, i]          # (B, N)
            new_a = torch.empty_like(a)
            for j in range(0, num_tags, self.chunk_size):
                end = min(j + self.chunk_size, num_tags)
                # (B, N, 1) + (1, N, chunk) → (B, N, chunk); logsumexp over "from" dim
                chunk = a.unsqueeze(-1) + transitions[:, :, j:end]
                new_a[:, j:end] = torch.logsumexp(chunk, dim=1) + feat[:, j:end]
            a = new_a

        return torch.logsumexp(a + self.stop_transitions.unsqueeze(0), dim=1)  # (B,)

    def _viterbi(self, feats):
        """
        Uses Viterbi algorithm to predict the best sequence.

        Like the partition function, the max-over-sources step is chunked
        to avoid the ``(B, N, N)`` peak allocation.  The backtracking
        indices are stored in a pre-allocated ``(T-1, B, N)`` tensor
        instead of a growing Python list.

        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns: Best tag sequence [batch size, sequence length]
        """
        batch_size, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))

        v = feats[:, 0] + self.start_transitions.unsqueeze(0)  # (B, N)
        transitions = self.transitions.unsqueeze(0)             # (1, N, N)

        # Pre-allocate backtracking table instead of growing a Python list.
        paths = torch.empty(seq_size - 1, batch_size, num_tags,
                            dtype=torch.long, device=feats.device)

        for i in range(1, seq_size):
            new_v = torch.empty_like(v)
            for j in range(0, num_tags, self.chunk_size):
                end = min(j + self.chunk_size, num_tags)
                # (B, N, 1) + (1, N, chunk) → (B, N, chunk); max over "from" dim
                chunk = v.unsqueeze(-1) + transitions[:, :, j:end]
                new_v[:, j:end], paths[i - 1, :, j:end] = chunk.max(1)
            v = new_v + feats[:, i]

        v, tag = (v + self.stop_transitions.unsqueeze(0)).max(1, keepdim=True)  # (B, 1)

        # Backtrack through the pre-allocated paths tensor
        tags = [tag]
        for i in range(seq_size - 2, -1, -1):
            tag = paths[i].gather(1, tag)
            tags.append(tag)

        tags.reverse()
        return torch.cat(tags, dim=1)  # (B, T)
