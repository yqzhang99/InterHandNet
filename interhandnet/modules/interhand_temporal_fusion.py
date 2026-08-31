"""InterHand Temporal Fusion (Section III-D, Fig. 7).

A plain temporal convolution aggregates a single hand over the temporal kernel

.. math:: f^T_{l,t} = \\sum_{k=0}^{K_t-1} F_k f^S_{l,t+k}              \\qquad (3)

InterHand Temporal Fusion replaces its input with cross-hand attention over the
whole time window before the convolution is applied

.. math::
    f^T_{l,t} = \\sum_{k=0}^{K_t-1} F_k
    \\left[(f^{S,R}_{l,t+k} \\circledast f^{S,L}_l) \\,\\|\\,
           (f^{S,L}_{l,t+k} \\circledast f^{S,R}_l)\\right]            \\qquad (4)

This module computes the bracketed term; the surrounding ``F_k`` is the temporal
convolution of the STGC block.

Reading Eq. (4) together with the worked example in Section III-D fixes the role
of every operand: to reconstruct the left hand at an occluded time step
``tau``, the *right* hand feature at ``tau`` acts as the query and the *left*
hand features across the time window act as keys and values, so that
``f^{S,L}_{0,tau-1}`` and ``f^{S,L}_{0,tau+1}`` can fill in the missing
``f^{T,L}_{0,tau}``. Attention therefore runs along the temporal axis
independently for every joint.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .reshape import from_temporal_tokens, merge_hands, split_hands, to_temporal_tokens


class InterHandTemporalFusion(nn.Module):
    """Cross-hand temporal attention, i.e. the bracketed term of Eq. (4).

    Args:
        channels: Number of feature channels ``C``.
        num_heads: Number of attention heads. ``channels`` must be divisible by it.

    Shape:
        - Input: ``(N, C, T, V)``
        - Output: ``(N, C, T, V)``
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_left = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.qkv_right = nn.Conv2d(channels, 3 * channels, kernel_size=1)

    def _attend(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Per-joint attention along the temporal window.

        All arguments are ``(N, C, T, J)``; the result has the same shape.
        """
        q = to_temporal_tokens(query, self.num_heads)  # (N, H, J, T, d)
        k = to_temporal_tokens(key, self.num_heads)
        v = to_temporal_tokens(value, self.num_heads)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (N, H, J, T, T)
        weights = torch.softmax(scores, dim=-1)
        return from_temporal_tokens(torch.matmul(weights, v))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = split_hands(x)
        query_left, key_left, value_left = self.qkv_left(left).chunk(3, dim=1)
        query_right, key_right, value_right = self.qkv_right(right).chunk(3, dim=1)

        fused_left = self._attend(query_right, key_left, value_left)
        fused_right = self._attend(query_left, key_right, value_right)

        return merge_hands(fused_left, fused_right)
