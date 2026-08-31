"""Interaction Attention (Section III-E, Fig. 8).

Scaled dot-product attention is

.. math:: \\mathrm{Attention}(Q,K,V) = \\mathrm{Softmax}
          \\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V             \\qquad (5)

Interaction Attention assigns one hand to the query and the other hand to the
key and the value, in both directions

.. math::
    f^A_R = f_R \\circledast f_L = \\mathrm{Softmax}
    \\!\\left(\\frac{f_R f_L^\\top}{\\sqrt{d_f}}\\right) f_L            \\qquad (6)

.. math::
    f^A_L = f_L \\circledast f_R = \\mathrm{Softmax}
    \\!\\left(\\frac{f_L f_R^\\top}{\\sqrt{d_f}}\\right) f_R            \\qquad (7)

The two results are concatenated back into a two-hand feature map: the left hand
half carries ``f^A_L`` (the left hand fused with the right hand) and the right
hand half carries ``f^A_R``.

Unlike InterHand Temporal Fusion, the attention here is spatio-temporal: every
(time step, joint) pair is a token, so the score matrix has ``(T * V/2)`` rows
and columns. With the paper's window of ``T = 30`` and 21 joints per hand that
is a 630 x 630 matrix per head, which dominates the memory footprint of a
training step.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .reshape import (
    from_spatiotemporal_tokens,
    merge_hands,
    split_hands,
    to_spatiotemporal_tokens,
)


class InteractionAttention(nn.Module):
    """Bidirectional cross-hand spatio-temporal attention of Eq. (6) and Eq. (7).

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
        """Spatio-temporal attention. All arguments and the result are ``(N, C, T, J)``."""
        num_frames, num_joints = query.shape[2], query.shape[3]

        q = to_spatiotemporal_tokens(query, self.num_heads)  # (N, H, T*J, d)
        k = to_spatiotemporal_tokens(key, self.num_heads)
        v = to_spatiotemporal_tokens(value, self.num_heads)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (N, H, T*J, T*J)
        weights = torch.softmax(scores, dim=-1)
        fused = torch.matmul(weights, v)
        return from_spatiotemporal_tokens(fused, num_frames, num_joints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = split_hands(x)
        query_left, key_left, value_left = self.qkv_left(left).chunk(3, dim=1)
        query_right, key_right, value_right = self.qkv_right(right).chunk(3, dim=1)

        # Eq. (7): the left hand queries the right hand.
        fused_left = self._attend(query_left, key_right, value_right)
        # Eq. (6): the right hand queries the left hand.
        fused_right = self._attend(query_right, key_left, value_left)

        return merge_hands(fused_left, fused_right)
