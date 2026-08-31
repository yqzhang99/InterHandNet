"""Tensor layout helpers for the two-hand attention modules.

All feature maps in this repository use the ``(N, C, T, V)`` layout, where ``N``
is the batch size, ``C`` the number of channels, ``T`` the temporal window and
``V = 42`` the number of joints. Joints ``0..20`` belong to the left hand and
joints ``21..41`` to the right hand.

The attention modules need three different multi-head views of a single-hand
feature map ``(N, C, T, J)``:

* per-joint sequences over time -- used by InterHand Temporal Fusion, whose
  attention runs along the temporal window for every joint independently;
* per-frame joint sets -- used by Interaction Attention in its default
  ``spatial`` scope, where each frame is attended independently;
* flattened spatio-temporal tokens -- used by Interaction Attention in its
  ``spatiotemporal`` scope, where a token is one (time step, joint) pair.

Writing these permutations once keeps the modules readable and avoids the
silent data scrambling that a bare ``view`` would cause.
"""

from __future__ import annotations

from typing import Tuple

import torch

from ..graph.hand_graph import NUM_JOINTS_PER_HAND


def split_hands(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split ``(N, C, T, V)`` into left and right hand feature maps."""
    if x.size(-1) != 2 * NUM_JOINTS_PER_HAND:
        raise ValueError(
            f"expected {2 * NUM_JOINTS_PER_HAND} joints in the last dimension, got {x.size(-1)}"
        )
    return x[..., :NUM_JOINTS_PER_HAND], x[..., NUM_JOINTS_PER_HAND:]


def merge_hands(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Concatenate left and right hand feature maps back into ``(N, C, T, V)``."""
    return torch.cat([left, right], dim=-1)


def to_temporal_tokens(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """``(N, C, T, J)`` -> ``(N, heads, J, T, C // heads)``.

    Every joint becomes an independent sequence of ``T`` tokens.
    """
    n, c, t, j = x.shape
    head_dim = c // num_heads
    return x.reshape(n, num_heads, head_dim, t, j).permute(0, 1, 4, 3, 2)


def from_temporal_tokens(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`to_temporal_tokens`."""
    n, heads, j, t, head_dim = x.shape
    return x.permute(0, 1, 4, 3, 2).reshape(n, heads * head_dim, t, j)


def to_spatial_tokens(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """``(N, C, T, J)`` -> ``(N, heads, T, J, C // heads)``.

    Every frame becomes an independent set of ``J`` tokens, so an attention
    matrix built from this view is block-diagonal in time.
    """
    n, c, t, j = x.shape
    head_dim = c // num_heads
    return x.reshape(n, num_heads, head_dim, t, j).permute(0, 1, 3, 4, 2)


def from_spatial_tokens(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`to_spatial_tokens`."""
    n, heads, t, j, head_dim = x.shape
    return x.permute(0, 1, 4, 2, 3).reshape(n, heads * head_dim, t, j)


def to_spatiotemporal_tokens(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """``(N, C, T, J)`` -> ``(N, heads, T * J, C // heads)``.

    Every (time step, joint) pair becomes one token.
    """
    n, c, t, j = x.shape
    head_dim = c // num_heads
    return (
        x.reshape(n, num_heads, head_dim, t, j)
        .permute(0, 1, 3, 4, 2)
        .reshape(n, num_heads, t * j, head_dim)
    )


def from_spatiotemporal_tokens(x: torch.Tensor, num_frames: int, num_joints: int) -> torch.Tensor:
    """Inverse of :func:`to_spatiotemporal_tokens`."""
    n, heads, _, head_dim = x.shape
    return (
        x.reshape(n, heads, num_frames, num_joints, head_dim)
        .permute(0, 1, 4, 2, 3)
        .reshape(n, heads * head_dim, num_frames, num_joints)
    )
