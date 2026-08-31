"""Tests for the multi-head tensor layouts.

A bare ``view`` on a ``(N, C, T, J)`` tensor silently scrambles the data when the
target layout needs a transpose, which is the kind of bug these round-trip tests
are here to catch.
"""

import torch

from interhandnet.graph import NUM_JOINTS, NUM_JOINTS_PER_HAND
from interhandnet.modules.reshape import (
    from_spatiotemporal_tokens,
    from_temporal_tokens,
    merge_hands,
    split_hands,
    to_spatiotemporal_tokens,
    to_temporal_tokens,
)

BATCH, CHANNELS, FRAMES, HEADS = 2, 8, 5, 4


def test_split_and_merge_are_inverse():
    x = torch.randn(BATCH, CHANNELS, FRAMES, NUM_JOINTS)
    left, right = split_hands(x)
    assert left.shape == (BATCH, CHANNELS, FRAMES, NUM_JOINTS_PER_HAND)
    assert torch.equal(merge_hands(left, right), x)


def test_temporal_token_round_trip():
    x = torch.randn(BATCH, CHANNELS, FRAMES, NUM_JOINTS_PER_HAND)
    tokens = to_temporal_tokens(x, HEADS)
    assert tokens.shape == (BATCH, HEADS, NUM_JOINTS_PER_HAND, FRAMES, CHANNELS // HEADS)
    assert torch.allclose(from_temporal_tokens(tokens), x)


def test_temporal_tokens_preserve_channel_head_split():
    x = torch.randn(BATCH, CHANNELS, FRAMES, NUM_JOINTS_PER_HAND)
    tokens = to_temporal_tokens(x, HEADS)
    head, dim, joint, frame = 2, 1, 7, 3
    assert torch.equal(
        tokens[0, head, joint, frame, dim],
        x[0, head * (CHANNELS // HEADS) + dim, frame, joint],
    )


def test_spatiotemporal_token_round_trip():
    x = torch.randn(BATCH, CHANNELS, FRAMES, NUM_JOINTS_PER_HAND)
    tokens = to_spatiotemporal_tokens(x, HEADS)
    assert tokens.shape == (
        BATCH,
        HEADS,
        FRAMES * NUM_JOINTS_PER_HAND,
        CHANNELS // HEADS,
    )
    restored = from_spatiotemporal_tokens(tokens, FRAMES, NUM_JOINTS_PER_HAND)
    assert torch.allclose(restored, x)


def test_spatiotemporal_tokens_are_ordered_time_major():
    x = torch.randn(BATCH, CHANNELS, FRAMES, NUM_JOINTS_PER_HAND)
    tokens = to_spatiotemporal_tokens(x, HEADS)
    frame, joint, head, dim = 3, 11, 1, 0
    token_index = frame * NUM_JOINTS_PER_HAND + joint
    assert torch.equal(
        tokens[0, head, token_index, dim],
        x[0, head * (CHANNELS // HEADS) + dim, frame, joint],
    )
