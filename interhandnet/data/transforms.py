"""Skeleton sequence transforms.

The evaluation protocol of Section IV-A resamples every sequence to a fixed
window of 30 frames along the temporal dimension, so that a 30 fps camera
yields one hand-washing prediction per second.
"""

from __future__ import annotations

import numpy as np

from ..graph.hand_graph import NUM_JOINTS, NUM_JOINTS_PER_HAND


def resample_sequence(sequence: np.ndarray, num_frames: int) -> np.ndarray:
    """Resample a skeleton sequence along time by linear interpolation.

    Args:
        sequence: ``(T, V, C)`` array.
        num_frames: Target window length.

    Returns:
        ``(num_frames, V, C)`` array.
    """
    if sequence.ndim != 3:
        raise ValueError(f"expected a (T, V, C) sequence, got shape {sequence.shape}")
    source_length = sequence.shape[0]
    if source_length == 0:
        raise ValueError("cannot resample an empty sequence")
    if source_length == num_frames:
        return sequence.astype(np.float32, copy=False)

    source_positions = np.linspace(0.0, 1.0, source_length)
    target_positions = np.linspace(0.0, 1.0, num_frames)
    flat = sequence.reshape(source_length, -1)
    resampled = np.empty((num_frames, flat.shape[1]), dtype=np.float32)
    for column in range(flat.shape[1]):
        resampled[:, column] = np.interp(target_positions, source_positions, flat[:, column])
    return resampled.reshape(num_frames, *sequence.shape[1:])


def to_model_layout(sequence: np.ndarray) -> np.ndarray:
    """Convert ``(T, V, C)`` into the model layout ``(C, T, V)``."""
    return np.ascontiguousarray(sequence.transpose(2, 0, 1), dtype=np.float32)


def missing_joint_mask(sequence: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
    """Boolean ``(T, V)`` mask marking joints the extractor did not detect.

    MediaPipe Hands reports nothing for an occluded hand, which the extraction
    script writes out as exact zeros. Those entries are the missing data that
    InterHand Temporal Fusion is designed to reconstruct.
    """
    return np.all(np.abs(sequence) <= tolerance, axis=-1)


def center_hands(sequence: np.ndarray) -> np.ndarray:
    """Translate each hand so that its palm keypoint sits at the origin.

    This removes the absolute position of the hands in camera space while
    keeping their shape. It also removes the inter-hand distance that the
    Interaction Graph relies on, so it is off by default and only useful for
    ablations that isolate pose from position.
    """
    if sequence.shape[1] != NUM_JOINTS:
        raise ValueError(f"expected {NUM_JOINTS} joints, got {sequence.shape[1]}")
    centered = sequence.astype(np.float32, copy=True)
    for offset in (0, NUM_JOINTS_PER_HAND):
        hand = centered[:, offset : offset + NUM_JOINTS_PER_HAND]
        detected = ~np.all(hand == 0, axis=(1, 2))
        palm = hand[:, :1]
        hand[detected] -= palm[detected]
    return centered
