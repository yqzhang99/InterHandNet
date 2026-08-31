"""Cross-validation splits that follow the evaluation protocol of Section IV-B.

The paper evaluates each of the six camera settings separately with 5-fold
cross-validation, and reports a weighted average over the settings:

    "We apply 5-fold cross-validation to each camera setting and obtain six
    groups of results. [...] To avoid data leakage, we divide the data based on
    temporal order, ensuring that the validation set is not immediately after
    the training set in the temporal sequence. A weighted average is used to
    calculate the final result, balancing the different dataset sizes across
    camera settings."

Windows that are adjacent in time are almost identical, so a random split would
leak validation content into training. Splitting on temporal order and dropping
a guard band around the validation block removes that leak.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

Fold = Tuple[np.ndarray, np.ndarray]


def temporal_kfold(
    order: Sequence[int],
    num_folds: int = 5,
    gap: int = 0,
) -> List[Fold]:
    """Split time-ordered sample indices into contiguous folds.

    Args:
        order: Sample indices sorted by recording time.
        num_folds: Number of folds (five in the paper).
        gap: Number of samples dropped from the training set on each side of the
            validation block, so that training data never sits immediately next
            to validation data.

    Returns:
        A list of ``(train_indices, validation_indices)`` pairs.
    """
    order = np.asarray(order, dtype=np.int64)
    num_samples = len(order)
    if num_folds < 2:
        raise ValueError(f"num_folds must be at least 2, got {num_folds}")
    if num_samples < num_folds:
        raise ValueError(f"cannot build {num_folds} folds from {num_samples} samples")
    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}")

    boundaries = np.linspace(0, num_samples, num_folds + 1).astype(int)
    folds: List[Fold] = []
    for fold in range(num_folds):
        start, stop = boundaries[fold], boundaries[fold + 1]
        validation = order[start:stop]
        train_mask = np.ones(num_samples, dtype=bool)
        train_mask[max(0, start - gap) : min(num_samples, stop + gap)] = False
        train = order[train_mask]
        if len(train) == 0:
            raise ValueError(
                f"fold {fold} has an empty training set; reduce gap ({gap}) or num_folds"
            )
        folds.append((train, validation))
    return folds


def group_indices_by_camera(
    cameras: Optional[Sequence],
    start_frames: Optional[Sequence] = None,
    num_samples: Optional[int] = None,
) -> Dict[object, np.ndarray]:
    """Group sample indices per camera setting, each group sorted by time.

    When ``cameras`` is ``None`` all samples form a single group, which is the
    right behaviour for datasets recorded with one camera. ``num_samples`` is
    only needed when neither ``cameras`` nor ``start_frames`` is available.
    """
    if cameras is not None:
        num_samples = len(cameras)
    elif start_frames is not None:
        num_samples = len(start_frames)
    elif num_samples is None:
        raise ValueError(
            "pass num_samples when the archive has neither cameras nor start_frames"
        )
    all_indices = np.arange(num_samples, dtype=np.int64)

    if cameras is None:
        groups = {None: all_indices}
    else:
        cameras = np.asarray(cameras)
        groups = {value: all_indices[cameras == value] for value in np.unique(cameras)}

    if start_frames is not None:
        start_frames = np.asarray(start_frames)
        groups = {
            key: indices[np.argsort(start_frames[indices], kind="stable")]
            for key, indices in groups.items()
        }
    return groups


def camera_wise_folds(
    cameras: Optional[Sequence],
    start_frames: Optional[Sequence] = None,
    num_folds: int = 5,
    gap: int = 0,
    num_samples: Optional[int] = None,
) -> Iterator[Tuple[object, int, np.ndarray, np.ndarray]]:
    """Yield ``(camera, fold_index, train_indices, validation_indices)``.

    This reproduces the six-camera by five-fold grid of Section IV-B.
    """
    groups = group_indices_by_camera(cameras, start_frames, num_samples)
    for camera, indices in groups.items():
        for fold_index, (train, validation) in enumerate(
            temporal_kfold(indices, num_folds=num_folds, gap=gap)
        ):
            yield camera, fold_index, train, validation


def weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
    """Weighted mean used to combine per-camera results."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return float((values * weights).sum() / total)
