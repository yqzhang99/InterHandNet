"""Hand-washing skeleton dataset.

The dataset reads the ``.npz`` archives produced by ``tools/extract_skeletons.py``
and yields fixed-length windows in the ``(C, T, V)`` layout expected by
:class:`~interhandnet.models.interhandnet.InterHandNet`.

Archive layout
--------------
======================  ==============================  ==================================
Key                     Shape / dtype                   Meaning
======================  ==============================  ==================================
``skeletons``           ``(S, T, 42, 3)`` float32       one window per sample; may also be
                        or object array of ``(T, 42, 3)``  variable-length sequences
``labels``              ``(S,)`` int64                  WHO hand-washing step, 0-based
``cameras``             ``(S,)`` (optional)             camera setting, e.g. 100..105
``session_ids``         ``(S,)`` (optional)             recording the window came from
``start_frames``        ``(S,)`` int64 (optional)       window position in the global recording
                                                        order, used to sort by time
======================  ==============================  ==================================

Occluded joints are stored as exact zeros, which is what the extractor writes
when MediaPipe Hands does not return a hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ..graph.hand_graph import NUM_JOINTS
from .transforms import center_hands, resample_sequence, to_model_layout

PathLike = Union[str, Path]

# The six steps of Fig. 1, in the order of the WHO guidelines.
WHO_STEP_NAMES = (
    "1: palm to palm",
    "2: palm over dorsum with fingers",
    "3: palm to palm with fingers interlaced",
    "4: backs of fingers to opposing palms",
    "5: rotational rubbing of thumb in palm",
    "6: rotational rubbing with fingers in palm",
)

# Class names for the 7-way label set of the dataset used in the paper: the
# movement code is the class index, and code 0 collects every movement that is
# not one of the six steps (including closing the faucet with a paper towel,
# which the dataset authors fold into 0 for classification).
CLASS_NAMES = ("0: other movement",) + WHO_STEP_NAMES


def class_names(num_classes: int) -> Tuple[str, ...]:
    """Names for a ``num_classes``-way label set, falling back to plain indices.

    ``num_classes == 7`` is the dataset's own label set, where class 0 is "other";
    ``num_classes == 6`` means the six WHO steps alone.
    """
    if num_classes == len(CLASS_NAMES):
        return CLASS_NAMES
    if num_classes == len(WHO_STEP_NAMES):
        return WHO_STEP_NAMES
    return tuple(f"class {index}" for index in range(num_classes))


class HandWashingSkeletonDataset(Dataset):
    """Fixed-length two-hand skeleton windows with WHO step labels.

    Args:
        archive: Path to the ``.npz`` archive.
        window_size: Target number of frames per window (30 in the paper).
        indices: Optional subset of sample indices, used by cross-validation.
        center: Translate each hand to its palm keypoint. Off by default, since
            it removes the inter-hand distance used by the Interaction Graph.
    """

    def __init__(
        self,
        archive: PathLike,
        window_size: int = 30,
        indices: Optional[Sequence[int]] = None,
        center: bool = False,
    ) -> None:
        self.archive_path = Path(archive)
        if not self.archive_path.exists():
            raise FileNotFoundError(f"skeleton archive not found: {self.archive_path}")

        with np.load(self.archive_path, allow_pickle=True) as data:
            self.skeletons = data["skeletons"]
            self.labels = data["labels"].astype(np.int64)
            self.cameras = data["cameras"] if "cameras" in data else None
            self.session_ids = data["session_ids"] if "session_ids" in data else None
            self.start_frames = (
                data["start_frames"].astype(np.int64) if "start_frames" in data else None
            )

        if len(self.skeletons) != len(self.labels):
            raise ValueError(
                f"skeletons and labels disagree: {len(self.skeletons)} vs {len(self.labels)}"
            )

        self.window_size = window_size
        self.center = center
        self.indices = (
            np.arange(len(self.labels)) if indices is None else np.asarray(indices, dtype=np.int64)
        )
        self.num_classes = int(self.labels.max()) + 1 if len(self.labels) else 0

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_index = int(self.indices[index])
        sequence = np.asarray(self.skeletons[sample_index], dtype=np.float32)
        if sequence.ndim != 3 or sequence.shape[1] != NUM_JOINTS:
            raise ValueError(
                f"sample {sample_index} has shape {sequence.shape}, "
                f"expected (T, {NUM_JOINTS}, 3)"
            )

        sequence = resample_sequence(sequence, self.window_size)
        if self.center:
            sequence = center_hands(sequence)

        skeleton = torch.from_numpy(to_model_layout(sequence))
        label = torch.tensor(int(self.labels[sample_index]), dtype=torch.long)
        return skeleton, label

    def subset(self, indices: Sequence[int]) -> "HandWashingSkeletonDataset":
        """Return a view of this dataset restricted to ``indices``.

        Indices refer to positions in the archive, matching what
        :mod:`interhandnet.data.splits` produces.
        """
        clone = object.__new__(HandWashingSkeletonDataset)
        clone.__dict__.update(self.__dict__)
        clone.indices = np.asarray(indices, dtype=np.int64)
        return clone

    def class_counts(self) -> np.ndarray:
        """Number of samples per class in the current subset."""
        return np.bincount(self.labels[self.indices], minlength=max(self.num_classes, 1))
