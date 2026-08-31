"""Skeleton graphs for two-hand hand-washing recognition.

Two graphs are defined, following Section III-C and Fig. 3 of the paper:

* :class:`HandGraph` -- the physical structure of the two hands (the black nodes
  and edges in Fig. 3). This is the normalized adjacency matrix ``A`` of Eq. (1).
* :class:`InteractionGraph` -- the cross-hand edges that connect corresponding
  keypoints of the two hands (the red edges in Fig. 3). This is ``A_IG`` of
  Eq. (2).

Joint layout (Section III-B, footnote 1): each hand contributes 21 MediaPipe
Hands keypoints indexed from the palm, then the base of the thumb to its tip,
and ending with the pinky finger. Nodes ``0..20`` are the left hand and nodes
``21..41`` are the right hand, so ``V = 42``.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .utils import Edge, build_partitioned_adjacency, normalize_digraph, self_loop_edges

NUM_JOINTS_PER_HAND = 21
NUM_JOINTS = 2 * NUM_JOINTS_PER_HAND

# Bones of a single hand, expressed with 1-based indices to mirror the joint
# ordering documented in the paper. Converted to 0-based in `single_hand_edges`.
_SINGLE_HAND_BONES_1_BASED: List[Edge] = [
    (1, 2), (2, 3), (3, 4), (4, 5),           # thumb
    (1, 6), (6, 7), (7, 8), (8, 9),           # index finger
    (1, 10), (10, 11), (11, 12), (12, 13),    # middle finger
    (1, 14), (14, 15), (15, 16), (16, 17),    # ring finger
    (1, 18), (18, 19), (19, 20), (20, 21),    # pinky finger
]


def single_hand_edges(offset: int = 0) -> List[Edge]:
    """Bones of one hand as 0-based edges, shifted by ``offset``."""
    return [(i - 1 + offset, j - 1 + offset) for i, j in _SINGLE_HAND_BONES_1_BASED]


def two_hand_edges() -> List[Edge]:
    """Bones of both hands, without any cross-hand edge."""
    return single_hand_edges(0) + single_hand_edges(NUM_JOINTS_PER_HAND)


def cross_hand_edges() -> List[Edge]:
    """Interaction Graph edges ``E = {e_ik | i = 1..N/2, k = i + N/2}``."""
    return [(i, i + NUM_JOINTS_PER_HAND) for i in range(NUM_JOINTS_PER_HAND)]


class HandGraph:
    """Physical two-hand skeleton graph, i.e. ``A`` in Eq. (1) and Eq. (2).

    Args:
        max_hop: Size of the spatial neighbourhood. ``A`` is partitioned into
            ``max_hop + 1`` subsets, one per hop distance.

    Attributes:
        A: ``(max_hop + 1, V, V)`` normalized adjacency tensor.
    """

    def __init__(self, max_hop: int = 1) -> None:
        if max_hop < 0:
            raise ValueError(f"max_hop must be non-negative, got {max_hop}")
        self.num_node = NUM_JOINTS
        self.max_hop = max_hop
        self.edges = self_loop_edges(self.num_node) + two_hand_edges()
        self.A = build_partitioned_adjacency(self.num_node, self.edges, max_hop)

    def __repr__(self) -> str:
        return f"HandGraph(num_node={self.num_node}, max_hop={self.max_hop})"


class InteractionGraph:
    """Cross-hand Interaction Graph, i.e. ``A_IG`` in Eq. (2).

    The edge set only connects corresponding keypoints of the two hands, so every
    node has exactly one cross-hand neighbour. After column normalization the
    matrix is the permutation that swaps the two hands.

    Args:
        self_loops: When ``True``, self-loops are added before normalization,
            which reproduces the behaviour of the original research code. The
            paper defines ``A_IG`` with cross-hand edges only, which is the
            default.

    Attributes:
        A: ``(V, V)`` normalized adjacency matrix.
    """

    def __init__(self, self_loops: bool = False) -> None:
        self.num_node = NUM_JOINTS
        self.self_loops = self_loops
        self.edges = cross_hand_edges()
        if self_loops:
            self.edges = self_loop_edges(self.num_node) + self.edges

        support = np.zeros((self.num_node, self.num_node))
        for i, j in self.edges:
            support[i, j] = 1
            support[j, i] = 1
        self.A = normalize_digraph(support)

    def __repr__(self) -> str:
        return f"InteractionGraph(num_node={self.num_node}, self_loops={self.self_loops})"
