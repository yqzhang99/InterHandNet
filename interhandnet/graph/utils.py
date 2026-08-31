"""Graph utilities shared by the hand graph and the Interaction Graph."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

Edge = Tuple[int, int]


def get_hop_distance(num_node: int, edges: Sequence[Edge], max_hop: int) -> np.ndarray:
    """Return the pairwise hop distance of an undirected graph.

    Unreachable pairs (and pairs further away than ``max_hop``) are ``inf``.
    """
    adjacency = np.zeros((num_node, num_node))
    for i, j in edges:
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    hop_distance = np.full((num_node, num_node), np.inf)
    transfer_mat = [np.linalg.matrix_power(adjacency, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    # Assign in decreasing order so that the shortest hop wins.
    for d in range(max_hop, -1, -1):
        hop_distance[arrive_mat[d]] = d
    return hop_distance


def normalize_digraph(adjacency: np.ndarray) -> np.ndarray:
    """Column-normalize an adjacency matrix (``A D^-1``), as in ST-GCN."""
    degree = adjacency.sum(axis=0)
    num_node = adjacency.shape[0]
    degree_inv = np.zeros((num_node, num_node))
    for i in range(num_node):
        if degree[i] > 0:
            degree_inv[i, i] = degree[i] ** -1
    return adjacency @ degree_inv


def build_partitioned_adjacency(
    num_node: int, edges: Sequence[Edge], max_hop: int
) -> np.ndarray:
    """Build the ST-GCN style adjacency tensor of shape ``(max_hop + 1, V, V)``.

    The neighbourhood of every node is split into ``max_hop + 1`` subsets, one per
    hop distance. Each subset gets its own weight matrix in the spatial graph
    convolution, which is the "distance partitioning" of ST-GCN.
    """
    hop_distance = get_hop_distance(num_node, edges, max_hop)
    valid_hops = range(max_hop + 1)

    support = np.zeros((num_node, num_node))
    for hop in valid_hops:
        support[hop_distance == hop] = 1
    normalized = normalize_digraph(support)

    adjacency = np.zeros((max_hop + 1, num_node, num_node))
    for index, hop in enumerate(valid_hops):
        mask = hop_distance == hop
        adjacency[index][mask] = normalized[mask]
    return adjacency


def self_loop_edges(num_node: int) -> List[Edge]:
    return [(i, i) for i in range(num_node)]
