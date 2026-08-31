"""Tests for the skeleton graphs of Fig. 3."""

import numpy as np
import pytest

from interhandnet.graph import (
    NUM_JOINTS,
    NUM_JOINTS_PER_HAND,
    HandGraph,
    InteractionGraph,
    cross_hand_edges,
    two_hand_edges,
)
from interhandnet.graph.utils import get_hop_distance, normalize_digraph


def test_joint_counts():
    assert NUM_JOINTS_PER_HAND == 21
    assert NUM_JOINTS == 42


def test_two_hand_edges_have_20_bones_per_hand():
    edges = two_hand_edges()
    assert len(edges) == 40
    left = [edge for edge in edges if max(edge) < NUM_JOINTS_PER_HAND]
    right = [edge for edge in edges if min(edge) >= NUM_JOINTS_PER_HAND]
    assert len(left) == len(right) == 20
    # The two hands are structurally identical, offset by 21.
    assert sorted(left) == sorted((i - 21, j - 21) for i, j in right)


def test_cross_hand_edges_pair_corresponding_keypoints():
    edges = cross_hand_edges()
    assert len(edges) == NUM_JOINTS_PER_HAND
    assert all(j - i == NUM_JOINTS_PER_HAND for i, j in edges)


@pytest.mark.parametrize("max_hop", [0, 1, 2])
def test_hand_graph_shapes_and_normalisation(max_hop):
    graph = HandGraph(max_hop=max_hop)
    assert graph.A.shape == (max_hop + 1, NUM_JOINTS, NUM_JOINTS)
    # Summing the partitions recovers the column-normalised neighbourhood, whose
    # columns sum to one wherever the node has neighbours.
    column_sums = graph.A.sum(axis=0).sum(axis=0)
    assert np.allclose(column_sums, 1.0)


def test_hand_graph_has_no_cross_hand_connection():
    graph = HandGraph(max_hop=1)
    combined = graph.A.sum(axis=0)
    left_to_right = combined[:NUM_JOINTS_PER_HAND, NUM_JOINTS_PER_HAND:]
    assert np.all(left_to_right == 0)


def test_interaction_graph_is_the_hand_swapping_permutation():
    graph = InteractionGraph()
    assert graph.A.shape == (NUM_JOINTS, NUM_JOINTS)

    expected = np.zeros((NUM_JOINTS, NUM_JOINTS))
    for i in range(NUM_JOINTS_PER_HAND):
        expected[i, i + NUM_JOINTS_PER_HAND] = 1.0
        expected[i + NUM_JOINTS_PER_HAND, i] = 1.0
    assert np.allclose(graph.A, expected)
    assert np.all(np.diag(graph.A) == 0)


def test_interaction_graph_with_self_loops_keeps_a_diagonal():
    graph = InteractionGraph(self_loops=True)
    assert np.allclose(np.diag(graph.A), 0.5)
    assert np.allclose(graph.A.sum(axis=0), 1.0)


def test_hop_distance_of_a_path():
    hop = get_hop_distance(3, [(0, 1), (1, 2)], max_hop=2)
    assert hop[0, 0] == 0
    assert hop[0, 1] == 1
    assert hop[0, 2] == 2


def test_normalize_digraph_columns_sum_to_one():
    adjacency = np.array([[1.0, 1.0], [1.0, 1.0]])
    normalized = normalize_digraph(adjacency)
    assert np.allclose(normalized.sum(axis=0), 1.0)
