"""Tests for the three modules proposed in the paper."""

import pytest
import torch

from interhandnet.graph import NUM_JOINTS, NUM_JOINTS_PER_HAND, HandGraph, InteractionGraph
from interhandnet.modules import (
    FeatureExtractor,
    InterHandTemporalFusion,
    InteractionAttention,
    SpatialGraphConv,
    SpatialInteractionGraphConv,
    pairwise_distance_matrix,
)

BATCH, CHANNELS, FRAMES = 2, 8, 6


def random_features(channels=CHANNELS, frames=FRAMES):
    return torch.randn(BATCH, channels, frames, NUM_JOINTS)


class TestDistanceMatrix:
    def test_shape_and_symmetry(self):
        coords = torch.randn(BATCH, 3, FRAMES, NUM_JOINTS)
        distance = pairwise_distance_matrix(coords)
        assert distance.shape == (BATCH, FRAMES, NUM_JOINTS, NUM_JOINTS)
        assert torch.allclose(distance, distance.transpose(-2, -1), atol=1e-6)

    def test_matches_manual_euclidean_distance(self):
        coords = torch.randn(1, 3, 1, NUM_JOINTS)
        distance = pairwise_distance_matrix(coords)
        expected = (coords[0, :, 0, 0] - coords[0, :, 0, 21]).norm()
        assert torch.allclose(distance[0, 0, 0, 21], expected, atol=1e-5)

    def test_gradients_are_finite_on_the_zero_diagonal(self):
        coords = torch.randn(1, 3, 2, NUM_JOINTS, requires_grad=True)
        pairwise_distance_matrix(coords).sum().backward()
        assert torch.isfinite(coords.grad).all()


class TestSpatialGraphConv:
    def test_output_shape(self):
        adjacency = torch.tensor(HandGraph(max_hop=1).A, dtype=torch.float32)
        conv = SpatialGraphConv(CHANNELS, 16, num_subsets=adjacency.size(0))
        out = conv(random_features(), adjacency)
        assert out.shape == (BATCH, 16, FRAMES, NUM_JOINTS)


class TestSpatialInteractionGraphConv:
    @pytest.mark.parametrize("distance_fusion", ["matmul", "hadamard"])
    def test_output_shape(self, distance_fusion):
        adjacency = torch.tensor(HandGraph(max_hop=1).A, dtype=torch.float32)
        interaction = torch.tensor(InteractionGraph().A, dtype=torch.float32)
        coords = torch.randn(BATCH, 3, FRAMES, NUM_JOINTS)
        distance = pairwise_distance_matrix(coords)

        conv = SpatialInteractionGraphConv(
            CHANNELS, 16, num_subsets=adjacency.size(0), distance_fusion=distance_fusion
        )
        out = conv(random_features(), adjacency, interaction, distance)
        assert out.shape == (BATCH, 16, FRAMES, NUM_JOINTS)

    def test_interaction_term_changes_the_output(self):
        """Eq. (2) must differ from Eq. (1); a zero distance collapses onto it."""
        adjacency = torch.tensor(HandGraph(max_hop=1).A, dtype=torch.float32)
        interaction = torch.tensor(InteractionGraph().A, dtype=torch.float32)
        features = random_features()
        distance = pairwise_distance_matrix(torch.randn(BATCH, 3, FRAMES, NUM_JOINTS))

        conv = SpatialInteractionGraphConv(CHANNELS, 16, num_subsets=adjacency.size(0))
        with_distance = conv(features, adjacency, interaction, distance)
        without_distance = conv(
            features, adjacency, interaction, torch.zeros_like(distance)
        )
        baseline = SpatialGraphConv.forward(conv, features, adjacency)

        assert not torch.allclose(with_distance, without_distance)
        assert torch.allclose(without_distance, baseline, atol=1e-5)

    def test_rejects_mismatched_temporal_length(self):
        adjacency = torch.tensor(HandGraph(max_hop=1).A, dtype=torch.float32)
        interaction = torch.tensor(InteractionGraph().A, dtype=torch.float32)
        conv = SpatialInteractionGraphConv(CHANNELS, 16, num_subsets=adjacency.size(0))
        distance = torch.zeros(BATCH, FRAMES + 1, NUM_JOINTS, NUM_JOINTS)
        with pytest.raises(ValueError, match="temporal length"):
            conv(random_features(), adjacency, interaction, distance)


class TestFeatureExtractor:
    def test_shape_is_preserved(self):
        extractor = FeatureExtractor(CHANNELS, dropout=0.0)
        x = random_features()
        assert extractor(x).shape == x.shape

    def test_residual_path_is_present(self):
        """With zeroed FC weights the module must return its input unchanged."""
        extractor = FeatureExtractor(CHANNELS, dropout=0.0).eval()
        with torch.no_grad():
            for layer in (extractor.fc1, extractor.fc2):
                layer.weight.zero_()
                layer.bias.zero_()
        x = random_features()
        assert torch.allclose(extractor(x), x)


class TestInterHandTemporalFusion:
    def test_output_shape(self):
        fusion = InterHandTemporalFusion(CHANNELS, num_heads=4).eval()
        x = random_features()
        assert fusion(x).shape == x.shape

    def test_rejects_indivisible_head_count(self):
        with pytest.raises(ValueError, match="divisible"):
            InterHandTemporalFusion(6, num_heads=4)

    def test_occluded_hand_is_reconstructed_from_other_time_steps(self):
        """Section III-D: an occluded left hand at time tau is filled in from the
        left hand at tau-1 and tau+1, queried by the right hand at tau."""
        fusion = InterHandTemporalFusion(CHANNELS, num_heads=2).eval()
        x = random_features()
        occluded = x.clone()
        occluded[:, :, 2, :NUM_JOINTS_PER_HAND] = 0.0

        with torch.no_grad():
            output = fusion(occluded)

        left_at_tau = output[:, :, 2, :NUM_JOINTS_PER_HAND]
        assert torch.isfinite(left_at_tau).all()
        assert left_at_tau.abs().sum() > 0

    def test_left_output_depends_on_left_hand_history(self):
        """Values come from the same hand's time window, so perturbing the left
        hand at another time step must move the left output at tau."""
        fusion = InterHandTemporalFusion(CHANNELS, num_heads=2).eval()
        x = random_features()
        x[:, :, 2, :NUM_JOINTS_PER_HAND] = 0.0

        perturbed = x.clone()
        perturbed[:, :, 1, :NUM_JOINTS_PER_HAND] += 5.0

        with torch.no_grad():
            before = fusion(x)[:, :, 2, :NUM_JOINTS_PER_HAND]
            after = fusion(perturbed)[:, :, 2, :NUM_JOINTS_PER_HAND]
        assert not torch.allclose(before, after)

    def test_left_output_depends_on_the_other_hand_query(self):
        """Queries come from the other hand, so perturbing the right hand at tau
        must move the left output at tau."""
        fusion = InterHandTemporalFusion(CHANNELS, num_heads=2).eval()
        x = random_features()
        perturbed = x.clone()
        perturbed[:, :, 2, NUM_JOINTS_PER_HAND:] += 5.0

        with torch.no_grad():
            before = fusion(x)[:, :, 2, :NUM_JOINTS_PER_HAND]
            after = fusion(perturbed)[:, :, 2, :NUM_JOINTS_PER_HAND]
        assert not torch.allclose(before, after)


class TestInteractionAttention:
    def test_output_shape(self):
        attention = InteractionAttention(CHANNELS, num_heads=4).eval()
        x = random_features()
        assert attention(x).shape == x.shape

    def test_each_half_is_driven_by_the_other_hand(self):
        """Eq. (6) and Eq. (7): the value tensor comes from the opposite hand, so
        perturbing one hand must change the other hand's output."""
        attention = InteractionAttention(CHANNELS, num_heads=2).eval()
        x = random_features()
        perturbed = x.clone()
        perturbed[:, :, :, NUM_JOINTS_PER_HAND:] += 3.0

        with torch.no_grad():
            before = attention(x)
            after = attention(perturbed)
        left_before = before[:, :, :, :NUM_JOINTS_PER_HAND]
        left_after = after[:, :, :, :NUM_JOINTS_PER_HAND]
        assert not torch.allclose(left_before, left_after)
