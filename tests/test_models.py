"""Tests for the STGC block and the full InterHandNet model."""

import pytest
import torch

from interhandnet.graph import NUM_JOINTS, HandGraph, InteractionGraph
from interhandnet.models import InterHandNet, STGCBlock, build_model
from interhandnet.modules import pairwise_distance_matrix

BATCH, FRAMES, NUM_CLASSES = 2, 12, 6
SMALL_MODEL = {
    "num_classes": NUM_CLASSES,
    "block_channels": (8, 16),
    "block_strides": (1, 2),
    "interaction_graph_blocks": (0,),
    "temporal_kernel_sizes": (3,),
    "num_heads": 2,
}


def skeleton_batch(frames=FRAMES, channels=3):
    return torch.randn(BATCH, channels, frames, NUM_JOINTS)


class TestSTGCBlock:
    @pytest.mark.parametrize("stride", [1, 2])
    def test_output_shape(self, stride):
        adjacency = torch.tensor(HandGraph(max_hop=1).A, dtype=torch.float32)
        interaction = torch.tensor(InteractionGraph().A, dtype=torch.float32)
        distance = pairwise_distance_matrix(skeleton_batch())

        block = STGCBlock(
            in_channels=8,
            out_channels=16,
            num_subsets=adjacency.size(0),
            stride=stride,
            temporal_kernel_sizes=(3, 5),
            num_heads=2,
        ).eval()
        out = block(torch.randn(BATCH, 8, FRAMES, NUM_JOINTS), adjacency, interaction, distance)
        assert out.shape == (BATCH, 16, (FRAMES + stride - 1) // stride, NUM_JOINTS)

    def test_interaction_graph_requires_distance(self):
        adjacency = torch.tensor(HandGraph(max_hop=1).A, dtype=torch.float32)
        block = STGCBlock(8, 16, num_subsets=adjacency.size(0), num_heads=2)
        with pytest.raises(ValueError, match="Interaction Graph needs"):
            block(torch.randn(BATCH, 8, FRAMES, NUM_JOINTS), adjacency)

    def test_even_temporal_kernel_is_rejected(self):
        with pytest.raises(ValueError, match="odd"):
            STGCBlock(8, 16, num_subsets=2, temporal_kernel_sizes=(4,))

    def test_edge_importance_is_learnable_per_branch(self):
        block = STGCBlock(8, 16, num_subsets=2, temporal_kernel_sizes=(3, 5), num_heads=2)
        assert len(block.edge_importance) == 2
        assert all(mask.shape == (2, NUM_JOINTS, NUM_JOINTS) for mask in block.edge_importance)
        assert all(mask.requires_grad for mask in block.edge_importance)


class TestInterHandNet:
    def test_forward_shape(self):
        model = InterHandNet(**SMALL_MODEL).eval()
        assert model(skeleton_batch()).shape == (BATCH, NUM_CLASSES)

    def test_default_configuration_matches_the_paper(self):
        model = InterHandNet(num_classes=NUM_CLASSES)
        assert len(model.blocks) == 6
        assert [block.stride for block in model.blocks] == [1, 1, 1, 2, 1, 1]
        assert [block.use_interaction_graph for block in model.blocks] == [
            True,
            True,
            True,
            False,
            False,
            False,
        ]

    def test_full_window_of_thirty_frames(self):
        """Section IV-A resamples every window to 30 frames."""
        model = InterHandNet(**SMALL_MODEL).eval()
        assert model(skeleton_batch(frames=30)).shape == (BATCH, NUM_CLASSES)

    @pytest.mark.parametrize(
        ("interaction_graph", "temporal_fusion", "interaction_attention"),
        [
            (False, False, False),  # ST-GCN baseline
            (True, False, False),  # +IG
            (False, False, True),  # +IA
            (True, False, True),  # +IG/IA
            (True, True, True),  # +IG/IA/ITF, full InterHandNet
        ],
    )
    def test_ablation_switches(self, interaction_graph, temporal_fusion, interaction_attention):
        model = InterHandNet(
            **SMALL_MODEL,
            use_interaction_graph=interaction_graph,
            use_interhand_temporal_fusion=temporal_fusion,
            use_interaction_attention=interaction_attention,
        ).eval()
        assert model.requires_distance is interaction_graph
        assert model(skeleton_batch()).shape == (BATCH, NUM_CLASSES)

    def test_disabled_modules_remove_their_parameters(self):
        baseline = InterHandNet(
            **SMALL_MODEL,
            use_interaction_graph=False,
            use_interhand_temporal_fusion=False,
            use_interaction_attention=False,
        )
        full = InterHandNet(**SMALL_MODEL)
        baseline_size = sum(p.numel() for p in baseline.parameters())
        full_size = sum(p.numel() for p in full.parameters())
        assert full_size > baseline_size

    def test_gradients_reach_every_parameter(self):
        model = InterHandNet(**SMALL_MODEL)
        model(skeleton_batch()).sum().backward()
        missing = [
            name for name, parameter in model.named_parameters() if parameter.grad is None
        ]
        assert not missing, f"parameters without gradient: {missing}"

    def test_distance_matrix_is_downsampled_with_the_stride(self):
        """A strided block halves T, so D must be resampled to stay aligned."""
        model = InterHandNet(
            **{**SMALL_MODEL, "block_strides": (2, 2), "interaction_graph_blocks": (0, 1)}
        ).eval()
        assert model(skeleton_batch(frames=16)).shape == (BATCH, NUM_CLASSES)

    def test_rejects_wrong_joint_count(self):
        model = InterHandNet(**SMALL_MODEL).eval()
        with pytest.raises(ValueError, match="joints"):
            model(torch.randn(BATCH, 3, FRAMES, 21))

    def test_rejects_too_few_input_channels(self):
        with pytest.raises(ValueError, match="at least 3"):
            InterHandNet(**{**SMALL_MODEL, "in_channels": 2})

    def test_occluded_hand_still_produces_a_prediction(self):
        """MediaPipe writes zeros for an undetected hand; the model must cope."""
        model = InterHandNet(**SMALL_MODEL).eval()
        x = skeleton_batch()
        x[:, :, 3:6, :21] = 0.0
        with torch.no_grad():
            logits = model(x)
        assert torch.isfinite(logits).all()

    def test_eval_mode_is_deterministic(self):
        model = InterHandNet(**SMALL_MODEL).eval()
        x = skeleton_batch()
        with torch.no_grad():
            assert torch.allclose(model(x), model(x))


class TestBuilder:
    def test_build_from_config(self):
        model = build_model({"name": "interhandnet", **SMALL_MODEL})
        assert isinstance(model, InterHandNet)

    def test_unknown_model_name(self):
        with pytest.raises(KeyError, match="unknown model"):
            build_model({"name": "does-not-exist"})


class TestOnnxExport:
    def test_export_runs(self, tmp_path):
        model = InterHandNet(**SMALL_MODEL).eval()
        destination = tmp_path / "model.onnx"
        torch.onnx.export(
            model,
            torch.zeros(1, 3, 30, NUM_JOINTS),
            str(destination),
            input_names=["skeleton"],
            output_names=["logits"],
            opset_version=16,
        )
        assert destination.exists() and destination.stat().st_size > 0

    def test_exported_graph_is_valid(self, tmp_path):
        onnx = pytest.importorskip("onnx")
        model = InterHandNet(**SMALL_MODEL).eval()
        destination = tmp_path / "model.onnx"
        torch.onnx.export(
            model,
            torch.zeros(1, 3, 30, NUM_JOINTS),
            str(destination),
            input_names=["skeleton"],
            output_names=["logits"],
            opset_version=16,
        )
        onnx.checker.check_model(onnx.load(str(destination)))
