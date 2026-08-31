"""Tests for the STGC block and the full InterHandNet model."""

import pytest
import torch

from interhandnet.graph import NUM_JOINTS, HandGraph, InteractionGraph
from interhandnet.models import STAGCN, InterHandNet, STGCBlock, build_model
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
SMALL_STA_GCN = {
    "num_classes": NUM_CLASSES,
    "feature_channels": (8, 8),
    "feature_strides": (1, 1),
    "branch_channels": (16, 16),
    "branch_strides": (2, 1),
    "interaction_graph_blocks": (1,),
    "num_attention_edges": 2,
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

    @pytest.mark.parametrize("stride", [1, 2])
    def test_residual_projects_when_shape_changes(self, stride):
        adjacency = torch.tensor(HandGraph(max_hop=1).A, dtype=torch.float32)
        block = STGCBlock(
            8,
            16,
            num_subsets=adjacency.size(0),
            stride=stride,
            temporal_kernel_sizes=(3,),
            num_heads=2,
            use_interaction_graph=False,
            residual=True,
        ).eval()
        out = block(torch.randn(BATCH, 8, FRAMES, NUM_JOINTS), adjacency)
        assert out.shape == (BATCH, 16, (FRAMES + stride - 1) // stride, NUM_JOINTS)

    def test_residual_can_be_disabled(self):
        kwargs = dict(
            in_channels=8,
            out_channels=8,
            num_subsets=2,
            temporal_kernel_sizes=(3,),
            num_heads=2,
            use_interaction_graph=False,
        )
        assert STGCBlock(residual=True, **kwargs).residual is not None
        assert STGCBlock(residual=False, **kwargs).residual is None

    def test_attention_edges_add_a_data_dependent_subset(self):
        """STA-GCN feeds per-sample adjacency matrices; they must reach the graph
        product, so changing them must change the output."""
        adjacency = torch.tensor(HandGraph(max_hop=1).A, dtype=torch.float32)
        block = STGCBlock(
            8,
            16,
            num_subsets=adjacency.size(0),
            temporal_kernel_sizes=(3,),
            num_heads=2,
            use_interaction_graph=False,
            num_attention_edges=2,
        ).eval()
        x = torch.randn(BATCH, 8, FRAMES, NUM_JOINTS)
        edges = torch.rand(BATCH, 2, NUM_JOINTS, NUM_JOINTS)

        with torch.no_grad():
            first = block(x, adjacency, attention_edges=edges)
            second = block(x, adjacency, attention_edges=edges * 0.5)
        assert first.shape == (BATCH, 16, FRAMES, NUM_JOINTS)
        assert not torch.allclose(first, second)

    def test_missing_attention_edges_is_an_error(self):
        adjacency = torch.tensor(HandGraph(max_hop=1).A, dtype=torch.float32)
        block = STGCBlock(
            8,
            16,
            num_subsets=adjacency.size(0),
            temporal_kernel_sizes=(3,),
            num_heads=2,
            use_interaction_graph=False,
            num_attention_edges=1,
        )
        with pytest.raises(ValueError, match="attention_edges"):
            block(torch.randn(BATCH, 8, FRAMES, NUM_JOINTS), adjacency)


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


class TestSTAGCN:
    def test_forward_shape(self):
        model = STAGCN(**SMALL_STA_GCN).eval()
        with torch.no_grad():
            logits = model(skeleton_batch())
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_auxiliary_head_returns_both_branches(self):
        model = STAGCN(**SMALL_STA_GCN).eval()
        with torch.no_grad():
            prediction, auxiliary = model.forward_with_auxiliary(skeleton_batch())
        assert prediction.shape == auxiliary.shape == (BATCH, NUM_CLASSES)

    def test_forward_returns_the_perception_branch(self):
        model = STAGCN(**SMALL_STA_GCN).eval()
        x = skeleton_batch()
        with torch.no_grad():
            assert torch.allclose(model(x), model.forward_with_auxiliary(x)[0])

    def test_attention_gate_and_edges_have_the_expected_shapes(self):
        model = STAGCN(**SMALL_STA_GCN).eval()
        with torch.no_grad():
            feature = model.extract_features(skeleton_batch())
            _, gate, edges = model._attention_branch(feature)
        # The gate is resampled back to the feature resolution so it can scale it.
        assert gate.shape == (BATCH, 1, feature.size(2), NUM_JOINTS)
        assert edges.shape == (BATCH, SMALL_STA_GCN["num_attention_edges"], NUM_JOINTS, NUM_JOINTS)
        # relu(tanh(.)) is a gate in [0, 1).
        assert (edges >= 0).all() and (edges < 1).all()
        assert (gate > 0).all() and (gate < 1).all()

    def test_gradients_reach_both_branches(self):
        model = STAGCN(**SMALL_STA_GCN)
        prediction, auxiliary = model.forward_with_auxiliary(skeleton_batch())
        (prediction.sum() + auxiliary.sum()).backward()

        for name in ("attention_blocks", "perception_blocks", "feature_blocks"):
            grads = [
                parameter.grad
                for parameter in getattr(model, name).parameters()
                if parameter.grad is not None
            ]
            assert grads, f"no gradient reached {name}"
            assert any(float(grad.abs().sum()) > 0 for grad in grads)

    def test_ablation_switches_change_the_parameter_count(self):
        full = STAGCN(**SMALL_STA_GCN)
        baseline = STAGCN(
            **SMALL_STA_GCN,
            use_interaction_graph=False,
            use_interaction_attention=False,
            use_interhand_temporal_fusion=False,
        )
        assert sum(p.numel() for p in baseline.parameters()) < sum(
            p.numel() for p in full.parameters()
        )

    def test_requires_at_least_one_attention_edge(self):
        with pytest.raises(ValueError, match="attention edge"):
            STAGCN(**{**SMALL_STA_GCN, "num_attention_edges": 0})


class TestBuilder:
    def test_build_from_config(self):
        model = build_model({"name": "interhandnet", **SMALL_MODEL})
        assert isinstance(model, InterHandNet)

    def test_build_sta_gcn_from_config(self):
        model = build_model({"name": "interhandnet_sta_gcn", **SMALL_STA_GCN})
        assert isinstance(model, STAGCN)

    def test_unknown_model_name(self):
        with pytest.raises(KeyError, match="unknown model"):
            build_model({"name": "does-not-exist"})

    def test_key_of_the_other_backbone_is_rejected(self):
        """`block_channels` describes the ST-GCN backbone only. Passing it to the
        STA-GCN backbone means the config inherited from the wrong parent."""
        with pytest.raises(KeyError, match="block_channels"):
            build_model(
                {"name": "interhandnet_sta_gcn", **SMALL_STA_GCN, "block_channels": [8, 8]}
            )


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
