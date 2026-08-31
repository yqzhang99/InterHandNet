"""InterHandNet with an STA-GCN backbone (Section III-F, Table II and Table III).

STA-GCN [22] splits the network into a feature extractor followed by two
branches. The attention branch predicts, from the shared feature, an attention
map over the nodes and a set of data-dependent ("attention") edges; the
perception branch then classifies the attention-weighted feature while its
spatial graph convolutions additionally use those predicted edges. Both branches
emit logits and both are supervised, which is why :meth:`STAGCN.forward_with_auxiliary`
exists.

The three proposed modules of InterHandNet live inside the STGC blocks, so this
backbone reuses :class:`~interhandnet.models.stgc_block.STGCBlock` unchanged.
Turning all three flags off therefore gives the plain STA-GCN baseline of
Table III, and turning them on gives ``InterHandNet(STA-GCN)``, the row with the
best accuracy, recall and F1 score of Table II.

Following the reference implementation, the Interaction Graph is applied inside
the feature extractor only: the branches run at a reduced temporal resolution,
where the distance matrix ``D`` no longer lines up with the features.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .base import TwoHandBackbone, initialize_weights
from .stgc_block import STGCBlock

DEFAULT_FEATURE_CHANNELS = (32, 32, 32)
DEFAULT_FEATURE_STRIDES = (1, 1, 1)
DEFAULT_BRANCH_CHANNELS = (64, 64, 64)
DEFAULT_BRANCH_STRIDES = (2, 1, 1)
DEFAULT_INTERACTION_GRAPH_BLOCKS = (1, 2)


class STAGCN(TwoHandBackbone):
    """InterHandNet with an STA-GCN backbone.

    Args:
        num_classes: Number of output classes. The default of 7 matches the label
            set of the Lulla et al. dataset: class 0 is "other movement" and
            classes 1..6 are the six WHO steps.
        in_channels: Input channels. The first three must be the 3D coordinates.
        feature_channels: Output channels of the feature extractor blocks.
        feature_strides: Temporal strides of the feature extractor blocks.
        branch_channels: Output channels of the blocks in each branch.
        branch_strides: Temporal strides of the blocks in each branch.
        interaction_graph_blocks: Indices of the *feature extractor* blocks that
            use the Interaction Graph.
        num_attention_edges: Number of attention edges predicted per sample.
        max_hop: Spatial neighbourhood size of the physical hand graph.
        temporal_kernel_sizes: Kernel size of each parallel temporal branch. Use
            a single kernel to match the original STA-GCN temporal convolution.
        num_heads: Attention heads of both attention modules.
        dropout: Dropout inside the temporal convolutions.
        attention_dropout: Dropout inside the Feature Extractors.
        use_interaction_graph: Master switch for Eq. (2).
        use_interhand_temporal_fusion: Master switch for Eq. (4).
        use_interaction_attention: Master switch for Eq. (6) and Eq. (7).
        interaction_graph_self_loops: Add self-loops to ``A_IG``.
        distance_fusion: ``"matmul"`` or ``"hadamard"``.
        attention_scope: Scope of Interaction Attention.
        residual: Add a block-level residual connection to every STGC block.

    Shape:
        - Input: ``(N, C_in, T, V)`` with ``V = 42`` joints, left hand first.
        - Output: ``(N, num_classes)`` logits of the perception branch.
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 3,
        feature_channels: Sequence[int] = DEFAULT_FEATURE_CHANNELS,
        feature_strides: Sequence[int] = DEFAULT_FEATURE_STRIDES,
        branch_channels: Sequence[int] = DEFAULT_BRANCH_CHANNELS,
        branch_strides: Sequence[int] = DEFAULT_BRANCH_STRIDES,
        interaction_graph_blocks: Sequence[int] = DEFAULT_INTERACTION_GRAPH_BLOCKS,
        num_attention_edges: int = 2,
        max_hop: int = 2,
        temporal_kernel_sizes: Sequence[int] = (5, 15),
        num_heads: int = 4,
        dropout: float = 0.5,
        attention_dropout: float = 0.1,
        use_interaction_graph: bool = True,
        use_interhand_temporal_fusion: bool = True,
        use_interaction_attention: bool = True,
        interaction_graph_self_loops: bool = False,
        distance_fusion: str = "matmul",
        attention_scope: str = "spatial",
        residual: bool = True,
    ) -> None:
        if len(feature_channels) != len(feature_strides):
            raise ValueError(
                "feature_channels and feature_strides must have the same length, got "
                f"{len(feature_channels)} and {len(feature_strides)}"
            )
        if len(branch_channels) != len(branch_strides):
            raise ValueError(
                "branch_channels and branch_strides must have the same length, got "
                f"{len(branch_channels)} and {len(branch_strides)}"
            )
        if not feature_channels or not branch_channels:
            raise ValueError("both the feature extractor and the branches need a block")
        if num_attention_edges < 1:
            raise ValueError(
                f"STA-GCN needs at least one attention edge, got {num_attention_edges}"
            )

        interaction_blocks = set(interaction_graph_blocks) if use_interaction_graph else set()
        super().__init__(
            in_channels=in_channels,
            max_hop=max_hop,
            interaction_graph_self_loops=interaction_graph_self_loops,
            requires_distance=bool(interaction_blocks),
        )

        self.num_classes = num_classes
        self.num_attention_edges = num_attention_edges

        block_kwargs = dict(
            num_subsets=self.num_subsets,
            num_nodes=self.num_nodes,
            temporal_kernel_sizes=temporal_kernel_sizes,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_interhand_temporal_fusion=use_interhand_temporal_fusion,
            use_interaction_attention=use_interaction_attention,
            distance_fusion=distance_fusion,
            attention_scope=attention_scope,
            residual=residual,
        )

        self.feature_blocks, feature_out = self._make_blocks(
            in_channels,
            feature_channels,
            feature_strides,
            interaction_blocks=interaction_blocks,
            num_attention_edges=0,
            **block_kwargs,
        )
        self.attention_blocks, attention_out = self._make_blocks(
            feature_out,
            branch_channels,
            branch_strides,
            interaction_blocks=set(),
            num_attention_edges=0,
            **block_kwargs,
        )
        self.perception_blocks, perception_out = self._make_blocks(
            feature_out,
            branch_channels,
            branch_strides,
            interaction_blocks=set(),
            num_attention_edges=num_attention_edges,
            **block_kwargs,
        )

        self.attention_classifier = nn.Linear(attention_out, num_classes)
        self.perception_classifier = nn.Linear(perception_out, num_classes)

        # Attention head of STA-GCN: a class-wise activation map is reduced once
        # to a per-node gate and once to a set of node-to-node edges.
        self.attention_bn = nn.BatchNorm2d(attention_out)
        self.attention_conv = nn.Conv2d(attention_out, num_classes, kernel_size=1, bias=False)
        self.node_conv = nn.Conv2d(num_classes, 1, kernel_size=1, bias=False)
        self.node_bn = nn.BatchNorm2d(1)
        self.edge_conv = nn.Conv2d(
            num_classes, num_attention_edges * self.num_nodes, kernel_size=1, bias=False
        )
        self.edge_bn = nn.BatchNorm2d(num_attention_edges * self.num_nodes)

        initialize_weights(self)

    @staticmethod
    def _make_blocks(
        in_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        interaction_blocks: set,
        num_attention_edges: int,
        **block_kwargs,
    ) -> Tuple[nn.ModuleList, int]:
        blocks = []
        current = in_channels
        for index, (out_channels, stride) in enumerate(zip(channels, strides)):
            blocks.append(
                STGCBlock(
                    in_channels=current,
                    out_channels=out_channels,
                    stride=stride,
                    use_interaction_graph=index in interaction_blocks,
                    num_attention_edges=num_attention_edges,
                    **block_kwargs,
                )
            )
            current = out_channels
        return nn.ModuleList(blocks), current

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the feature extractor and return its ``(N, C, T, V)`` output."""
        self._check_input(x)
        distance: Optional[torch.Tensor] = self._distance_matrix(x)
        x = self._normalize_input(x)
        for block in self.feature_blocks:
            x = block(x, self.adjacency, self.interaction_adjacency, distance)
            if distance is not None and block.stride > 1:
                distance = distance[:, :: block.stride]
        return x

    def _attention_branch(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the branch logits, the node gate and the attention edges.

        The node gate is resampled to the temporal length of ``x`` so that it can
        gate the feature extractor output directly.
        """
        n, _, num_frames, num_nodes = x.shape
        for block in self.attention_blocks:
            x = block(x, self.adjacency)

        logits = self.attention_classifier(x.mean(dim=(2, 3)))

        class_map = self.attention_conv(self.attention_bn(x))  # (N, num_classes, T', V)

        gate = self.node_bn(self.node_conv(class_map))  # (N, 1, T', V)
        gate = F.interpolate(gate, size=(num_frames, num_nodes))
        node_attention = torch.sigmoid(gate)

        edges = class_map.mean(dim=2, keepdim=True)  # (N, num_classes, 1, V)
        edges = self.edge_bn(self.edge_conv(edges))  # (N, A * V, 1, V)
        edges = edges.view(n, self.num_attention_edges, num_nodes, num_nodes)
        edge_attention = F.relu(torch.tanh(edges))

        return logits, node_attention, edge_attention

    def _perception_branch(
        self, x: torch.Tensor, attention_edges: torch.Tensor
    ) -> torch.Tensor:
        for block in self.perception_blocks:
            x = block(x, self.adjacency, attention_edges=attention_edges)
        return self.perception_classifier(x.mean(dim=(2, 3)))

    def forward_with_auxiliary(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(perception_logits, attention_logits)``.

        STA-GCN supervises both branches, so training should use this method and
        sum the two cross-entropy terms. Prediction uses the perception logits.
        """
        feature = self.extract_features(x)
        attention_logits, node_attention, attention_edges = self._attention_branch(feature)
        perception_logits = self._perception_branch(feature * node_attention, attention_edges)
        return perception_logits, attention_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_auxiliary(x)[0]
