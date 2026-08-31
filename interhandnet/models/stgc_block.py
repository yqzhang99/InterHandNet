"""Spatial-Temporal Graph Convolution block of InterHandNet (Fig. 2 and Fig. 5).

The data flow inside one block is

1. spatial graph convolution, optionally with the Interaction Graph term of
   Eq. (2);
2. InterHand Temporal Fusion followed by the temporal convolution, which
   together form Eq. (4);
3. a Feature Extractor with an element-wise residual (Fig. 5);
4. Interaction Attention of Eq. (6) and Eq. (7), added back to the hand feature
   as drawn in Fig. 8;
5. a second Feature Extractor.

Following the reference implementation, the temporal stage has several parallel
branches with different kernel sizes. The branches share the spatial
convolution weights but own their edge-importance masks, and their outputs are
summed.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from ..graph import NUM_JOINTS
from ..modules import (
    FeatureExtractor,
    InterHandTemporalFusion,
    InteractionAttention,
    SpatialGraphConv,
    SpatialInteractionGraphConv,
)


class TemporalConv(nn.Module):
    """Temporal convolution ``F_k`` of Eq. (3), in pre-activation form."""

    def __init__(self, channels: int, kernel_size: int, stride: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"temporal kernel size must be odd, got {kernel_size}")
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=((kernel_size - 1) // 2, 0),
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class STGCBlock(nn.Module):
    """One Spatial-Temporal Graph Convolution block.

    Args:
        in_channels: Input channels ``C_l``.
        out_channels: Output channels ``C_{l+1}``.
        num_subsets: Number of adjacency subsets ``K`` of the physical graph.
        num_nodes: Number of joints ``V``, needed to size the edge-importance masks.
        stride: Temporal stride of the temporal convolutions.
        temporal_kernel_sizes: One odd kernel size per parallel temporal branch.
        num_heads: Attention heads used by the two attention modules.
        dropout: Dropout inside the temporal convolutions.
        attention_dropout: Dropout inside the Feature Extractors.
        use_interaction_graph: Enable the ``A_IG D`` term of Eq. (2).
        use_interhand_temporal_fusion: Enable Eq. (4).
        use_interaction_attention: Enable Eq. (6) and Eq. (7).
        distance_fusion: How ``A_IG`` and ``D`` are combined, see
            :class:`~interhandnet.modules.interaction_graph.SpatialInteractionGraphConv`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_subsets: int,
        num_nodes: int = NUM_JOINTS,
        stride: int = 1,
        temporal_kernel_sizes: Sequence[int] = (9, 5),
        num_heads: int = 4,
        dropout: float = 0.5,
        attention_dropout: float = 0.1,
        use_interaction_graph: bool = True,
        use_interhand_temporal_fusion: bool = True,
        use_interaction_attention: bool = True,
        distance_fusion: str = "matmul",
    ) -> None:
        super().__init__()
        if not temporal_kernel_sizes:
            raise ValueError("at least one temporal kernel size is required")

        self.stride = stride
        self.use_interaction_graph = use_interaction_graph

        if use_interaction_graph:
            self.spatial_conv: SpatialGraphConv = SpatialInteractionGraphConv(
                in_channels, out_channels, num_subsets, distance_fusion=distance_fusion
            )
        else:
            self.spatial_conv = SpatialGraphConv(in_channels, out_channels, num_subsets)

        num_branches = len(temporal_kernel_sizes)
        # ST-GCN style edge importance weighting: `A + M` of Eq. (1) is realised
        # as `A * M` with M initialised to ones. Each temporal branch learns its
        # own mask while sharing the spatial convolution weights.
        self.edge_importance = nn.ParameterList(
            nn.Parameter(torch.ones(num_subsets, num_nodes, num_nodes))
            for _ in range(num_branches)
        )
        self.interaction_edge_importance = (
            nn.ParameterList(
                nn.Parameter(torch.ones(num_nodes, num_nodes)) for _ in range(num_branches)
            )
            if use_interaction_graph
            else None
        )
        self.temporal_convs = nn.ModuleList(
            TemporalConv(out_channels, kernel_size, stride, dropout)
            for kernel_size in temporal_kernel_sizes
        )

        self.temporal_fusion = (
            InterHandTemporalFusion(out_channels, num_heads=num_heads)
            if use_interhand_temporal_fusion
            else None
        )
        self.interaction_attention = (
            InteractionAttention(out_channels, num_heads=num_heads)
            if use_interaction_attention
            else None
        )
        self.temporal_feature_extractor = FeatureExtractor(out_channels, attention_dropout)
        self.attention_feature_extractor = FeatureExtractor(out_channels, attention_dropout)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        interaction_adjacency: Optional[torch.Tensor] = None,
        distance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            x: ``(N, C_in, T, V)`` node features.
            adjacency: ``(K, V, V)`` physical adjacency ``A``.
            interaction_adjacency: ``(V, V)`` cross-hand adjacency ``A_IG``.
            distance: ``(N, T, V, V)`` distance matrix ``D``.
        """
        if self.use_interaction_graph and (interaction_adjacency is None or distance is None):
            raise ValueError(
                "the Interaction Graph needs both interaction_adjacency and distance"
            )

        branch_outputs = []
        for index, temporal_conv in enumerate(self.temporal_convs):
            weighted_adjacency = adjacency * self.edge_importance[index]
            if self.interaction_edge_importance is not None:
                spatial = self.spatial_conv(
                    x,
                    weighted_adjacency,
                    interaction_adjacency * self.interaction_edge_importance[index],
                    distance,
                )
            else:
                spatial = self.spatial_conv(x, weighted_adjacency)

            if self.temporal_fusion is not None:
                spatial = self.temporal_fusion(spatial)
            branch_outputs.append(temporal_conv(spatial))

        temporal = branch_outputs[0]
        for extra in branch_outputs[1:]:
            temporal = temporal + extra
        temporal = self.temporal_feature_extractor(temporal)

        if self.interaction_attention is not None:
            temporal = temporal + self.interaction_attention(temporal)
        return self.attention_feature_extractor(temporal)
