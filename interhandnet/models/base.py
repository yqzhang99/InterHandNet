"""Plumbing shared by every InterHandNet backbone.

Whatever the backbone is, the parts around the STGC blocks are the same: the two
graphs of Fig. 3 are registered as buffers, the raw coordinates are
batch-normalised per joint as in ST-GCN, and the distance matrix ``D`` of Eq. (2)
is computed from the coordinates before that normalisation so that it carries
metric distances.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..graph import NUM_JOINTS, HandGraph, InteractionGraph
from ..modules import pairwise_distance_matrix


def initialize_weights(module: nn.Module) -> None:
    """Apply the initialisation used across the repository, in place."""
    for submodule in module.modules():
        if isinstance(submodule, nn.Conv2d):
            nn.init.kaiming_normal_(submodule.weight, mode="fan_out", nonlinearity="relu")
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
        elif isinstance(submodule, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(submodule.weight)
            nn.init.zeros_(submodule.bias)
        elif isinstance(submodule, nn.Linear):
            nn.init.normal_(submodule.weight, std=0.01)
            nn.init.zeros_(submodule.bias)


class TwoHandBackbone(nn.Module):
    """Base class holding the graphs, the input normalisation and ``D``.

    Args:
        in_channels: Input channels. The first three must be the 3D coordinates.
        max_hop: Spatial neighbourhood size of the physical hand graph.
        interaction_graph_self_loops: Add self-loops to ``A_IG``.
        requires_distance: Whether any block consumes ``D``. When ``False`` the
            distance matrix is never built, which saves a ``(N, T, V, V)`` tensor.

    Attributes:
        adjacency: ``(K, V, V)`` physical adjacency ``A``.
        interaction_adjacency: ``(V, V)`` cross-hand adjacency ``A_IG``.
    """

    def __init__(
        self,
        in_channels: int,
        max_hop: int,
        interaction_graph_self_loops: bool,
        requires_distance: bool,
    ) -> None:
        super().__init__()
        if in_channels < 3:
            raise ValueError(f"in_channels must be at least 3 (x, y, z), got {in_channels}")

        self.in_channels = in_channels
        self.num_nodes = NUM_JOINTS
        self.requires_distance = requires_distance

        hand_graph = HandGraph(max_hop=max_hop)
        interaction_graph = InteractionGraph(self_loops=interaction_graph_self_loops)
        self.register_buffer(
            "adjacency", torch.tensor(hand_graph.A, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "interaction_adjacency",
            torch.tensor(interaction_graph.A, dtype=torch.float32),
            persistent=False,
        )

        self.input_bn = nn.BatchNorm1d(in_channels * self.num_nodes)

    @property
    def num_subsets(self) -> int:
        """Number of adjacency subsets ``K`` of the physical graph."""
        return self.adjacency.size(0)

    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError(f"expected a 4D input (N, C, T, V), got shape {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {x.size(1)}")
        if x.size(3) != self.num_nodes:
            raise ValueError(f"expected {self.num_nodes} joints, got {x.size(3)}")

    def _distance_matrix(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """``D`` of Eq. (2), built from the raw coordinates, or ``None``."""
        if not self.requires_distance:
            return None
        return pairwise_distance_matrix(x[:, :3])

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Batch-normalise per joint and channel, as in ST-GCN."""
        n, c, t, v = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)
        x = self.input_bn(x)
        return x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()
