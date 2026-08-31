"""Interaction Graph (Section III-C).

The standard spatial graph convolution of ST-GCN style backbones is

.. math:: f^S_{l,t} = W f_{l,t} (A + M)                              \\qquad (1)

and the Interaction Graph replaces it with

.. math:: f^S_{l,t} = W f_{l,t} (A_{IG} D + A + M)                    \\qquad (2)

where ``A_IG`` holds the cross-hand edges of Fig. 3 and ``D`` is the distance
matrix at time ``t``. Keeping the additive form of Eq. (1) is what makes the
module droppable into any STGCN-based backbone.

Following ST-GCN, ``M`` is realised as a learnable multiplicative edge-importance
mask, so ``A + M`` is implemented as ``A * M`` with ``M`` initialised to ones.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

DistanceFusion = str
_VALID_FUSIONS = ("matmul", "hadamard")


def pairwise_distance_matrix(coords: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Euclidean distances between every pair of joints at every time step.

    Args:
        coords: ``(N, 3, T, V)`` tensor of 3D joint coordinates.
        eps: Floor applied to the squared distances. Without it the zero
            diagonal produces infinite gradients through ``sqrt``.

    Returns:
        ``(N, T, V, V)`` distance matrix ``D``.
    """
    xyz = coords.permute(0, 2, 3, 1)  # (N, T, V, 3)
    diff = xyz.unsqueeze(-2) - xyz.unsqueeze(-3)  # (N, T, V, V, 3)
    return diff.pow(2).sum(-1).clamp_min(eps).sqrt()


class SpatialGraphConv(nn.Module):
    """Spatial graph convolution of Eq. (1) with distance partitioning.

    The neighbourhood is split into ``K = max_hop + 1`` subsets by hop distance,
    each with its own weight matrix.

    Backbones such as STA-GCN additionally feed data-dependent adjacency matrices
    ("attention edges") that are predicted per sample. Requesting
    ``num_attention_edges > 0`` adds that many extra subsets, whose graph product
    uses the per-sample matrices instead of the fixed ``A``.

    Args:
        in_channels: Input channels ``C_l``.
        out_channels: Output channels ``C_{l+1}``.
        num_subsets: Number of fixed adjacency subsets ``K`` (``max_hop + 1``).
        num_attention_edges: Number of per-sample adjacency subsets.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_subsets: int,
        num_attention_edges: int = 0,
    ) -> None:
        super().__init__()
        if num_attention_edges < 0:
            raise ValueError(
                f"num_attention_edges must be non-negative, got {num_attention_edges}"
            )
        self.num_subsets = num_subsets
        self.num_attention_edges = num_attention_edges
        self.out_channels = out_channels
        total_subsets = num_subsets + num_attention_edges
        self.conv = nn.Conv2d(in_channels, out_channels * total_subsets, kernel_size=1)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, C_in, T, V)`` -> ``(N, K + A, C_out, T, V)``."""
        x = self.conv(x)
        n, _, t, v = x.shape
        return x.view(n, self.num_subsets + self.num_attention_edges, self.out_channels, t, v)

    def _attention_edge_product(
        self, projected: torch.Tensor, attention_edges: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Graph product over the per-sample adjacency subsets, if any."""
        if not self.num_attention_edges:
            return None
        if attention_edges is None:
            raise ValueError(
                "this convolution was built with num_attention_edges="
                f"{self.num_attention_edges} but no attention_edges were passed"
            )
        return torch.einsum(
            "nkctv,nkvw->nctw", projected[:, self.num_subsets :], attention_edges
        )

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        attention_edges: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            x: ``(N, C_in, T, V)`` node features.
            adjacency: ``(K, V, V)`` adjacency tensor, already multiplied by the
                edge-importance mask.
            attention_edges: ``(N, A, V, V)`` per-sample adjacency matrices.
        """
        projected = self._project(x)
        out = torch.einsum(
            "nkctv,kvw->nctw", projected[:, : self.num_subsets], adjacency
        )
        data_dependent = self._attention_edge_product(projected, attention_edges)
        if data_dependent is not None:
            out = out + data_dependent
        return out.contiguous()


class SpatialInteractionGraphConv(SpatialGraphConv):
    """Spatial graph convolution with the Interaction Graph term of Eq. (2).

    Args:
        in_channels: Input channels ``C_l``.
        out_channels: Output channels ``C_{l+1}``.
        num_subsets: Number of adjacency subsets ``K``.
        distance_fusion: How ``A_IG`` and ``D`` are combined.

            * ``"matmul"`` (default) computes the matrix product ``A_IG @ D``,
              reproducing the original research code that produced the reported
              results. Because ``A_IG`` is a permutation, row ``i`` of the
              product carries the distances seen from the corresponding joint of
              the other hand.
            * ``"hadamard"`` computes ``A_IG * D``, which keeps exactly the
              lengths of the red cross-hand edges of Fig. 3 and matches the
              narrower reading of "the length of the red edge" in Section III-C.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_subsets: int,
        num_attention_edges: int = 0,
        distance_fusion: DistanceFusion = "matmul",
    ) -> None:
        super().__init__(in_channels, out_channels, num_subsets, num_attention_edges)
        if distance_fusion not in _VALID_FUSIONS:
            raise ValueError(
                f"distance_fusion must be one of {_VALID_FUSIONS}, got {distance_fusion!r}"
            )
        self.distance_fusion = distance_fusion

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        interaction_adjacency: torch.Tensor,
        distance: torch.Tensor,
        attention_edges: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            x: ``(N, C_in, T, V)`` node features.
            adjacency: ``(K, V, V)`` physical adjacency times edge importance.
            interaction_adjacency: ``(V, V)`` cross-hand adjacency ``A_IG`` times
                its edge-importance mask.
            distance: ``(N, T, V, V)`` distance matrix ``D``.
            attention_edges: ``(N, A, V, V)`` per-sample adjacency matrices.
        """
        if distance.size(1) != x.size(2):
            raise ValueError(
                "distance matrix and features disagree on the temporal length: "
                f"{distance.size(1)} vs {x.size(2)}"
            )

        projected = self._project(x)
        fixed = projected[:, : self.num_subsets]
        out = torch.einsum("nkctv,kvw->nctw", fixed, adjacency)

        data_dependent = self._attention_edge_product(projected, attention_edges)
        if data_dependent is not None:
            out = out + data_dependent

        if self.distance_fusion == "matmul":
            weighted = torch.matmul(interaction_adjacency, distance)
        else:
            weighted = interaction_adjacency * distance

        # A_IG does not depend on the adjacency subset, so the subsets can be
        # summed before the graph product instead of once per subset.
        pooled = fixed.sum(dim=1).permute(0, 2, 1, 3)  # (N, T, C_out, V)
        interaction = torch.matmul(pooled, weighted).permute(0, 2, 1, 3)  # (N, C_out, T, V)

        return (out + interaction).contiguous()
