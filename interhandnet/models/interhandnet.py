"""InterHandNet (Section III-A, Fig. 2).

The network stacks ``L`` STGC blocks on top of an input batch-normalisation, and
predicts the hand-washing step with a fully connected layer on the globally
pooled feature map.

The three proposed modules are individually switchable, which reproduces every
row of the ablation studies in Table III and Table IV:

======================================  =========================================
Configuration                            Flags
======================================  =========================================
ST-GCN baseline                          all three flags ``False``
``+IG``                                  ``interaction_graph=True``
``+IA``                                  ``interaction_attention=True``
``+IG/IA``                               both of the above
``+IG/IA/ITF`` (full InterHandNet)       all three flags ``True``
======================================  =========================================
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from ..graph import NUM_JOINTS, HandGraph, InteractionGraph
from ..modules import pairwise_distance_matrix
from .stgc_block import STGCBlock

DEFAULT_BLOCK_CHANNELS = (32, 32, 32, 64, 64, 64)
DEFAULT_BLOCK_STRIDES = (1, 1, 1, 2, 1, 1)
DEFAULT_INTERACTION_GRAPH_BLOCKS = (0, 1, 2)


class InterHandNet(nn.Module):
    """InterHandNet with an ST-GCN backbone.

    Args:
        num_classes: Number of hand-washing steps (six WHO steps by default).
        in_channels: Input channels. The first three must be the 3D joint
            coordinates, because the distance matrix ``D`` is computed from them.
        block_channels: Output channels of each STGC block.
        block_strides: Temporal stride of each STGC block.
        interaction_graph_blocks: Indices of the blocks that use the Interaction
            Graph. The reference configuration applies it to the first three
            blocks, which run at full temporal resolution.
        max_hop: Spatial neighbourhood size of the physical hand graph.
        temporal_kernel_sizes: Kernel size of each parallel temporal branch.
        num_heads: Attention heads of both attention modules.
        dropout: Dropout inside the temporal convolutions.
        attention_dropout: Dropout inside the Feature Extractors.
        use_interaction_graph: Master switch for Eq. (2).
        use_interhand_temporal_fusion: Master switch for Eq. (4).
        use_interaction_attention: Master switch for Eq. (6) and Eq. (7).
        interaction_graph_self_loops: Add self-loops to ``A_IG`` before
            normalisation. The paper defines cross-hand edges only.
        distance_fusion: ``"matmul"`` or ``"hadamard"``, see
            :class:`~interhandnet.modules.interaction_graph.SpatialInteractionGraphConv`.

    Shape:
        - Input: ``(N, C_in, T, V)`` with ``V = 42`` joints, left hand first.
        - Output: ``(N, num_classes)`` class logits.
    """

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 3,
        block_channels: Sequence[int] = DEFAULT_BLOCK_CHANNELS,
        block_strides: Sequence[int] = DEFAULT_BLOCK_STRIDES,
        interaction_graph_blocks: Sequence[int] = DEFAULT_INTERACTION_GRAPH_BLOCKS,
        max_hop: int = 1,
        temporal_kernel_sizes: Sequence[int] = (9, 5),
        num_heads: int = 4,
        dropout: float = 0.5,
        attention_dropout: float = 0.1,
        use_interaction_graph: bool = True,
        use_interhand_temporal_fusion: bool = True,
        use_interaction_attention: bool = True,
        interaction_graph_self_loops: bool = False,
        distance_fusion: str = "matmul",
    ) -> None:
        super().__init__()
        if len(block_channels) != len(block_strides):
            raise ValueError(
                "block_channels and block_strides must have the same length, got "
                f"{len(block_channels)} and {len(block_strides)}"
            )
        if in_channels < 3:
            raise ValueError(f"in_channels must be at least 3 (x, y, z), got {in_channels}")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_nodes = NUM_JOINTS

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

        interaction_blocks = set(interaction_graph_blocks) if use_interaction_graph else set()
        self.requires_distance = bool(interaction_blocks)

        self.input_bn = nn.BatchNorm1d(in_channels * self.num_nodes)

        blocks = []
        current_channels = in_channels
        for index, (out_channels, stride) in enumerate(zip(block_channels, block_strides)):
            blocks.append(
                STGCBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    num_subsets=self.adjacency.size(0),
                    num_nodes=self.num_nodes,
                    stride=stride,
                    temporal_kernel_sizes=temporal_kernel_sizes,
                    num_heads=num_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    use_interaction_graph=index in interaction_blocks,
                    use_interhand_temporal_fusion=use_interhand_temporal_fusion,
                    use_interaction_attention=use_interaction_attention,
                    distance_fusion=distance_fusion,
                )
            )
            current_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

        self.classifier = nn.Linear(current_channels, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Batch-normalise per joint and channel, as in ST-GCN."""
        n, c, t, v = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)
        x = self.input_bn(x)
        return x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the STGC blocks and return the ``(N, C_L)`` pooled feature."""
        if x.dim() != 4:
            raise ValueError(f"expected a 4D input (N, C, T, V), got shape {tuple(x.shape)}")
        if x.size(3) != self.num_nodes:
            raise ValueError(f"expected {self.num_nodes} joints, got {x.size(3)}")

        distance: Optional[torch.Tensor] = None
        if self.requires_distance:
            # D is built from the raw coordinates, before input normalisation, so
            # that it carries metric distances between the two hands.
            distance = pairwise_distance_matrix(x[:, :3])

        x = self._normalize_input(x)
        for block in self.blocks:
            x = block(x, self.adjacency, self.interaction_adjacency, distance)
            if distance is not None and block.stride > 1:
                distance = distance[:, :: block.stride]
        return x.mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))
