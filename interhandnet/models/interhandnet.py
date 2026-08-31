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

from .base import TwoHandBackbone, initialize_weights
from .stgc_block import STGCBlock

DEFAULT_BLOCK_CHANNELS = (32, 32, 32, 64, 64, 64)
DEFAULT_BLOCK_STRIDES = (1, 1, 1, 2, 1, 1)
DEFAULT_INTERACTION_GRAPH_BLOCKS = (0, 1, 2)


class InterHandNet(TwoHandBackbone):
    """InterHandNet with an ST-GCN backbone.

    Args:
        num_classes: Number of output classes. The default of 7 matches the label
            set of the Lulla et al. dataset used in the paper: class 0 is "other
            movement" and classes 1..6 are the six WHO steps. Set it to 6 to
            classify the WHO steps alone.
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
        attention_scope: Scope of Interaction Attention, see
            :class:`~interhandnet.modules.interaction_attention.InteractionAttention`.
        residual: Add a block-level residual connection to every STGC block.

    Shape:
        - Input: ``(N, C_in, T, V)`` with ``V = 42`` joints, left hand first.
        - Output: ``(N, num_classes)`` class logits.
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 3,
        block_channels: Sequence[int] = DEFAULT_BLOCK_CHANNELS,
        block_strides: Sequence[int] = DEFAULT_BLOCK_STRIDES,
        interaction_graph_blocks: Sequence[int] = DEFAULT_INTERACTION_GRAPH_BLOCKS,
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
        if len(block_channels) != len(block_strides):
            raise ValueError(
                "block_channels and block_strides must have the same length, got "
                f"{len(block_channels)} and {len(block_strides)}"
            )
        interaction_blocks = set(interaction_graph_blocks) if use_interaction_graph else set()
        super().__init__(
            in_channels=in_channels,
            max_hop=max_hop,
            interaction_graph_self_loops=interaction_graph_self_loops,
            requires_distance=bool(interaction_blocks),
        )

        self.num_classes = num_classes

        blocks = []
        current_channels = in_channels
        for index, (out_channels, stride) in enumerate(zip(block_channels, block_strides)):
            blocks.append(
                STGCBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    num_subsets=self.num_subsets,
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
                    attention_scope=attention_scope,
                    residual=residual,
                )
            )
            current_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

        self.classifier = nn.Linear(current_channels, num_classes)
        initialize_weights(self)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the STGC blocks and return the ``(N, C_L)`` pooled feature."""
        self._check_input(x)
        distance: Optional[torch.Tensor] = self._distance_matrix(x)
        x = self._normalize_input(x)
        for block in self.blocks:
            x = block(x, self.adjacency, self.interaction_adjacency, distance)
            if distance is not None and block.stride > 1:
                distance = distance[:, :: block.stride]
        return x.mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))
