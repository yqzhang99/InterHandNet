"""InterHandNet: Capturing Two-hand Interaction for Robust Hand-washing Activity Recognition.

Reference implementation of the PerCom 2025 paper by Yiqing Zhang and Takuya
Maekawa. The package is organised around the three modules the paper proposes:

* :class:`~interhandnet.modules.interaction_graph.SpatialInteractionGraphConv` -- Interaction Graph, Eq. (2)
* :class:`~interhandnet.modules.interhand_temporal_fusion.InterHandTemporalFusion` -- InterHand Temporal Fusion, Eq. (4)
* :class:`~interhandnet.modules.interaction_attention.InteractionAttention` -- Interaction Attention, Eq. (6) and Eq. (7)

All three plug into an STGCN-based backbone, which is what Section III-F calls
the strong compatibility of the approach.
"""

from .graph import NUM_JOINTS, NUM_JOINTS_PER_HAND, HandGraph, InteractionGraph
from .models import InterHandNet, build_model
from .modules import (
    FeatureExtractor,
    InterHandTemporalFusion,
    InteractionAttention,
    SpatialGraphConv,
    SpatialInteractionGraphConv,
)

__version__ = "1.0.0"

__all__ = [
    "NUM_JOINTS",
    "NUM_JOINTS_PER_HAND",
    "FeatureExtractor",
    "HandGraph",
    "InterHandNet",
    "InterHandTemporalFusion",
    "InteractionAttention",
    "InteractionGraph",
    "SpatialGraphConv",
    "SpatialInteractionGraphConv",
    "__version__",
    "build_model",
]
