"""Feature Extractor of Fig. 5.

The paper describes it as "a combination of fully connected (FC) layers, ReLU,
and dropout layers to extract the feature (Feature Extractor), which is then
element-wise added to the original hand feature". It is applied twice inside
every STGC block: once after InterHand Temporal Fusion and once after
Interaction Attention.

The fully connected layers are realised as ``1 x 1`` convolutions so that the
same weights are shared across time steps and joints, which is how FC layers are
implemented in ST-GCN style architectures.
"""

from __future__ import annotations

import torch
from torch import nn


class FeatureExtractor(nn.Module):
    """``Dropout(FC2(ReLU(FC1(x))))`` plus an element-wise residual connection.

    Args:
        channels: Number of input and output channels.
        dropout: Dropout probability applied after the second FC layer.
    """

    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.fc2(self.relu(self.fc1(x))))
        return out + x
