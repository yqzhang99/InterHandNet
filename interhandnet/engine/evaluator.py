"""Model evaluation."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import ClassificationMetrics, compute_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: Optional[torch.device] = None,
    criterion: Optional[nn.Module] = None,
) -> ClassificationMetrics:
    """Run the model over ``loader`` and return the classification metrics.

    When ``criterion`` is given, the mean loss is attached to the returned
    metrics as the ``loss`` attribute.
    """
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    all_targets: List[int] = []
    all_predictions: List[int] = []
    loss_sum = 0.0
    loss_count = 0

    for skeletons, labels in loader:
        skeletons = skeletons.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(skeletons)
        if criterion is not None:
            loss_sum += float(criterion(logits, labels)) * labels.size(0)
            loss_count += labels.size(0)
        all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    model.train(was_training)

    metrics = compute_metrics(all_targets, all_predictions, num_classes)
    if criterion is not None and loss_count:
        metrics.loss = loss_sum / loss_count
    return metrics
