"""High-level entry points shared by the command-line tools.

These helpers turn a resolved configuration dictionary into datasets, loaders
and a trained model, so that the scripts in ``tools/`` stay thin.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from .data import HandWashingSkeletonDataset
from .engine import ClassificationMetrics, Trainer, TrainingConfig
from .models import build_model


def build_dataset(data_config: Dict[str, Any]) -> HandWashingSkeletonDataset:
    """Instantiate the skeleton dataset described by the ``data`` section."""
    return HandWashingSkeletonDataset(
        archive=data_config["archive"],
        window_size=data_config.get("window_size", 30),
        center=data_config.get("center", False),
    )


def build_loader(
    dataset: HandWashingSkeletonDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def build_fold_loaders(
    dataset: HandWashingSkeletonDataset,
    data_config: Dict[str, Any],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
) -> Tuple[DataLoader, DataLoader]:
    """Build the training and validation loaders of one cross-validation fold."""
    batch_size = data_config.get("batch_size", 64)
    num_workers = data_config.get("num_workers", 4)
    return (
        build_loader(dataset.subset(train_indices), batch_size, True, num_workers),
        build_loader(dataset.subset(validation_indices), batch_size, False, num_workers),
    )


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_fold(
    config: Dict[str, Any],
    dataset: HandWashingSkeletonDataset,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    device: Optional[torch.device] = None,
    logger=print,
) -> Tuple[Trainer, Optional[ClassificationMetrics]]:
    """Train one fold and return the trainer together with its best metrics."""
    model = build_model(config["model"])
    training_config = TrainingConfig(**config.get("training", {}))
    trainer = Trainer(
        model=model,
        num_classes=config["model"]["num_classes"],
        config=training_config,
        device=device or resolve_device(),
        logger=logger,
    )

    train_loader, validation_loader = build_fold_loaders(
        dataset, config["data"], train_indices, validation_indices
    )
    metrics = trainer.fit(train_loader, validation_loader)
    return trainer, metrics


def load_checkpoint(
    path: str, config: Dict[str, Any], device: Optional[torch.device] = None
) -> torch.nn.Module:
    """Build the configured model and load parameters from a checkpoint."""
    device = device or resolve_device()
    model = build_model(config["model"]).to(device)
    payload = torch.load(path, map_location=device)
    state_dict = payload.get("state_dict", payload)
    model.load_state_dict(state_dict)
    model.eval()
    return model
