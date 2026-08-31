"""Training loop.

Implementation details of Section IV-C: 50 epochs, learning rate 0.01, SGD with
momentum 0.9 and weight decay 0.0005, cross-entropy loss, and a batch size
between 32 and 128 depending on the dataset scale. The parameters from the epoch
with the best validation metric are the ones that get exported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from .evaluator import evaluate
from .metrics import METRIC_NAMES, ClassificationMetrics

PathLike = Union[str, Path]


@dataclass
class TrainingConfig:
    """Optimisation settings, defaulting to the values reported in the paper."""

    epochs: int = 50
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0005
    nesterov: bool = False
    grad_clip: Optional[float] = None
    lr_steps: List[int] = field(default_factory=list)
    lr_gamma: float = 0.1
    best_metric: str = "f1"
    log_interval: int = 50

    def __post_init__(self) -> None:
        if self.best_metric not in METRIC_NAMES:
            raise ValueError(
                f"best_metric must be one of {METRIC_NAMES}, got {self.best_metric!r}"
            )


class Trainer:
    """Trains a model and tracks the best validation checkpoint.

    Args:
        model: Model to optimise.
        num_classes: Number of hand-washing steps.
        config: Optimisation settings.
        device: Torch device; defaults to CUDA when available.
        logger: Callable used for progress output.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
        logger=print,
    ) -> None:
        self.config = config or TrainingConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.num_classes = num_classes
        self.logger = logger

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
            nesterov=self.config.nesterov,
        )
        self.scheduler = (
            torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.config.lr_steps, gamma=self.config.lr_gamma
            )
            if self.config.lr_steps
            else None
        )

        self.best_metrics: Optional[ClassificationMetrics] = None
        self.best_epoch = -1
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.history: List[Dict[str, float]] = []

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        loss_sum = 0.0
        sample_count = 0

        for step, (skeletons, labels) in enumerate(loader):
            skeletons = skeletons.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            loss = self.criterion(self.model(skeletons), labels)
            loss.backward()
            if self.config.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            batch_size = labels.size(0)
            loss_sum += float(loss) * batch_size
            sample_count += batch_size

            if self.config.log_interval and step % self.config.log_interval == 0:
                self.logger(
                    f"epoch {epoch:3d} step {step:4d}/{len(loader)}  loss {float(loss):.4f}"
                )

        return loss_sum / max(sample_count, 1)

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: Optional[DataLoader] = None,
    ) -> Optional[ClassificationMetrics]:
        """Train for the configured number of epochs.

        Returns the best validation metrics, or ``None`` when no validation
        loader is given.
        """
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            record: Dict[str, float] = {"epoch": float(epoch), "train_loss": train_loss}

            if validation_loader is not None:
                metrics = evaluate(
                    self.model,
                    validation_loader,
                    self.num_classes,
                    device=self.device,
                    criterion=self.criterion,
                )
                record.update({f"val_{k}": v for k, v in metrics.to_dict().items()})
                self.logger(
                    f"epoch {epoch:3d}  train_loss {train_loss:.4f}  {metrics.format_summary()}"
                )
                self._update_best(metrics, epoch)
            else:
                self.logger(f"epoch {epoch:3d}  train_loss {train_loss:.4f}")

            if self.scheduler is not None:
                self.scheduler.step()
            self.history.append(record)

        return self.best_metrics

    def _update_best(self, metrics: ClassificationMetrics, epoch: int) -> None:
        current = getattr(metrics, self.config.best_metric)
        best = (
            getattr(self.best_metrics, self.config.best_metric)
            if self.best_metrics is not None
            else float("-inf")
        )
        if current > best:
            self.best_metrics = metrics
            self.best_epoch = epoch
            self.best_state_dict = {
                key: value.detach().clone().cpu()
                for key, value in self.model.state_dict().items()
            }

    def load_best(self) -> None:
        """Restore the parameters of the best epoch into the model."""
        if self.best_state_dict is None:
            raise RuntimeError("no validated epoch recorded; call fit with a validation loader")
        self.model.load_state_dict(self.best_state_dict)

    def save_checkpoint(self, path: PathLike, extra: Optional[Dict] = None) -> None:
        """Save the best parameters, or the current ones if no validation ran."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.best_state_dict or self.model.state_dict(),
            "epoch": self.best_epoch,
            "training_config": asdict(self.config),
            "metrics": self.best_metrics.to_dict() if self.best_metrics else None,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        self.logger(f"checkpoint written to {path}")
