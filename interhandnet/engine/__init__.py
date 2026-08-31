"""Training and evaluation engine."""

from .evaluator import evaluate
from .metrics import METRIC_NAMES, ClassificationMetrics, compute_metrics, confusion_matrix
from .trainer import Trainer, TrainingConfig

__all__ = [
    "METRIC_NAMES",
    "ClassificationMetrics",
    "Trainer",
    "TrainingConfig",
    "compute_metrics",
    "confusion_matrix",
    "evaluate",
]
