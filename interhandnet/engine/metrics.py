"""Classification metrics used in the evaluation (Section IV-B).

The paper reports accuracy, precision, recall and F1 score. Precision, recall
and F1 are macro-averaged, i.e. computed per hand-washing step and then averaged
without class-frequency weighting, which is why they sit below the accuracy for
the weaker baselines in Table II.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

METRIC_NAMES = ("accuracy", "precision", "recall", "f1")


@dataclass
class ClassificationMetrics:
    """Aggregated classification scores plus the confusion matrix."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    num_samples: int
    per_class_precision: np.ndarray = field(repr=False)
    per_class_recall: np.ndarray = field(repr=False)
    per_class_f1: np.ndarray = field(repr=False)
    confusion_matrix: np.ndarray = field(repr=False)
    loss: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        values = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "num_samples": float(self.num_samples),
        }
        if self.loss is not None:
            values["loss"] = self.loss
        return values

    def format_summary(self) -> str:
        summary = (
            f"accuracy {self.accuracy:.4f}  precision {self.precision:.4f}  "
            f"recall {self.recall:.4f}  f1 {self.f1:.4f}"
        )
        if self.loss is not None:
            summary = f"loss {self.loss:.4f}  " + summary
        return f"{summary}  (n={self.num_samples})"


def confusion_matrix(
    targets: Sequence[int], predictions: Sequence[int], num_classes: int
) -> np.ndarray:
    """Rows are ground-truth classes, columns are predicted classes."""
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if targets.shape != predictions.shape:
        raise ValueError(
            f"targets and predictions must have the same shape, got "
            f"{targets.shape} and {predictions.shape}"
        )
    for name, values in (("targets", targets), ("predictions", predictions)):
        if values.size and (values.min() < 0 or values.max() >= num_classes):
            raise ValueError(
                f"{name} must lie in [0, {num_classes - 1}], got "
                f"[{values.min()}, {values.max()}]"
            )

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    return matrix


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=np.float64)
    positive = denominator > 0
    result[positive] = numerator[positive] / denominator[positive]
    return result


def compute_metrics(
    targets: Sequence[int], predictions: Sequence[int], num_classes: int
) -> ClassificationMetrics:
    """Compute accuracy and macro-averaged precision, recall and F1."""
    matrix = confusion_matrix(targets, predictions, num_classes)
    true_positive = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)
    total = float(matrix.sum())

    precision = _safe_divide(true_positive, predicted)
    recall = _safe_divide(true_positive, actual)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return ClassificationMetrics(
        accuracy=float(true_positive.sum() / total) if total else 0.0,
        precision=float(precision.mean()),
        recall=float(recall.mean()),
        f1=float(f1.mean()),
        num_samples=int(total),
        per_class_precision=precision,
        per_class_recall=recall,
        per_class_f1=f1,
        confusion_matrix=matrix,
    )
