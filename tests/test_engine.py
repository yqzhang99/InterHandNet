"""Tests for the metrics, the evaluator and the training loop."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from interhandnet.engine import Trainer, TrainingConfig, compute_metrics, confusion_matrix, evaluate
from interhandnet.graph import NUM_JOINTS
from interhandnet.models import InterHandNet

NUM_CLASSES = 3
SMALL_MODEL = {
    "num_classes": NUM_CLASSES,
    "block_channels": (8,),
    "block_strides": (1,),
    "interaction_graph_blocks": (0,),
    "temporal_kernel_sizes": (3,),
    "num_heads": 2,
}


@pytest.fixture
def loader():
    torch.manual_seed(0)
    skeletons = torch.randn(12, 3, 8, NUM_JOINTS)
    labels = torch.arange(12) % NUM_CLASSES
    return DataLoader(TensorDataset(skeletons, labels), batch_size=4)


class DualHeadModel(nn.Module):
    """Stand-in for a two-branch backbone such as STA-GCN.

    Keeping the trainer test independent of the real STA-GCN makes it clear that
    the dual supervision hinges on nothing but the presence of
    ``forward_with_auxiliary``.
    """

    def __init__(self):
        super().__init__()
        self.main = nn.Linear(3, NUM_CLASSES)
        self.auxiliary = nn.Linear(3, NUM_CLASSES)

    def forward_with_auxiliary(self, x):
        pooled = x.mean(dim=(2, 3))
        return self.main(pooled), self.auxiliary(pooled)

    def forward(self, x):
        return self.forward_with_auxiliary(x)[0]


class TestConfusionMatrix:
    def test_counts_land_in_the_right_cells(self):
        matrix = confusion_matrix([0, 1, 2, 1], [0, 1, 1, 0], num_classes=3)
        assert matrix[0, 0] == 1
        assert matrix[1, 1] == 1
        assert matrix[1, 0] == 1
        assert matrix[2, 1] == 1
        assert matrix.sum() == 4

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="same shape"):
            confusion_matrix([0, 1], [0], num_classes=2)


class TestMetrics:
    def test_perfect_prediction_scores_one(self):
        targets = [0, 1, 2, 0, 1, 2]
        metrics = compute_metrics(targets, targets, num_classes=3)
        assert metrics.accuracy == pytest.approx(1.0)
        assert metrics.precision == pytest.approx(1.0)
        assert metrics.recall == pytest.approx(1.0)
        assert metrics.f1 == pytest.approx(1.0)
        assert metrics.num_samples == 6

    def test_macro_average_matches_the_manual_computation(self):
        # Class 0: TP=2, FP=1, FN=0 -> P=2/3, R=1
        # Class 1: TP=1, FP=0, FN=1 -> P=1,   R=1/2
        targets = [0, 0, 1, 1]
        predictions = [0, 0, 1, 0]
        metrics = compute_metrics(targets, predictions, num_classes=2)
        assert metrics.accuracy == pytest.approx(0.75)
        assert metrics.precision == pytest.approx((2 / 3 + 1.0) / 2)
        assert metrics.recall == pytest.approx((1.0 + 0.5) / 2)
        f1_per_class = [2 * (2 / 3) * 1 / (2 / 3 + 1), 2 * 1 * 0.5 / (1 + 0.5)]
        assert metrics.f1 == pytest.approx(np.mean(f1_per_class))

    def test_absent_class_scores_zero_instead_of_nan(self):
        metrics = compute_metrics([0, 0], [0, 0], num_classes=3)
        assert np.isfinite(metrics.f1)
        assert metrics.per_class_f1[2] == 0.0

    def test_summary_mentions_every_metric(self):
        summary = compute_metrics([0, 1], [0, 1], num_classes=2).format_summary()
        for name in ("accuracy", "precision", "recall", "f1"):
            assert name in summary


class TestEvaluator:
    def test_returns_metrics_over_the_loader(self, loader):
        model = InterHandNet(**SMALL_MODEL).eval()
        metrics = evaluate(model, loader, NUM_CLASSES, device=torch.device("cpu"))
        assert metrics.num_samples == 12
        assert 0.0 <= metrics.accuracy <= 1.0

    def test_loss_is_reported_when_a_criterion_is_given(self, loader):
        model = InterHandNet(**SMALL_MODEL).eval()
        metrics = evaluate(
            model,
            loader,
            NUM_CLASSES,
            device=torch.device("cpu"),
            criterion=torch.nn.CrossEntropyLoss(),
        )
        assert metrics.loss is not None and metrics.loss > 0

    def test_training_mode_is_restored(self, loader):
        model = InterHandNet(**SMALL_MODEL).train()
        evaluate(model, loader, NUM_CLASSES, device=torch.device("cpu"))
        assert model.training


class TestTrainingConfig:
    def test_paper_defaults(self):
        config = TrainingConfig()
        assert config.epochs == 50
        assert config.learning_rate == pytest.approx(0.01)
        assert config.momentum == pytest.approx(0.9)
        assert config.weight_decay == pytest.approx(0.0005)

    def test_unknown_best_metric_is_rejected(self):
        with pytest.raises(ValueError, match="best_metric"):
            TrainingConfig(best_metric="auc")

    def test_negative_auxiliary_weight_is_rejected(self):
        with pytest.raises(ValueError, match="auxiliary_loss_weight"):
            TrainingConfig(auxiliary_loss_weight=-1.0)


class TestTrainer:
    def test_fit_records_history_and_best_checkpoint(self, loader, tmp_path):
        trainer = Trainer(
            InterHandNet(**SMALL_MODEL),
            num_classes=NUM_CLASSES,
            config=TrainingConfig(epochs=2, log_interval=0),
            device=torch.device("cpu"),
            logger=lambda *_: None,
        )
        metrics = trainer.fit(loader, loader)

        assert metrics is not None
        assert len(trainer.history) == 2
        assert trainer.best_epoch in (1, 2)

        checkpoint = tmp_path / "best.pt"
        trainer.save_checkpoint(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu")
        assert "state_dict" in payload
        assert payload["training_config"]["epochs"] == 2

    def test_training_updates_the_parameters(self, loader):
        model = InterHandNet(**SMALL_MODEL)
        before = model.classifier.weight.detach().clone()
        trainer = Trainer(
            model,
            num_classes=NUM_CLASSES,
            config=TrainingConfig(epochs=1, log_interval=0),
            device=torch.device("cpu"),
            logger=lambda *_: None,
        )
        trainer.fit(loader)
        assert not torch.allclose(before, model.classifier.weight)

    def test_single_head_model_does_not_use_the_auxiliary_path(self):
        trainer = Trainer(
            InterHandNet(**SMALL_MODEL),
            num_classes=NUM_CLASSES,
            device=torch.device("cpu"),
            logger=lambda *_: None,
        )
        assert not trainer.uses_auxiliary_head

    def test_dual_head_loss_sums_both_branches(self):
        """STA-GCN supervises both branches, so the loss must contain both terms."""
        torch.manual_seed(0)
        model = DualHeadModel()
        trainer = Trainer(
            model,
            num_classes=NUM_CLASSES,
            config=TrainingConfig(auxiliary_loss_weight=0.5, log_interval=0),
            device=torch.device("cpu"),
            logger=lambda *_: None,
        )
        assert trainer.uses_auxiliary_head

        skeletons = torch.randn(4, 3, 8, NUM_JOINTS)
        labels = torch.arange(4) % NUM_CLASSES
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            main_logits, auxiliary_logits = model.forward_with_auxiliary(skeletons)
            expected = criterion(main_logits, labels) + 0.5 * criterion(
                auxiliary_logits, labels
            )
            loss = trainer.compute_loss(skeletons, labels)
        assert float(loss) == pytest.approx(float(expected))

    def test_zero_auxiliary_weight_falls_back_to_the_prediction_head(self):
        torch.manual_seed(0)
        model = DualHeadModel()
        trainer = Trainer(
            model,
            num_classes=NUM_CLASSES,
            config=TrainingConfig(auxiliary_loss_weight=0.0, log_interval=0),
            device=torch.device("cpu"),
            logger=lambda *_: None,
        )
        assert not trainer.uses_auxiliary_head

        skeletons = torch.randn(4, 3, 8, NUM_JOINTS)
        labels = torch.arange(4) % NUM_CLASSES
        with torch.no_grad():
            expected = torch.nn.CrossEntropyLoss()(model(skeletons), labels)
            loss = trainer.compute_loss(skeletons, labels)
        assert float(loss) == pytest.approx(float(expected))

    def test_dual_head_training_updates_both_branches(self, loader):
        torch.manual_seed(0)
        model = DualHeadModel()
        before = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
        }
        trainer = Trainer(
            model,
            num_classes=NUM_CLASSES,
            config=TrainingConfig(epochs=1, log_interval=0),
            device=torch.device("cpu"),
            logger=lambda *_: None,
        )
        trainer.fit(loader)
        for name, parameter in model.named_parameters():
            assert not torch.allclose(before[name], parameter), f"{name} did not move"

    def test_load_best_requires_validation(self, loader):
        trainer = Trainer(
            InterHandNet(**SMALL_MODEL),
            num_classes=NUM_CLASSES,
            config=TrainingConfig(epochs=1, log_interval=0),
            device=torch.device("cpu"),
            logger=lambda *_: None,
        )
        trainer.fit(loader)
        with pytest.raises(RuntimeError, match="no validated epoch"):
            trainer.load_best()
