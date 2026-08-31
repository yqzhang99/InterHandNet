"""Tests for the metrics, the evaluator and the training loop."""

import numpy as np
import pytest
import torch
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
