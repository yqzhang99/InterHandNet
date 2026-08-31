"""Tests for configuration loading and the shipped experiment configs."""

from pathlib import Path

import pytest
import torch

from interhandnet.graph import NUM_JOINTS
from interhandnet.models import build_model
from interhandnet.utils import apply_overrides, load_config, merge_dicts, save_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
EXPERIMENT_CONFIGS = sorted(path for path in CONFIG_DIR.glob("*.yaml") if path.name != "base.yaml")

# Per-backbone overrides that shrink a shipped config to a test-sized network.
SHRINK = {
    "interhandnet": {
        "block_channels": [8, 8],
        "block_strides": [1, 2],
        "interaction_graph_blocks": [0],
        "temporal_kernel_sizes": [3],
        "num_heads": 2,
    },
    "interhandnet_sta_gcn": {
        "feature_channels": [8, 8],
        "feature_strides": [1, 1],
        "branch_channels": [16],
        "branch_strides": [2],
        "interaction_graph_blocks": [1],
        "temporal_kernel_sizes": [3],
        "num_heads": 2,
    },
}


class TestMerge:
    def test_nested_dicts_are_merged_recursively(self):
        base = {"model": {"num_classes": 6, "num_heads": 4}, "seed": 1}
        override = {"model": {"num_heads": 8}}
        merged = merge_dicts(base, override)
        assert merged == {"model": {"num_classes": 6, "num_heads": 8}, "seed": 1}

    def test_inputs_are_not_mutated(self):
        base = {"model": {"num_heads": 4}}
        merge_dicts(base, {"model": {"num_heads": 8}})
        assert base["model"]["num_heads"] == 4


class TestOverrides:
    def test_dotted_key_is_applied(self):
        config = apply_overrides({"training": {"epochs": 50}}, ["training.epochs=3"])
        assert config["training"]["epochs"] == 3

    def test_values_are_parsed_as_yaml_scalars(self):
        config = apply_overrides(
            {"model": {}}, ["model.use_interaction_attention=false", "model.num_heads=8"]
        )
        assert config["model"]["use_interaction_attention"] is False
        assert config["model"]["num_heads"] == 8

    def test_lists_can_be_overridden(self):
        config = apply_overrides({"model": {}}, ["model.block_strides=[1, 2, 1]"])
        assert config["model"]["block_strides"] == [1, 2, 1]

    def test_malformed_override_is_rejected(self):
        with pytest.raises(ValueError, match="key=value"):
            apply_overrides({}, ["training.epochs"])


class TestLoadConfig:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "absent.yaml")

    def test_base_inheritance(self, tmp_path):
        (tmp_path / "base.yaml").write_text("a: 1\nnested:\n  x: 1\n  y: 2\n", encoding="utf-8")
        (tmp_path / "child.yaml").write_text(
            "_base_: base.yaml\nnested:\n  y: 3\n", encoding="utf-8"
        )
        config = load_config(tmp_path / "child.yaml")
        assert config == {"a": 1, "nested": {"x": 1, "y": 3}}

    def test_round_trip_through_save(self, tmp_path):
        config = {"model": {"num_classes": 6}, "seed": 1}
        save_config(config, tmp_path / "out.yaml")
        assert load_config(tmp_path / "out.yaml") == config


class TestShippedConfigs:
    def test_configs_exist(self):
        assert EXPERIMENT_CONFIGS, f"no experiment configs found in {CONFIG_DIR}"

    @pytest.mark.parametrize("path", EXPERIMENT_CONFIGS, ids=lambda p: p.stem)
    def test_config_has_the_expected_sections(self, path):
        config = load_config(path)
        for section in ("data", "model", "training", "cross_validation"):
            assert section in config, f"{path.name} is missing the {section!r} section"
        # 7 classes: "other" plus the six WHO steps, as labelled in the dataset.
        assert config["model"]["num_classes"] == 7
        assert config["data"]["window_size"] == 30

    @pytest.mark.parametrize("path", EXPERIMENT_CONFIGS, ids=lambda p: p.stem)
    def test_model_builds_and_runs(self, path):
        config = load_config(path)
        # Shrink the network so the forward pass stays cheap in tests.
        config["model"].update(SHRINK[config["model"].get("name", "interhandnet")])
        model = build_model(config["model"]).eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 3, config["data"]["window_size"], NUM_JOINTS))
        assert logits.shape == (1, config["model"]["num_classes"])

    def test_paper_hyperparameters_in_base(self):
        config = load_config(CONFIG_DIR / "base.yaml")
        training = config["training"]
        assert training["epochs"] == 50
        assert training["learning_rate"] == pytest.approx(0.01)
        assert training["momentum"] == pytest.approx(0.9)
        assert training["weight_decay"] == pytest.approx(0.0005)
        assert config["cross_validation"]["num_folds"] == 5

    def test_experiment_configs_are_distinct(self):
        """No two shipped configs may describe the same experiment. The backbone
        is part of the identity, since the same module flags are used with both."""
        signatures = {}
        for path in EXPERIMENT_CONFIGS:
            model = load_config(path)["model"]
            signatures[path.stem] = (
                model.get("name", "interhandnet"),
                model["use_interaction_graph"],
                model["use_interaction_attention"],
                model["use_interhand_temporal_fusion"],
                model.get("residual", True),
                model.get("interaction_graph_self_loops", False),
            )
        duplicates = len(signatures) - len(set(signatures.values()))
        assert not duplicates, f"duplicate experiment configs: {signatures}"

    def test_every_config_names_a_known_model(self):
        for path in EXPERIMENT_CONFIGS:
            name = load_config(path)["model"].get("name", "interhandnet")
            assert name in SHRINK, f"{path.name} uses unknown model {name!r}"
