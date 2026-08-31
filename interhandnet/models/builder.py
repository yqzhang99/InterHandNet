"""Model construction from a configuration dictionary."""

from __future__ import annotations

from typing import Any, Dict

from torch import nn

from .interhandnet import InterHandNet

_MODEL_REGISTRY = {
    "interhandnet": InterHandNet,
}


def available_models() -> list:
    return sorted(_MODEL_REGISTRY)


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Instantiate a model from a ``model`` configuration section.

    The ``name`` key selects the class; every other key is forwarded as a keyword
    argument. Baselines and ablations are expressed by toggling the
    ``use_interaction_graph`` / ``use_interhand_temporal_fusion`` /
    ``use_interaction_attention`` flags.
    """
    config = dict(config)
    name = config.pop("name", "interhandnet")
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {available_models()}")
    return _MODEL_REGISTRY[name](**config)
