"""Model construction from a configuration dictionary."""

from __future__ import annotations

import inspect
from typing import Any, Dict

from torch import nn

from .interhandnet import InterHandNet
from .sta_gcn import STAGCN

_MODEL_REGISTRY = {
    # InterHandNet with an ST-GCN backbone.
    "interhandnet": InterHandNet,
    # InterHandNet with an STA-GCN backbone, the best configuration of Table II.
    "interhandnet_sta_gcn": STAGCN,
}


def available_models() -> list:
    return sorted(_MODEL_REGISTRY)


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Instantiate a model from a ``model`` configuration section.

    The ``name`` key selects the class; every other key is forwarded as a keyword
    argument. Baselines and ablations are expressed by toggling the
    ``use_interaction_graph`` / ``use_interhand_temporal_fusion`` /
    ``use_interaction_attention`` flags.

    Keys the selected backbone does not accept are reported by name, since the
    two backbones describe their block layout differently and a config inherited
    from the wrong parent is otherwise hard to diagnose.
    """
    config = dict(config)
    name = config.pop("name", "interhandnet")
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {available_models()}")

    model_class = _MODEL_REGISTRY[name]
    accepted = set(inspect.signature(model_class).parameters)
    unexpected = sorted(set(config) - accepted)
    if unexpected:
        raise KeyError(
            f"model {name!r} does not accept the config key(s) {unexpected}; "
            f"it accepts {sorted(accepted)}"
        )
    return model_class(**config)
