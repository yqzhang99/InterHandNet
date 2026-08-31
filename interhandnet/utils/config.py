"""YAML configuration loading with command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Union

import yaml

PathLike = Union[str, Path]


def load_config(path: PathLike) -> Dict[str, Any]:
    """Load a YAML config, resolving an optional ``_base_`` inheritance key.

    ``_base_`` may be a single path or a list of paths, relative to the config
    file itself. Later entries and the config's own keys take precedence.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError(f"config root must be a mapping, got {type(config).__name__}")

    bases = config.pop("_base_", None)
    if bases is None:
        return config

    if isinstance(bases, (str, Path)):
        bases = [bases]
    merged: Dict[str, Any] = {}
    for base in bases:
        merged = merge_dicts(merged, load_config(path.parent / base))
    return merge_dicts(merged, config)


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply ``dotted.key=value`` overrides parsed as YAML scalars.

    Example: ``model.use_interaction_attention=false training.epochs=10``
    """
    result = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must look like key=value, got {override!r}")
        dotted_key, raw_value = override.split("=", 1)
        keys = dotted_key.strip().split(".")
        target = result
        for key in keys[:-1]:
            existing = target.get(key)
            if not isinstance(existing, dict):
                existing = {}
                target[key] = existing
            target = existing
        target[keys[-1]] = yaml.safe_load(raw_value)
    return result


def save_config(config: Dict[str, Any], path: PathLike) -> None:
    """Write a resolved config next to the run's artefacts for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
