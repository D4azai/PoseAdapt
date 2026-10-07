"""Small YAML configuration loader with explicit base-file inheritance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML, optionally merging a sibling file named by the `base` key."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Top-level YAML value must be a mapping: {config_path}")

    base_reference = loaded.pop("base", None)
    if base_reference is None:
        return loaded
    base_path = (config_path.parent / str(base_reference)).resolve()
    if base_path == config_path:
        raise ValueError("A configuration file cannot inherit from itself")
    return deep_merge(load_config(base_path), loaded)
