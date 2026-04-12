"""
Configuration loader with YAML merge support.
==============================================
Loads base.yaml and merges with model/method overrides.

Usage:
    from src.utils.config import load_config
    cfg = load_config("configs/base.yaml", model="gemma2_9b", method="grpo")
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_config(
    config_path: str | Path = "configs/base.yaml",
    model: str | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    """
    Load and merge YAML configuration.

    1. Load base config from config_path
    2. If model specified, merge configs/models/{model}.yaml (if exists)
       and set model.key
    3. If method specified, merge configs/methods/{method}.yaml (if exists)
       and set training.method

    Args:
        config_path: Path to base YAML config
        model: Model key override (e.g. "gemma2_9b")
        method: Training method override (e.g. "grpo")

    Returns:
        Merged configuration dict
    """
    config_path = Path(config_path)
    base_dir = config_path.parent

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if model:
        # Try to load model-specific override
        model_cfg_path = base_dir / "models" / f"{model}.yaml"
        if model_cfg_path.exists():
            with open(model_cfg_path, encoding="utf-8") as f:
                model_cfg = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, model_cfg)
        # Set the active model key
        cfg.setdefault("model", {})["key"] = model

    if method:
        method_cfg_path = base_dir / "methods" / f"{method}.yaml"
        if method_cfg_path.exists():
            with open(method_cfg_path, encoding="utf-8") as f:
                method_cfg = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, method_cfg)
        cfg.setdefault("training", {})["method"] = method

    return cfg


def get_active_model_config(cfg: dict) -> dict[str, Any]:
    """Extract the active model's config from the models registry."""
    model_key = cfg.get("model", {}).get("key", "mistral_7b")
    return cfg.get("models", {}).get(model_key, {})


def resolve_paths(cfg: dict, base_dir: str | Path = ".") -> dict:
    """Resolve relative data paths to absolute paths."""
    base = Path(base_dir)
    data = cfg.get("data", {})
    for key in ("catalog", "safety_rules", "train_dataset", "test_missions"):
        if key in data and not Path(data[key]).is_absolute():
            data[key] = str(base / data[key])
    return cfg
