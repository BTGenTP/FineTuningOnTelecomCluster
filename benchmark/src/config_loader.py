from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import ExperimentConfig, GenerationConfig, ModelConfig, PeftConfig, PromptConfig, TrainingConfig


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Invalid config at {config_path}")
    return ExperimentConfig(
        name=str(raw["name"]),
        task=str(raw["task"]),
        output_root=str(raw["output_root"]),
        method=str(raw["method"]),
        model=ModelConfig(**dict(raw.get("model", {}))),
        peft=PeftConfig(**dict(raw.get("peft", {}))),
        prompt=PromptConfig(**dict(raw.get("prompt", {}))),
        training=TrainingConfig(**dict(raw.get("training", {}))),
        generation=GenerationConfig(**dict(raw.get("generation", {}))),
        catalog_path=raw.get("catalog_path"),
        xsd_path=raw.get("xsd_path"),
        metadata=dict(raw.get("metadata", {})),
    )
