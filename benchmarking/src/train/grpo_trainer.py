"""
GRPO Trainer wrapper for NAV4RAIL benchmarking.
================================================
Wraps trl.GRPOTrainer with validate_bt-based reward function.
GRPO is ideal for NAV4RAIL: no reward model needed, validate_bt is the verifier.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class GRPOTrainerWrapper:
    """Wraps trl.GRPOTrainer for NAV4RAIL GRPO experiments."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def train(self) -> dict[str, Any]:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer

        from src.data.skills_loader import SkillsCatalog
        from src.reward.reward_fn import make_reward_fn
        from src.utils.config import get_active_model_config, resolve_paths
        from src.utils.model_loader import load_for_training

        cfg = resolve_paths(self.cfg)
        model_config = get_active_model_config(cfg)
        grpo_cfg = cfg.get("grpo", {})

        model, tokenizer = load_for_training(cfg)

        # Load prompts (missions only, no labels needed for GRPO)
        data_path = cfg["data"]["train_dataset"]
        prompts = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ex = json.loads(line)
                    prompts.append({"prompt": ex.get("mission", ex.get("input", ""))})

        ds = Dataset.from_list(prompts)

        # Build reward function
        catalog = SkillsCatalog(cfg["data"]["catalog"])
        reward_fn = make_reward_fn(catalog, cfg)

        model_key = cfg.get("model", {}).get("key", "unknown")
        output_dir = f"runs/grpo_{model_key}"

        training_args = GRPOConfig(
            output_dir=output_dir,
            num_generations=grpo_cfg.get("num_generations", 8),
            max_completion_length=grpo_cfg.get("max_completion_length", 4096),
            num_train_epochs=grpo_cfg.get("num_train_epochs", 3),
            per_device_train_batch_size=grpo_cfg.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=grpo_cfg.get("gradient_accumulation_steps", 8),
            learning_rate=grpo_cfg.get("learning_rate", 5e-6),
            bf16=model_config.get("bf16", True),
            gradient_checkpointing=True,
            report_to=cfg.get("training", {}).get("report_to", "wandb"),
            seed=cfg["experiment"]["seed"],
        )

        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=ds,
            processing_class=tokenizer,
            reward_funcs=reward_fn,
        )

        logger.info("Starting GRPO training (reward = validate_bt)...")
        result = trainer.train()
        trainer.save_model(f"{output_dir}/final_adapter")
        return result
