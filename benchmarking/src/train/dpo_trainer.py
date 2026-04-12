"""
DPO Trainer wrapper for NAV4RAIL benchmarking.
===============================================
Wraps trl.DPOTrainer for preference optimization.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class DPOTrainerWrapper:
    """Wraps trl.DPOTrainer for NAV4RAIL DPO experiments."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def train(self) -> dict[str, Any]:
        from datasets import Dataset
        from transformers import TrainingArguments
        from trl import DPOConfig, DPOTrainer

        from src.utils.config import get_active_model_config
        from src.utils.model_loader import load_for_training

        model_config = get_active_model_config(self.cfg)
        dpo_cfg = self.cfg.get("dpo", {})
        train_cfg = self.cfg.get("training", {})

        model, tokenizer = load_for_training(self.cfg)

        # Load preference dataset
        data_path = self.cfg["data"].get("preference_dataset", "data/dataset_dpo.jsonl")
        examples = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        ds = Dataset.from_list(examples)
        split = ds.train_test_split(test_size=0.05, seed=self.cfg["experiment"]["seed"])

        model_key = self.cfg.get("model", {}).get("key", "unknown")
        output_dir = f"runs/dpo_{model_key}"

        training_args = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=train_cfg.get("epochs", 3),
            per_device_train_batch_size=train_cfg.get("batch_size", 1),
            gradient_accumulation_steps=train_cfg.get("grad_accum", 8),
            learning_rate=train_cfg.get("lr", 5e-6),
            beta=dpo_cfg.get("beta", 0.1),
            loss_type=dpo_cfg.get("loss_type", "sigmoid"),
            bf16=model_config.get("bf16", True),
            gradient_checkpointing=True,
            report_to=train_cfg.get("report_to", "wandb"),
            seed=self.cfg["experiment"]["seed"],
        )

        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            processing_class=tokenizer,
        )

        logger.info("Starting DPO training...")
        result = trainer.train()
        trainer.save_model(f"{output_dir}/final_adapter")
        return result
