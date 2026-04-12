"""
SFT Trainer wrapper for NAV4RAIL benchmarking.
===============================================
Wraps trl.SFTTrainer with NAV4RAIL-specific dataset loading and formatting.
Reuses patterns from finetune/finetune_llama3_nav4rail.py.

Usage:
    from src.train.sft_trainer import SFTTrainerWrapper
    trainer = SFTTrainerWrapper(cfg)
    trainer.train()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_dataset(cfg: dict, tokenizer) -> tuple:
    """Load and format SFT dataset from JSONL."""
    from datasets import Dataset

    from src.data.prompt_builder import build_sft_example
    from src.utils.config import get_active_model_config

    model_config = get_active_model_config(cfg)
    data_path = cfg["data"]["train_dataset"]
    test_size = cfg["data"].get("test_size", 0.05)
    seed = cfg["experiment"]["seed"]

    # Load JSONL
    examples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    # Format as chat messages or text
    formatted = []
    for ex in examples:
        mission = ex.get("mission", ex.get("input", ""))
        xml = ex.get("xml", ex.get("output", ""))
        messages = build_sft_example(mission, xml, model_config)

        if isinstance(messages, list):
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            text = messages

        formatted.append({"text": text})

    ds = Dataset.from_list(formatted)
    split = ds.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


class SFTTrainerWrapper:
    """Wraps trl.SFTTrainer for NAV4RAIL SFT experiments."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def train(self) -> dict[str, Any]:
        from transformers import TrainingArguments
        from trl import SFTTrainer

        from src.utils.config import get_active_model_config
        from src.utils.model_loader import load_for_training

        model_config = get_active_model_config(self.cfg)
        train_cfg = self.cfg.get("training", {})

        # Load model
        self.model, self.tokenizer = load_for_training(self.cfg)

        # Load dataset
        train_ds, eval_ds = _load_dataset(self.cfg, self.tokenizer)
        logger.info(f"Dataset: {len(train_ds)} train, {len(eval_ds)} eval")

        # Training arguments
        epochs = train_cfg.get("epochs", model_config.get("default_epochs", 10))
        batch_size = train_cfg.get("batch_size", model_config.get("default_batch", 1))
        grad_accum = train_cfg.get("grad_accum", model_config.get("default_grad_accum", 16))

        model_key = self.cfg.get("model", {}).get("key", "unknown")
        output_dir = f"runs/sft_{model_key}"

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=train_cfg.get("lr", 2e-4),
            lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
            warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
            weight_decay=train_cfg.get("weight_decay", 0.01),
            max_grad_norm=train_cfg.get("max_grad_norm", 0.3),
            gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
            optim=train_cfg.get("optim", "paged_adamw_8bit"),
            eval_strategy=train_cfg.get("eval_strategy", "epoch"),
            save_strategy=train_cfg.get("save_strategy", "epoch"),
            load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
            logging_steps=10,
            bf16=model_config.get("bf16", True),
            report_to=train_cfg.get("report_to", "wandb"),
            seed=self.cfg["experiment"]["seed"],
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            max_seq_length=train_cfg.get("max_seq_len", 8192),
            dataset_text_field="text",
        )

        logger.info("Starting SFT training...")
        result = self.trainer.train()
        logger.info("Training complete.")

        # Save adapter
        self.trainer.save_model(f"{output_dir}/final_adapter")
        return result

    def save_adapter(self, path: str | None = None):
        if self.trainer and path:
            self.trainer.save_model(path)
