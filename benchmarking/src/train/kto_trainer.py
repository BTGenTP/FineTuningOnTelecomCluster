"""KTO Trainer wrapper. Uses trl.KTOTrainer — needs only good/bad labels, not pairs."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class KTOTrainerWrapper:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def train(self) -> dict[str, Any]:
        from datasets import Dataset
        from trl import KTOConfig, KTOTrainer

        from src.utils.config import get_active_model_config
        from src.utils.model_loader import load_for_training

        model_config = get_active_model_config(self.cfg)
        kto_cfg = self.cfg.get("kto", {})
        train_cfg = self.cfg.get("training", {})

        model, tokenizer = load_for_training(self.cfg)

        data_path = self.cfg["data"].get("kto_dataset", "data/dataset_kto.jsonl")
        examples = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        ds = Dataset.from_list(examples)
        split = ds.train_test_split(test_size=0.05, seed=self.cfg["experiment"]["seed"])

        model_key = self.cfg.get("model", {}).get("key", "unknown")
        output_dir = f"runs/kto_{model_key}"

        training_args = KTOConfig(
            output_dir=output_dir,
            num_train_epochs=train_cfg.get("epochs", 3),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=5e-6,
            beta=kto_cfg.get("beta", 0.1),
            desirable_weight=kto_cfg.get("desirable_weight", 1.0),
            undesirable_weight=kto_cfg.get("undesirable_weight", 1.0),
            bf16=model_config.get("bf16", True),
            gradient_checkpointing=True,
            report_to=train_cfg.get("report_to", "wandb"),
            seed=self.cfg["experiment"]["seed"],
        )

        trainer = KTOTrainer(
            model=model,
            args=training_args,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            processing_class=tokenizer,
        )

        logger.info("Starting KTO training...")
        result = trainer.train()
        trainer.save_model(f"{output_dir}/final_adapter")
        return result
