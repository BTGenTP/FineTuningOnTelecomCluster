"""
Unified training entry point for NAV4RAIL benchmarking.
========================================================
Dispatches to the appropriate trainer based on configuration.

Usage:
    python -m src.train.unified_trainer --config configs/base.yaml --model gemma2_9b
    python -m src.train.unified_trainer --config configs/base.yaml --model gemma2_9b --method grpo
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

TRAINERS = {
    "sft": "src.train.sft_trainer:SFTTrainerWrapper",
    "dpo": "src.train.dpo_trainer:DPOTrainerWrapper",
    "grpo": "src.train.grpo_trainer:GRPOTrainerWrapper",
    "kto": "src.train.kto_trainer:KTOTrainerWrapper",
    "orpo": "src.train.orpo_trainer:ORPOTrainerWrapper",
    "ppo": "src.train.ppo_trainer:PPOTrainerWrapper",
}


def _import_trainer(path: str):
    module_path, class_name = path.rsplit(":", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def train(cfg: dict) -> dict[str, Any]:
    """Run training based on config."""
    method = cfg.get("training", {}).get("method", "sft")

    if method not in TRAINERS:
        raise ValueError(f"Unknown training method: {method}. Available: {list(TRAINERS.keys())}")

    # Setup wandb
    wandb_project = cfg.get("experiment", {}).get("wandb_project", "nav4rail-bench")
    model_key = cfg.get("model", {}).get("key", "unknown")
    peft_method = cfg.get("peft", {}).get("method", "qlora")
    phase = cfg.get("experiment", {}).get("phase", "?")

    if cfg.get("training", {}).get("report_to") == "wandb":
        try:
            import wandb

            wandb.init(
                project=wandb_project,
                config=cfg,
                name=f"{method}_{model_key}_{peft_method}",
                tags=[method, model_key, peft_method, f"phase_{phase}"],
            )
        except ImportError:
            logger.warning("wandb not installed, skipping tracking")

    # Import and run trainer
    TrainerClass = _import_trainer(TRAINERS[method])
    trainer = TrainerClass(cfg)
    result = trainer.train()

    # Auto-evaluate after training
    auto_eval = cfg.get("eval", {}).get("auto_eval", True)
    if auto_eval:
        logger.info("Running automatic post-training evaluation...")
        try:
            from src.eval.benchmark import run_benchmark

            run_benchmark(cfg)
        except Exception as e:
            logger.error(f"Auto-eval failed: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="NAV4RAIL Unified Trainer")
    parser.add_argument("--config", default="configs/base.yaml", help="Base config YAML")
    parser.add_argument("--model", default=None, help="Model key (e.g. gemma2_9b)")
    parser.add_argument("--method", default=None, help="Training method (e.g. grpo)")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    from src.utils.config import load_config

    cfg = load_config(args.config, model=args.model, method=args.method)

    if args.resume:
        cfg["resume_from_checkpoint"] = args.resume

    model_key = cfg.get("model", {}).get("key", "unknown")
    method = cfg.get("training", {}).get("method", "sft")
    logger.info(f"Starting {method} training for {model_key}")

    train(cfg)


if __name__ == "__main__":
    main()
