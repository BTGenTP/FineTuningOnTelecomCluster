"""PPO Trainer wrapper. Uses validate_bt as reward via value head."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class PPOTrainerWrapper:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def train(self) -> dict[str, Any]:
        # PPO requires AutoModelForCausalLMWithValueHead from trl
        # More complex setup than other trainers
        from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

        from src.data.skills_loader import SkillsCatalog
        from src.reward.reward_fn import make_reward_fn
        from src.utils.config import get_active_model_config, resolve_paths

        cfg = resolve_paths(self.cfg)
        model_config = get_active_model_config(cfg)

        logger.info("PPO training — complex setup, consider GRPO as simpler alternative")

        # TODO: Full PPO implementation requires:
        # 1. Load model with value head (AutoModelForCausalLMWithValueHead)
        # 2. Setup reference model
        # 3. Configure PPO with KL penalty
        # 4. Training loop: generate -> reward -> PPO step
        # This is intentionally left as a stub — GRPO is recommended over PPO.

        raise NotImplementedError(
            "PPO trainer not yet implemented. Use GRPO (GRPOTrainerWrapper) "
            "which achieves similar results without a value head or reference model."
        )
