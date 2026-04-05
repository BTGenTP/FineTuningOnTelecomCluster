from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GrpoTrainStats:
    reward_mean: float
    reward_min: float
    reward_max: float
    loss: float
    mean_kl: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward_mean": self.reward_mean,
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "loss": self.loss,
            "mean_kl": self.mean_kl,
        }

