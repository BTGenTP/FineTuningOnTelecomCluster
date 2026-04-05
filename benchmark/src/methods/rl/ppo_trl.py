from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PpoTrainStats:
    reward: float
    ppo_stats: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {"reward": self.reward, "ppo_stats": dict(self.ppo_stats)}

