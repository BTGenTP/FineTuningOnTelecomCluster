from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Callable, Sequence


@dataclass(slots=True)
class GrpoCandidate:
    prompt: str
    response: str
    reward: float
    normalized_advantage: float


def _normalize_group(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    avg = mean(scores)
    std = pstdev(scores) or 1.0
    return [(score - avg) / std for score in scores]


def run_grpo_epoch(
    *,
    prompts: Sequence[str],
    group_size: int,
    generate_group_fn: Callable[[str, int], list[str]],
    reward_fn: Callable[[str, str], float],
) -> list[GrpoCandidate]:
    candidates: list[GrpoCandidate] = []
    for prompt in prompts:
        responses = generate_group_fn(prompt, group_size)
        rewards = [reward_fn(prompt, response) for response in responses]
        advantages = _normalize_group(rewards)
        for response, reward, advantage in zip(responses, rewards, advantages):
            candidates.append(
                GrpoCandidate(
                    prompt=prompt,
                    response=response,
                    reward=reward,
                    normalized_advantage=advantage,
                )
            )
    return candidates
