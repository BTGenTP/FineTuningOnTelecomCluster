from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


@dataclass(slots=True)
class PpoStepResult:
    prompt: str
    response: str
    reward: float
    advantage: float


def run_ppo_epoch(
    *,
    prompts: Sequence[str],
    generate_fn: Callable[[str], str],
    reward_fn: Callable[[str, str], float],
    value_fn: Callable[[str, str], float],
) -> list[PpoStepResult]:
    results: list[PpoStepResult] = []
    for prompt in prompts:
        response = generate_fn(prompt)
        reward = reward_fn(prompt, response)
        baseline = value_fn(prompt, response)
        results.append(PpoStepResult(prompt=prompt, response=response, reward=reward, advantage=reward - baseline))
    return results
