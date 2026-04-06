from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from ...contracts import ExperimentConfig
from ...models.factory import load_model_bundle


@dataclass(slots=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float


def build_preference_dataset(
    *,
    prompts: Sequence[str],
    chosen_fn: Callable[[str], str],
    rejected_fn: Callable[[str], str],
    reward_fn: Callable[[str, str], float],
) -> list[PreferencePair]:
    pairs: list[PreferencePair] = []
    for prompt in prompts:
        chosen = chosen_fn(prompt)
        rejected = rejected_fn(prompt)
        chosen_score = reward_fn(prompt, chosen)
        rejected_score = reward_fn(prompt, rejected)
        if chosen_score < rejected_score:
            chosen, rejected = rejected, chosen
            chosen_score, rejected_score = rejected_score, chosen_score
        pairs.append(
            PreferencePair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                chosen_score=chosen_score,
                rejected_score=rejected_score,
            )
        )
    return pairs


def build_preference_pairs_from_completions(
    *,
    prompts: Sequence[str],
    chosen_texts: Sequence[str],
    rejected_texts: Sequence[str],
    reward_fn: Callable[[str, str], float],
) -> list[PreferencePair]:
    """Same pairing logic as `build_preference_dataset`, but with precomputed generations (e.g. one GPU model at a time)."""
    if not (len(prompts) == len(chosen_texts) == len(rejected_texts)):
        raise ValueError("prompts, chosen_texts, and rejected_texts must have the same length")
    pairs: list[PreferencePair] = []
    for prompt, chosen, rejected in zip(prompts, chosen_texts, rejected_texts, strict=True):
        chosen_score = reward_fn(prompt, chosen)
        rejected_score = reward_fn(prompt, rejected)
        if chosen_score < rejected_score:
            chosen, rejected = rejected, chosen
            chosen_score, rejected_score = rejected_score, chosen_score
        pairs.append(
            PreferencePair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                chosen_score=chosen_score,
                rejected_score=rejected_score,
            )
        )
    return pairs


def run_dpo(
    config: ExperimentConfig,
    preference_pairs: Sequence[PreferencePair],
) -> dict[str, Any]:
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import DPOTrainer

    model, tokenizer = load_model_bundle(config.model, config.peft)
    dataset = Dataset.from_list([asdict(pair) for pair in preference_pairs])
    args = TrainingArguments(
        output_dir=config.training.output_dir,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        beta=0.1,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    train_output = trainer.train()
    trainer.save_model(config.training.output_dir)
    return {
        "trainer": "trl.DPOTrainer",
        "preference_pairs": len(preference_pairs),
        "train_runtime": getattr(train_output, "metrics", {}).get("train_runtime"),
        "train_loss": getattr(train_output, "metrics", {}).get("train_loss"),
    }
