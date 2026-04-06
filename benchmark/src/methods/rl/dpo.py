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
    import os
    import inspect
    import torch

    # Reduces allocator fragmentation on long runs (safe default if the job did not export it).
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if not getattr(config.peft, "method", None) and total_gb < 30:
            raise SystemExit(
                "DPO OOM safeguard: full fine-tune of a 7B model is not stable on ~16GB GPUs.\n"
                "Use QLoRA/LoRA instead (example for P100):\n"
                "- model.quantization: nf4\n"
                "- peft.method: lora (with target_modules)\n"
                "- training.max_seq_length: 1024-2048\n"
            )

    model, tokenizer = load_model_bundle(config.model, config.peft)
    # Long XML + DPO concatenates chosen/rejected in one forward; cap length to limit attention memory.
    max_length = min(int(config.training.max_seq_length), 4096)

    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    dataset = Dataset.from_list([asdict(pair) for pair in preference_pairs])
    # DPOConfig/DPOTrainer kwargs changed across TRL releases; only pass supported ones.
    _cfg_sig = inspect.signature(DPOConfig.__init__)
    _cfg_params = set(_cfg_sig.parameters.keys()) - {"self"}
    _beta = 0.1
    _dpo_cfg_kwargs = {
        "output_dir": config.training.output_dir,
        "per_device_train_batch_size": config.training.batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "num_train_epochs": config.training.num_train_epochs,
        "logging_steps": getattr(config.training, "logging_steps", 10),
        "save_steps": getattr(config.training, "save_steps", 200),
        "report_to": [],
        "beta": _beta,
        "max_length": max_length,
        # Avoid policy forward + ref forward in the same step (peak VRAM on ~16 GiB cards).
        "precompute_ref_log_probs": True,
        "precompute_ref_batch_size": 1,
        "gradient_checkpointing": True,
    }
    args = DPOConfig(**{k: v for k, v in _dpo_cfg_kwargs.items() if k in _cfg_params})

    _trainer_sig = inspect.signature(DPOTrainer.__init__)
    _trainer_params = set(_trainer_sig.parameters.keys()) - {"self"}
    _trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": args,
        "train_dataset": dataset,
    }
    # Some TRL versions keep `beta` on the trainer instead of config.
    if "beta" in _trainer_params and "beta" not in _cfg_params:
        _trainer_kwargs["beta"] = _beta
    # tokenizer kwarg name varies: `processing_class` (new) vs `tokenizer` (older)
    if "processing_class" in _trainer_params:
        _trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in _trainer_params:
        _trainer_kwargs["tokenizer"] = tokenizer
    trainer = DPOTrainer(**_trainer_kwargs)
    train_output = trainer.train()
    trainer.save_model(config.training.output_dir)
    return {
        "trainer": "trl.DPOTrainer",
        "preference_pairs": len(preference_pairs),
        "train_runtime": getattr(train_output, "metrics", {}).get("train_runtime"),
        "train_loss": getattr(train_output, "metrics", {}).get("train_loss"),
    }
