from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Mapping, Sequence

from ..contracts import ExperimentConfig
from ..data.formatting import format_sft_record, render_system_prompt
from ..models.factory import load_model_bundle


def prepare_sft_records(records: Sequence[Mapping[str, Any]], catalog: Mapping[str, Any], *, include_schema: bool) -> list[dict[str, Any]]:
    system_prompt = render_system_prompt(catalog, include_schema=include_schema)
    return [format_sft_record(record, system_prompt=system_prompt) for record in records]


def _amp_flags_for_cuda(bf16: bool, fp16: bool) -> tuple[bool, bool]:
    """Tune Trainer AMP flags for the active CUDA device.

    Pre-Ampere (capability major < 8, e.g. P100): disable both bf16 and fp16 Trainer AMP.
    Using fp16=True there enables GradScaler while some gradients can still be BF16
    (TRL/transformers/bitsandbytes), which triggers:
    RuntimeError: _amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16

    QLoRA/SFT still trains: adapter weights in FP32, quantized matmuls use BitsAndBytes compute dtype.
    """
    import torch

    if not torch.cuda.is_available():
        return bf16, fp16
    major, _minor = torch.cuda.get_device_capability()
    if major >= 8:
        return bf16, fp16
    return False, False


def run_sft(
    config: ExperimentConfig,
    train_rows: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from trl import SFTTrainer

    try:
        from trl import SFTConfig

        _trl_has_sft_config = True
    except ImportError:
        _trl_has_sft_config = False

    model, tokenizer = load_model_bundle(config.model, config.peft)
    formatted_rows = prepare_sft_records(train_rows, catalog, include_schema=config.prompt.include_schema)
    dataset = Dataset.from_list(formatted_rows)
    dtype = config.model.dtype.lower()
    bf16 = dtype in {"bf16", "bfloat16"}
    fp16 = dtype in {"fp16", "float16"}
    bf16, fp16 = _amp_flags_for_cuda(bf16, fp16)

    if _trl_has_sft_config:
        # TRL >= 1.0: DataCollatorForCompletionOnlyLM removed; use prompt+completion + SFTConfig.
        sft_args = SFTConfig(
            output_dir=config.training.output_dir,
            learning_rate=config.training.learning_rate,
            per_device_train_batch_size=config.training.batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            num_train_epochs=config.training.num_train_epochs,
            logging_steps=config.training.logging_steps,
            save_steps=config.training.save_steps,
            warmup_ratio=config.training.warmup_ratio,
            bf16=bf16,
            fp16=fp16,
            report_to=[],
            max_length=config.training.max_seq_length,
            packing=False,
            completion_only_loss=True,
        )
        # TRL/__post_init__ may tweak AMP; keep pre-Ampere GPUs on no-Trainer-AMP (see _amp_flags_for_cuda).
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
            sft_args.bf16 = False
            sft_args.fp16 = False
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=sft_args,
        )
    else:
        from transformers import TrainingArguments

        from ..data.collators import build_completion_only_collator

        collator = build_completion_only_collator(tokenizer, response_template="XML:\n")
        args = TrainingArguments(
            output_dir=config.training.output_dir,
            learning_rate=config.training.learning_rate,
            per_device_train_batch_size=config.training.batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            num_train_epochs=config.training.num_train_epochs,
            logging_steps=config.training.logging_steps,
            save_steps=config.training.save_steps,
            warmup_ratio=config.training.warmup_ratio,
            bf16=bf16,
            fp16=fp16,
            report_to=[],
        )
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
            args.bf16 = False
            args.fp16 = False
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=config.training.max_seq_length,
            data_collator=collator,
            args=args,
        )
    train_output = trainer.train()
    trainer.save_model(config.training.output_dir)
    return {
        "trainer": "trl.SFTTrainer",
        "train_samples": len(formatted_rows),
        "train_runtime": getattr(train_output, "metrics", {}).get("train_runtime"),
        "train_loss": getattr(train_output, "metrics", {}).get("train_loss"),
        "config": asdict(config.training),
    }
