from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Mapping, Sequence

from ..contracts import ExperimentConfig
from ..data.collators import build_completion_only_collator
from ..data.formatting import format_sft_record, render_system_prompt
from ..models.factory import load_model_bundle


def prepare_sft_records(records: Sequence[Mapping[str, Any]], catalog: Mapping[str, Any], *, include_schema: bool) -> list[dict[str, Any]]:
    system_prompt = render_system_prompt(catalog, include_schema=include_schema)
    return [format_sft_record(record, system_prompt=system_prompt) for record in records]


def run_sft(
    config: ExperimentConfig,
    train_rows: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer

    model, tokenizer = load_model_bundle(config.model, config.peft)
    formatted_rows = prepare_sft_records(train_rows, catalog, include_schema=config.prompt.include_schema)
    dataset = Dataset.from_list(formatted_rows)
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
        bf16=config.model.dtype.lower() in {"bf16", "bfloat16"},
        fp16=config.model.dtype.lower() in {"fp16", "float16"},
        report_to=[],
    )
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
