from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset  # type: ignore
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training  # type: ignore
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer  # type: ignore

from finetune_Nav2.catalog.catalog_io import default_catalog_path, load_catalog
from finetune_Nav2.train.model_registry import MODELS, ModelSpec
from finetune_Nav2.train.prompting import (
    build_chat_messages,
    build_mistral_inst_prompt,
    build_phi2_prompt,
)


def _load_jsonl_dataset(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def build_text_row(spec: ModelSpec, tokenizer, catalog: Dict[str, Any], mission: str, steps_json: str) -> str:
    """
    Build final text that includes prompt + expected completion.
    The completion-only collator will mask tokens before response_anchor.
    """
    if spec.chat_template:
        msgs = build_chat_messages(mission=mission, catalog=catalog)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return prompt + "\n### Steps JSON:\n" + steps_json

    if spec.key == "phi2":
        prompt, _anchor = build_phi2_prompt(mission=mission, catalog=catalog)
        return prompt + steps_json

    # default: Mistral-style [INST]
    prompt, _anchor = build_mistral_inst_prompt(mission=mission, catalog=catalog)
    return prompt + steps_json + " </s>"


def load_dataset_jsonl(path: Path, spec: ModelSpec, tokenizer, catalog: Dict[str, Any]) -> Dataset:
    rows = _load_jsonl_dataset(path)
    if not rows:
        raise ValueError(f"Empty dataset: {path}")

    examples: List[Dict[str, str]] = []
    for r in rows:
        mission = str(r.get("mission") or "").strip()
        steps_json = str(r.get("steps_json") or "").strip()
        if not mission or not steps_json:
            continue
        text = build_text_row(spec, tokenizer, catalog, mission, steps_json)
        examples.append({"text": text})

    ds = Dataset.from_list(examples)
    split = ds.train_test_split(test_size=0.1, seed=42)
    split["train"].shuffle(seed=42)
    split["test"].shuffle(seed=42)
    return split


def load_model_and_tokenizer(spec: ModelSpec):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    # Filter LoRA targets to modules that actually exist (robust across model variants).
    present_leaf_names = set()
    for name, _mod in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf:
            present_leaf_names.add(leaf)
    targets = [t for t in spec.lora_targets if t in present_leaf_names]
    if not targets:
        raise RuntimeError(
            f"No LoRA target modules found for model {spec.hf_id}. "
            f"Configured={spec.lora_targets} present_sample={sorted(list(present_leaf_names))[:30]}"
        )

    lora_config = LoraConfig(
        r=int(spec.lora_r),
        lora_alpha=int(spec.lora_alpha),
        target_modules=list(targets),
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA SFT: mission -> steps JSON (Nav2 proxy).")
    p.add_argument("--model-key", choices=sorted(MODELS.keys()), required=True)
    p.add_argument("--dataset", type=str, required=True, help="Dataset JSONL path.")
    p.add_argument("--catalog", type=str, default=str(default_catalog_path()))
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    spec0 = MODELS[str(args.model_key)]
    spec = ModelSpec(
        **{
            **spec0.__dict__,
            "epochs": int(args.epochs) if args.epochs is not None else spec0.epochs,
            "lr": float(args.lr) if args.lr is not None else spec0.lr,
            "max_seq_len": int(args.max_seq_len) if args.max_seq_len is not None else spec0.max_seq_len,
            "batch_size": int(args.batch_size) if args.batch_size is not None else spec0.batch_size,
            "grad_accum": int(args.grad_accum) if args.grad_accum is not None else spec0.grad_accum,
        }
    )

    dataset_path = Path(args.dataset).expanduser().resolve()
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(__file__).resolve().parents[1] / "outputs" / f"nav2_steps_{spec.key}_lora"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(args.catalog)
    model, tokenizer = load_model_and_tokenizer(spec)

    split = load_dataset_jsonl(dataset_path, spec, tokenizer, catalog)

    collator = DataCollatorForCompletionOnlyLM(
        response_template=str(spec.response_anchor),
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(spec.batch_size),
        per_device_eval_batch_size=int(spec.batch_size),
        gradient_accumulation_steps=int(spec.grad_accum),
        num_train_epochs=float(spec.epochs),
        learning_rate=float(spec.lr),
        fp16=True,
        bf16=False,
        logging_steps=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        dataset_text_field="text",
        data_collator=collator,
        max_seq_length=int(spec.max_seq_len),
        packing=False,
    )

    trainer.train()

    adapter_dir = out_dir / "lora_adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved adapter: {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

