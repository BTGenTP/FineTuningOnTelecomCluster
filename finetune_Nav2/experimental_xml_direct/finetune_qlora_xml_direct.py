from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset  # type: ignore
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments  # type: ignore
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer  # type: ignore

from finetune_Nav2.catalog.catalog_io import default_catalog_path, load_catalog
from finetune_Nav2.eval.bt_validation import validate_bt_xml


MODELS = {
    "mistral7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_r": 16,
        "lora_alpha": 32,
        "max_seq_len": 1536,
        "batch_size": 4,
        "grad_accum": 16,
        "epochs": 8,
        "lr": 2e-4,
    }
}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def load_dataset(path: Path) -> Dataset:
    rows = _load_jsonl(path)
    ds = Dataset.from_list([{"text": str(r["prompt"])} for r in rows if isinstance(r, dict) and r.get("prompt")])
    split = ds.train_test_split(test_size=0.1, seed=42)
    return split


def load_model_and_tokenizer(cfg: Dict[str, Any]):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    # Filter LoRA targets to existing leaf module names.
    present_leaf = {name.split(".")[-1] for name, _m in model.named_modules() if name}
    targets = [t for t in cfg["lora_targets"] if t in present_leaf]
    if not targets:
        raise RuntimeError(f"No LoRA targets found. Config={cfg['lora_targets']}")

    lora_config = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        target_modules=list(targets),
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA SFT: mission -> BT XML (experimental).")
    p.add_argument("--model", choices=sorted(MODELS.keys()), default="mistral7b")
    p.add_argument("--dataset", type=str, required=True, help="XML-direct dataset JSONL (with field prompt).")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = dict(MODELS[str(args.model)])
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)

    dataset_path = Path(args.dataset).expanduser().resolve()
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(__file__).resolve().parent / "outputs" / f"nav2_xml_direct_{args.model}_lora"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(cfg)
    split = load_dataset(dataset_path)

    collator = DataCollatorForCompletionOnlyLM(
        response_template="[/INST]",
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(cfg["batch_size"]),
        per_device_eval_batch_size=int(cfg["batch_size"]),
        gradient_accumulation_steps=int(cfg["grad_accum"]),
        num_train_epochs=float(cfg["epochs"]),
        learning_rate=float(cfg["lr"]),
        fp16=True,
        bf16=False,
        logging_steps=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        dataset_text_field="text",
        data_collator=collator,
        max_seq_length=int(cfg["max_seq_len"]),
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

