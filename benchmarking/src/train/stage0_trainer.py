"""
Stage 0 — Continued Pretraining on BT XML grammar.
==================================================

Goal: teach the LLM the BTCPP v4 XML grammar (tags, attributes, port shapes)
without burning the catalogue vocabulary into the weights. Conditional —
activate only if the SFT baseline plateaus (mean_score < 0.85) AND structural
diversity is measurably deficient.

Design:
  - Plain causal-LM training over `data/stage0_corpus.jsonl` (text-only,
    no chat template, no system prompt).
  - QLoRA 4-bit + MLPs unblocked (gate_proj, up_proj, down_proj) — same
    target_modules pattern as BTGenBot, gives the MLPs room to absorb the
    grammar without disturbing attention.
  - Low LR (1e-5), short schedule (1-2 epochs) — minimise catastrophic
    forgetting of the base model's general capabilities.
  - Perplexity probe BEFORE and AFTER on a held-out set of 50 valid BTs.
    If post-training perplexity on un-masked NAV4RAIL skills INCREASES
    (catastrophic forgetting on catalogue vocabulary), abort: this Stage 0
    run is harmful.

Inputs (cfg keys):
  stage0:
    corpus: "data/stage0_corpus.jsonl"          # built by build_stage0_corpus.py
    eval_size: 0.05
    epochs: 2
    lr: 1.0e-5
    perplexity_probe: "data/stage0_perplexity_probe.jsonl"  # held-out NAV4RAIL BTs
    perplexity_threshold_pct: 5.0  # abort if perplexity rises by > this %

Output:
  runs/stage0_<model>/  — adapter compatible with later SFT runs
"""

from __future__ import annotations

import inspect
import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_corpus_dataset(cfg: dict, tokenizer):
    """Load JSONL `text` field as a tokenised dataset for causal-LM training."""
    from datasets import Dataset

    stage0_cfg = cfg.get("stage0", {})
    corpus_path = stage0_cfg.get("corpus", "data/stage0_corpus.jsonl")
    eval_size = float(stage0_cfg.get("eval_size", 0.05))
    seed = cfg.get("experiment", {}).get("seed", 42)
    max_seq_len = int(stage0_cfg.get("max_seq_len", 4096))

    examples = []
    with open(corpus_path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            if text:
                examples.append({"text": text})

    if not examples:
        raise RuntimeError(
            f"Stage 0 corpus is empty: {corpus_path}. Run "
            "`python -m src.data.build_stage0_corpus --root ... --out ...` first."
        )

    logger.info(f"Stage 0 corpus: {len(examples)} BT documents")
    ds = Dataset.from_list(examples)
    ds = ds.shuffle(seed=seed)
    split = ds.train_test_split(test_size=eval_size, seed=seed)

    def _tokenise(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )

    train_tok = split["train"].map(_tokenise, batched=True, remove_columns=["text"])
    eval_tok = split["test"].map(_tokenise, batched=True, remove_columns=["text"])
    return train_tok, eval_tok


def _perplexity(model, tokenizer, jsonl_path: str | Path, max_seq_len: int = 4096) -> float:
    """Compute mean per-token perplexity on a JSONL of raw BT XMLs (`text` field)."""
    import torch

    path = Path(jsonl_path)
    if not path.is_file():
        logger.warning(f"Perplexity probe missing ({path}) — skipping ppl check")
        return float("nan")

    texts: list[str] = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            rec = json.loads(line)
            t = rec.get("text") or rec.get("xml") or ""
            if t.strip():
                texts.append(t.strip())

    if not texts:
        return float("nan")

    model.eval()
    nll_sum = 0.0
    n_tokens = 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
            input_ids = enc["input_ids"].to(model.device)
            if input_ids.shape[1] < 2:
                continue
            outputs = model(input_ids=input_ids, labels=input_ids)
            # outputs.loss is averaged per token, multiply by tok count
            n = input_ids.shape[1] - 1  # next-token prediction
            nll_sum += float(outputs.loss.item()) * n
            n_tokens += n
    if n_tokens == 0:
        return float("nan")
    return math.exp(nll_sum / n_tokens)


class Stage0TrainerWrapper:
    """Continued pretraining (causal LM) on BT XML corpus.

    Compatible with SLURM/vast.ai. Same `train()` interface as the other
    `*TrainerWrapper` classes so it can be wired into `unified_trainer.py`.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _build_training_args(self, output_dir: str):
        from transformers import TrainingArguments

        train_cfg = self.cfg.get("training", {})
        stage0_cfg = self.cfg.get("stage0", {})
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=int(stage0_cfg.get("epochs", 2)),
            per_device_train_batch_size=int(train_cfg.get("batch_size", 1)),
            gradient_accumulation_steps=int(train_cfg.get("grad_accum", 16)),
            learning_rate=float(stage0_cfg.get("lr", 1e-5)),
            lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
            warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 0.3)),
            gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
            optim=train_cfg.get("optim", "paged_adamw_8bit"),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=10,
            bf16=self.cfg.get("model", {}).get("bf16", True),
            report_to=train_cfg.get("report_to", "wandb"),
            seed=self.cfg["experiment"]["seed"],
        )

    def train(self) -> dict[str, Any]:
        from transformers import DataCollatorForLanguageModeling, Trainer

        from src.utils.model_loader import load_for_training

        stage0_cfg = self.cfg.get("stage0", {})
        model_key = self.cfg.get("model", {}).get("key", "unknown")
        output_dir = stage0_cfg.get("output_dir") or f"runs/stage0_{model_key}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Force the loader to apply the BTGenBot-style MLP-unblocked LoRA
        # target modules unless the caller already overrode them.
        peft = self.cfg.setdefault("peft", {})
        if not peft.get("target_modules"):
            peft["target_modules"] = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        self.model, self.tokenizer = load_for_training(self.cfg)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Pre-training perplexity probe ──────────────────────────────────
        probe_path = stage0_cfg.get(
            "perplexity_probe", "data/stage0_perplexity_probe.jsonl"
        )
        ppl_before = _perplexity(self.model, self.tokenizer, probe_path)
        logger.info(f"[Stage 0] perplexity BEFORE on {probe_path}: {ppl_before:.3f}")

        # ── Build dataset ──────────────────────────────────────────────────
        train_ds, eval_ds = _load_corpus_dataset(self.cfg, self.tokenizer)

        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        training_args = self._build_training_args(output_dir)

        trainer_kwargs: dict[str, Any] = dict(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
        )
        # transformers 4.46 renamed `tokenizer` → `processing_class` on Trainer.
        try:
            params = inspect.signature(Trainer.__init__).parameters
            if "processing_class" in params:
                trainer_kwargs["processing_class"] = self.tokenizer
            elif "tokenizer" in params:
                trainer_kwargs["tokenizer"] = self.tokenizer
        except Exception:
            trainer_kwargs["tokenizer"] = self.tokenizer

        self.trainer = Trainer(**trainer_kwargs)

        logger.info("[Stage 0] starting continued pretraining ...")
        result = self.trainer.train()
        logger.info("[Stage 0] training complete.")

        # ── Post-training perplexity probe + abort gate ────────────────────
        ppl_after = _perplexity(self.model, self.tokenizer, probe_path)
        logger.info(f"[Stage 0] perplexity AFTER on {probe_path}: {ppl_after:.3f}")

        threshold_pct = float(stage0_cfg.get("perplexity_threshold_pct", 5.0))
        verdict = {
            "ppl_before": ppl_before,
            "ppl_after": ppl_after,
            "delta_pct": (
                100.0 * (ppl_after - ppl_before) / ppl_before
                if math.isfinite(ppl_before) and ppl_before > 0
                else float("nan")
            ),
            "threshold_pct": threshold_pct,
            "decision": "accept",
        }
        if (
            math.isfinite(ppl_before)
            and math.isfinite(ppl_after)
            and ppl_before > 0
            and (ppl_after - ppl_before) / ppl_before * 100.0 > threshold_pct
        ):
            verdict["decision"] = "REVERT (catastrophic forgetting suspected)"
            logger.error(
                f"[Stage 0] REVERT: perplexity rose {verdict['delta_pct']:.2f}% > {threshold_pct}% threshold."
            )

        # Persist the adapter and the verdict either way (verdict guides next step).
        self.trainer.save_model(f"{output_dir}/final_adapter")
        Path(output_dir, "stage0_verdict.json").write_text(
            json.dumps(verdict, indent=2), encoding="utf-8"
        )
        logger.info(f"[Stage 0] verdict written: {verdict}")

        return {"trainer_result": result, "verdict": verdict}
