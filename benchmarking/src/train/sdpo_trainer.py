"""
SDPO — Self-Distillation Iterated DPO with rich/text feedback.
==============================================================

Pipeline (one outer iteration):
  1. Sample N candidates per training mission from the current model
     (T=0.8, top_p=0.9). Optionally extend with programmatic perturbations.
  2. Score each completion via `compute_rich_feedback` (5-component dense
     score + text errors/warnings/hallucination list).
  3. Build preference pairs:
        - "single"      : top-1 (chosen) vs bottom-1 (rejected) per mission
        - "multi"       : K paires (chosen_i, rejected_j) covering more of the
                          score gradient — exploits the rich feedback ordering
        - "augmented"   : multi + a synthetic refinement chosen built by
                          appending the validator NL feedback as a
                          system/teacher hint **only on the chosen side**
                          (training-time text feedback, never at inference)
  4. Run trl.DPOTrainer on the resulting preference dataset.
  5. Optionally repeat (n_iterations) with the freshly trained model as the
     candidate generator.

The rich feedback (errors_text / warnings_text / hallucinated) is logged
alongside each pair so multi-pair construction can prefer pairs whose score
delta is informative across components, not just on the dense scalar.

Config (cfg["sdpo"]):
    n_candidates: 8
    pair_strategy: "multi"          # single | multi | augmented
    K_pairs: 4                      # for multi/augmented
    n_iterations: 1
    sampling:
      temperature: 0.8
      top_p: 0.9
      max_new_tokens: 4096
    weights:                        # rich feedback components
      parse: 0.3
      structure: 0.2
      semantic: 0.2
      coherence: 0.2
      hallucination: 0.1
    dpo:
      beta: 0.1
      lr: 5.0e-6
      epochs: 1
"""

from __future__ import annotations

import inspect
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _sample_candidates(
    model,
    tokenizer,
    prompt_texts: list[str],
    n_candidates: int,
    sampling_cfg: dict,
) -> list[list[str]]:
    """For each prompt, generate `n_candidates` completions.

    Returns a list of length len(prompt_texts), each element a list of
    `n_candidates` completion strings.
    """
    import torch

    temperature = float(sampling_cfg.get("temperature", 0.8))
    top_p = float(sampling_cfg.get("top_p", 0.9))
    max_new_tokens = int(sampling_cfg.get("max_new_tokens", 4096))

    out: list[list[str]] = []
    for prompt in prompt_texts:
        comps: list[str] = []
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        for _ in range(n_candidates):
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            new_tokens = gen[0][inputs["input_ids"].shape[1] :]
            comps.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
        out.append(comps)
    return out


def _build_pairs(
    prompt_text: str,
    completions: list[str],
    feedbacks: list,  # list[RichFeedback]
    strategy: str,
    k_pairs: int,
) -> list[dict[str, Any]]:
    """Construct (prompt, chosen, rejected) records from scored completions."""
    if not completions:
        return []
    ranked = sorted(
        zip(completions, feedbacks),
        key=lambda x: x[1].dense_score,
        reverse=True,
    )
    if len(ranked) < 2 or ranked[0][1].dense_score == ranked[-1][1].dense_score:
        return []  # no usable preference signal

    if strategy == "single":
        chosen, rej = ranked[0], ranked[-1]
        return [{
            "prompt": prompt_text,
            "chosen": chosen[0],
            "rejected": rej[0],
            "chosen_score": chosen[1].dense_score,
            "rejected_score": rej[1].dense_score,
            "chosen_components": chosen[1].components,
            "rejected_components": rej[1].components,
        }]

    pairs: list[dict[str, Any]] = []
    half = max(1, len(ranked) // 2)
    top = ranked[:half]
    bot = ranked[half:]
    # cartesian product, capped at k_pairs, biased toward larger score gaps
    cart = [(c, r) for c in top for r in bot]
    cart.sort(key=lambda cr: cr[0][1].dense_score - cr[1][1].dense_score, reverse=True)
    for c, r in cart[:k_pairs]:
        if c[1].dense_score <= r[1].dense_score:
            continue
        rec = {
            "prompt": prompt_text,
            "chosen": c[0],
            "rejected": r[0],
            "chosen_score": c[1].dense_score,
            "rejected_score": r[1].dense_score,
            "chosen_components": c[1].components,
            "rejected_components": r[1].components,
        }
        if strategy == "augmented":
            # Append the validator NL feedback to the chosen side as a teacher
            # hint. This is training-time only — the resulting `chosen` text
            # is a *target* the policy learns to imitate, not a prompt.
            rec["chosen"] = c[0] + "\n\n<!-- teacher_feedback\n" + c[1].to_prompt_text() + "\n-->"
        pairs.append(rec)
    return pairs


def _load_train_missions(cfg: dict) -> list[dict[str, Any]]:
    """Load missions for SDPO sampling. Reuses the SFT train file's prompts."""
    path = Path(cfg["data"]["train_dataset"])
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            rec = json.loads(line)
            mission = rec.get("mission") or rec.get("input") or rec.get("prompt")
            if mission:
                out.append({"mission": mission, **rec})
    return out


class SDPOTrainerWrapper:
    """Self-Distillation Iterated DPO with rich-feedback pair construction."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None

    def _generate_one_pass(self) -> list[dict[str, Any]]:
        """One iteration of (sample → score → build pairs).

        Returns the list of preference records to feed to DPOTrainer.
        """
        from src.data.prompt_builder import build_prompt
        from src.data.skills_loader import SafetyRulesLoader, SkillsCatalog
        from src.reward.rich_feedback import compute_rich_feedback
        from src.utils.config import get_active_model_config

        sdpo = self.cfg.get("sdpo", {})
        n_cand = int(sdpo.get("n_candidates", 8))
        strategy = sdpo.get("pair_strategy", "multi")
        k_pairs = int(sdpo.get("K_pairs", 4))
        weights = sdpo.get("weights")
        sampling_cfg = sdpo.get("sampling", {})

        catalog = SkillsCatalog()
        try:
            safety_rules = SafetyRulesLoader()
        except Exception:
            safety_rules = None
        model_config = get_active_model_config(self.cfg)

        missions = _load_train_missions(self.cfg)
        n_subset = int(sdpo.get("missions_per_iteration", min(len(missions), 200)))
        rng = random.Random(self.cfg["experiment"]["seed"])
        rng.shuffle(missions)
        missions = missions[:n_subset]
        logger.info(f"[SDPO] sampling {n_cand} candidates × {len(missions)} missions")

        prompts: list[str] = []
        for m in missions:
            built = build_prompt(
                mode="zero_shot",
                mission=m["mission"],
                model_config=model_config,
                catalog=catalog,
                safety_rules=safety_rules,
            )
            if isinstance(built, list):
                prompts.append(self.tokenizer.apply_chat_template(
                    built, tokenize=False, add_generation_prompt=True,
                ))
            else:
                prompts.append(built)

        all_completions = _sample_candidates(
            self.model, self.tokenizer, prompts, n_cand, sampling_cfg
        )

        all_pairs: list[dict[str, Any]] = []
        for mission_data, prompt, completions in zip(missions, prompts, all_completions):
            feedbacks = [
                compute_rich_feedback(c, mission_data["mission"], catalog, weights)
                for c in completions
            ]
            pairs = _build_pairs(prompt, completions, feedbacks, strategy, k_pairs)
            all_pairs.extend(pairs)

        logger.info(f"[SDPO] built {len(all_pairs)} preference pairs (strategy={strategy})")
        return all_pairs

    def _run_dpo_pass(self, pairs: list[dict[str, Any]], iteration: int):
        """Run trl.DPOTrainer on the constructed pairs."""
        from datasets import Dataset
        from transformers import TrainingArguments
        from trl import DPOTrainer

        sdpo = self.cfg.get("sdpo", {})
        dpo_cfg = sdpo.get("dpo", {})

        model_key = self.cfg.get("model", {}).get("key", "unknown")
        output_dir = f"runs/sdpo_{model_key}_iter{iteration}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Persist the pair dataset for reproducibility
        pair_path = Path(output_dir) / "pairs.jsonl"
        with pair_path.open("w", encoding="utf-8") as fp:
            for p in pairs:
                fp.write(json.dumps(p, ensure_ascii=False) + "\n")

        ds = Dataset.from_list([
            {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
            for p in pairs
        ])

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=int(dpo_cfg.get("epochs", 1)),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=int(dpo_cfg.get("grad_accum", 8)),
            learning_rate=float(dpo_cfg.get("lr", 5e-6)),
            lr_scheduler_type=dpo_cfg.get("lr_scheduler", "cosine"),
            warmup_ratio=0.03,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            logging_steps=10,
            bf16=self.cfg.get("model", {}).get("bf16", True),
            report_to=self.cfg.get("training", {}).get("report_to", "wandb"),
            seed=self.cfg["experiment"]["seed"],
        )

        try:
            params = inspect.signature(DPOTrainer.__init__).parameters
            kwargs: dict[str, Any] = dict(
                model=self.model,
                args=args,
                train_dataset=ds,
            )
            if "beta" in params:
                kwargs["beta"] = float(dpo_cfg.get("beta", 0.1))
            if "processing_class" in params:
                kwargs["processing_class"] = self.tokenizer
            elif "tokenizer" in params:
                kwargs["tokenizer"] = self.tokenizer
        except Exception:
            kwargs = {
                "model": self.model,
                "args": args,
                "train_dataset": ds,
                "tokenizer": self.tokenizer,
                "beta": float(dpo_cfg.get("beta", 0.1)),
            }

        trainer = DPOTrainer(**kwargs)
        result = trainer.train()
        trainer.save_model(f"{output_dir}/final_adapter")
        return result, output_dir

    def train(self) -> dict[str, Any]:
        from src.utils.model_loader import load_for_training

        self.model, self.tokenizer = load_for_training(self.cfg)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        n_iter = int(self.cfg.get("sdpo", {}).get("n_iterations", 1))
        last_dir = None
        for it in range(1, n_iter + 1):
            logger.info(f"[SDPO] iteration {it}/{n_iter}")
            pairs = self._generate_one_pass()
            if not pairs:
                logger.warning("[SDPO] no pairs produced — aborting iteration")
                break
            _, last_dir = self._run_dpo_pass(pairs, it)

        return {"final_adapter": f"{last_dir}/final_adapter" if last_dir else None}
