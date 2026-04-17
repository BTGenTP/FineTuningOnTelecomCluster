"""
Benchmark evaluation runner for NAV4RAIL.
==========================================
Evaluates a model on the fixed test set and produces metrics + detailed results.

Usage:
    python -m src.eval.benchmark --config configs/base.yaml --model mistral_7b
    python -m src.eval.benchmark --adapter runs/slurm/nav4rail_sft_lora_XXXX/best_checkpoint
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _init_wandb_for_eval(cfg: dict, adapter_path: str | None) -> tuple[Any, bool]:
    """
    Initialize W&B for an evaluation run.

    Returns (wandb_module_or_None, owned_by_us).
    `owned_by_us` is True iff we called wandb.init() here and should wandb.finish().
    Reuses an existing run if one is active (e.g. called from unified_trainer after training).
    """
    eval_cfg = cfg.get("eval", {})
    train_cfg = cfg.get("training", {})
    report_to = eval_cfg.get("report_to", train_cfg.get("report_to", "none"))
    if report_to != "wandb":
        return None, False

    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed — skipping inference tracking")
        return None, False

    # Already inside an active run (e.g. training just called us): reuse it.
    if wandb.run is not None:
        return wandb, False

    model_key = cfg.get("model", {}).get("key", "unknown")
    method = train_cfg.get("method", "zero_shot")
    prompt_mode = (
        method if method in ("zero_shot", "few_shot", "schema_guided", "chain_of_thought")
        else "zero_shot"
    )
    peft_method = cfg.get("peft", {}).get("method", "none")
    phase = cfg.get("experiment", {}).get("phase", "?")
    project = cfg.get("experiment", {}).get("wandb_project", "nav4rail-bench")
    entity = cfg.get("experiment", {}).get("wandb_entity")

    run_kind = "eval_adapter" if adapter_path else f"eval_{prompt_mode}"
    run_name = f"{run_kind}_{model_key}"
    tags = ["eval", model_key, prompt_mode, peft_method, f"phase_{phase}"]
    if adapter_path:
        tags.append("adapter")

    try:
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=tags,
            group=os.environ.get("WANDB_RUN_GROUP") or f"eval_{method}",
            job_type="inference",
            config={
                **cfg,
                "adapter_path": adapter_path,
                "prompt_mode": prompt_mode,
            },
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("wandb.init failed (%s) — continuing without tracking", e)
        return None, False

    return wandb, True


def _generate_xml(
    model, tokenizer, prompt, eval_cfg: dict
) -> tuple[str, float, int]:
    """Generate XML from prompt, return (xml_str, latency_s, n_tokens)."""
    import torch

    temperature = eval_cfg.get("temperature", 0.0)
    max_new_tokens = eval_cfg.get("max_new_tokens", 4096)
    top_p = eval_cfg.get("top_p", 1.0)

    if isinstance(prompt, list):
        # Chat template
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    latency = time.time() - t0

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    n_tokens = len(new_tokens)
    xml_str = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return xml_str, latency, n_tokens


def run_benchmark(
    cfg: dict,
    model=None,
    tokenizer=None,
    adapter_path: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """
    Run full benchmark evaluation.

    Args:
        cfg: Configuration dict
        model: Pre-loaded model (if None, loads from cfg)
        tokenizer: Pre-loaded tokenizer
        adapter_path: Path to PEFT adapter
        output_dir: Where to save results (default: runs/<experiment_id>/)

    Returns:
        Aggregated metrics dict
    """
    from src.data.prompt_builder import build_prompt
    from src.data.skills_loader import SafetyRulesLoader, SkillsCatalog
    from src.eval.metrics import aggregate_metrics, compute_all_metrics
    from src.eval.validate_bt import enrich_ports
    from src.utils.config import get_active_model_config, resolve_paths

    cfg = resolve_paths(cfg, base_dir=Path(__file__).parent.parent.parent)

    # Load catalog
    catalog = SkillsCatalog(cfg["data"]["catalog"])
    safety_rules = SafetyRulesLoader(cfg["data"]["safety_rules"])

    # Load model if not provided
    if model is None or tokenizer is None:
        from src.utils.model_loader import load_for_inference

        model, tokenizer = load_for_inference(cfg, adapter_path=adapter_path)

    model_config = get_active_model_config(cfg)
    eval_cfg = cfg.get("eval", {})

    # Determine prompt mode
    training_method = cfg.get("training", {}).get("method", "sft")
    if training_method in ("zero_shot", "few_shot", "schema_guided", "chain_of_thought"):
        prompt_mode = training_method
    else:
        prompt_mode = "zero_shot"  # Default for trained models

    # Load test missions
    test_path = Path(cfg["data"]["test_missions"])
    if not test_path.is_file():
        raise FileNotFoundError(
            f"Test missions file missing: {test_path}\n"
            "Generate the fixed eval set with:\n"
            "  python -m src.data.generate_sft_dataset "
            "--output data/test_missions.json --n 100 --seed 42"
        )
    with open(test_path, encoding="utf-8") as f:
        test_missions = json.load(f)

    logger.info(f"Running benchmark on {len(test_missions)} missions...")

    # ── W&B tracking (reuses active run if called from trainer) ──────────────
    wandb_mod, wandb_owned = _init_wandb_for_eval(cfg, adapter_path)

    all_metrics = []
    detailed_results = []

    for i, mission_data in enumerate(test_missions):
        mission_text = mission_data["mission"]
        category = mission_data["category"]
        mission_id = mission_data["id"]

        # Build prompt
        prompt = build_prompt(
            mode=prompt_mode,
            mission=mission_text,
            model_config=model_config,
            catalog=catalog,
            safety_rules=safety_rules,
        )

        # Generate
        xml_str, latency_s, n_tokens = _generate_xml(model, tokenizer, prompt, eval_cfg)

        # Optionally enrich ports
        if eval_cfg.get("enrich_ports", True):
            xml_str = enrich_ports(xml_str, catalog)

        # Compute metrics
        metrics = compute_all_metrics(
            xml_str=xml_str,
            reference_xml=mission_data.get("reference_xml"),
            latency_s=latency_s,
            n_tokens=n_tokens,
            catalog=catalog,
        )

        all_metrics.append(metrics)
        detailed_results.append({
            "id": mission_id,
            "mission": mission_text,
            "category": category,
            "xml": xml_str,
            **asdict(metrics),
        })

        # Stream per-sample metrics to W&B (lightweight: no raw XML to avoid cost)
        if wandb_mod is not None:
            m_dict = asdict(metrics)
            wandb_mod.log({
                "eval/step": i,
                "eval/score": m_dict.get("score", 0.0),
                "eval/valid": int(bool(m_dict.get("valid", False))),
                "eval/latency_s": m_dict.get("latency_s", 0.0),
                "eval/n_tokens": m_dict.get("n_tokens", 0),
                f"eval/by_cat/{category}/score": m_dict.get("score", 0.0),
            })

        if (i + 1) % 10 == 0:
            logger.info(f"  [{i + 1}/{len(test_missions)}] score={metrics.score:.2f}")

    # Aggregate
    summary = aggregate_metrics(all_metrics)
    summary["model_key"] = cfg.get("model", {}).get("key", "unknown")
    summary["training_method"] = training_method
    summary["peft_method"] = cfg.get("peft", {}).get("method", "none")
    summary["adapter_path"] = adapter_path

    # Per-category breakdown
    from collections import defaultdict

    by_cat: dict[str, list] = defaultdict(list)
    for m, d in zip(all_metrics, detailed_results):
        by_cat[d["category"]].append(m)
    summary["per_category"] = {
        cat: {
            "n": len(ms),
            "validity_rate": sum(1 for m in ms if m.valid) / len(ms),
            "mean_score": sum(m.score for m in ms) / len(ms),
        }
        for cat, ms in by_cat.items()
    }

    # Save results
    if output_dir is None:
        model_key = cfg.get("model", {}).get("key", "unknown")
        run_id = f"{training_method}_{model_key}_{int(time.time())}"
        output_dir = f"runs/{run_id}"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out_path / "results_detail.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    with open(out_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)

    logger.info(f"Results saved to {out_path}/")
    logger.info(f"  Validity: {summary['xml_validity_rate']:.1%}")
    logger.info(f"  Mean score: {summary['mean_score']:.3f}")
    logger.info(f"  Perfect rate: {summary['perfect_score_rate']:.1%}")

    # ── W&B: log summary + artifacts ─────────────────────────────────────────
    if wandb_mod is not None:
        flat_summary = {
            "eval_summary/xml_validity_rate": summary.get("xml_validity_rate", 0.0),
            "eval_summary/mean_score": summary.get("mean_score", 0.0),
            "eval_summary/perfect_score_rate": summary.get("perfect_score_rate", 0.0),
            "eval_summary/n_missions": len(test_missions),
        }
        for cat, stats in summary.get("per_category", {}).items():
            flat_summary[f"eval_summary/per_cat/{cat}/mean_score"] = stats["mean_score"]
            flat_summary[f"eval_summary/per_cat/{cat}/validity_rate"] = stats["validity_rate"]
            flat_summary[f"eval_summary/per_cat/{cat}/n"] = stats["n"]

        for k, v in flat_summary.items():
            wandb_mod.run.summary[k] = v
        wandb_mod.log(flat_summary)

        # Upload detailed results as artifact
        try:
            artifact = wandb_mod.Artifact(
                name=f"eval-results-{summary['model_key']}-{int(time.time())}",
                type="eval-results",
                metadata={
                    "model_key": summary["model_key"],
                    "training_method": summary["training_method"],
                    "adapter_path": summary["adapter_path"],
                },
            )
            artifact.add_file(str(out_path / "metrics.json"))
            artifact.add_file(str(out_path / "results_detail.json"))
            wandb_mod.log_artifact(artifact)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to upload W&B artifact: %s", e)

        if wandb_owned:
            wandb_mod.finish()

    return summary


def main():
    parser = argparse.ArgumentParser(description="NAV4RAIL Benchmark Runner")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", default=None, help="Model key override")
    parser.add_argument("--adapter", default=None, help="Path to PEFT adapter")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--prompt-mode", default=None,
                        choices=["zero_shot", "few_shot", "schema_guided", "chain_of_thought"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from src.utils.config import load_config

    cfg = load_config(args.config, model=args.model)
    if args.prompt_mode:
        cfg.setdefault("training", {})["method"] = args.prompt_mode

    run_benchmark(cfg, adapter_path=args.adapter, output_dir=args.output)


if __name__ == "__main__":
    main()
