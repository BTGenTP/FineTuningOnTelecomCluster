"""
Benchmark evaluation runner for NAV4RAIL.
==========================================
Evaluates a model on the fixed test set and produces metrics + detailed results.

Usage:
    python -m src.eval.benchmark --config configs/base.yaml --model mistral_7b
    python -m src.eval.benchmark --adapter runs/slurm/nav4rail_sft_lora_XXXX/best_checkpoint
    python -m src.eval.benchmark --constraint gbnf        # transformers-cfg decoding
    python -m src.eval.benchmark --constraint outlines    # outlines JSON -> XML
    python -m src.eval.benchmark --constraint all         # iterate over none/gbnf/outlines
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
        method if method in (
            "zero_shot", "few_shot", "schema_guided", "chain_of_thought",
            "pot", "react_agent",
        )
        else "zero_shot"
    )
    peft_method = cfg.get("peft", {}).get("method", "none")
    phase = cfg.get("experiment", {}).get("phase", "?")
    project = cfg.get("experiment", {}).get("wandb_project", "nav4rail-bench")
    entity = cfg.get("experiment", {}).get("wandb_entity")

    constraint_mode = (
        cfg.get("eval", {}).get("constraint", {}).get("mode", "none") or "none"
    )
    run_kind = "eval_adapter" if adapter_path else f"eval_{prompt_mode}"
    run_name = f"{run_kind}_{model_key}"
    if constraint_mode != "none":
        run_name = f"{run_name}_{constraint_mode}"
    tags = [
        "eval", model_key, prompt_mode, peft_method, f"phase_{phase}",
        f"constraint_{constraint_mode}",
    ]
    if adapter_path:
        tags.append("adapter")

    try:
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=tags,
            group=os.environ.get("WANDB_RUN_GROUP") or f"eval_{method}_{constraint_mode}",
            job_type="inference_constrained" if constraint_mode != "none" else "inference_baseline",
            config={
                **cfg,
                "adapter_path": adapter_path,
                "prompt_mode": prompt_mode,
                "constraint_mode": constraint_mode,
            },
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("wandb.init failed (%s) — continuing without tracking", e)
        return None, False

    return wandb, True


def _generate_xml(
    model, tokenizer, prompt, eval_cfg: dict, constraint=None,
) -> tuple[str, float, int]:
    """Generate XML from prompt, return (xml_str, latency_s, n_tokens).

    `constraint` is an optional `ConstraintHandle` (from src.eval.constrained).
    When active, its LogitsProcessor is merged into generate kwargs; if the
    backend emits JSON (Outlines JSON-mode), its post-processor converts to XML.
    """
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

    if constraint is not None and constraint.is_active():
        from src.eval.constrained import apply_to_generate_kwargs

        apply_to_generate_kwargs(gen_kwargs, constraint)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    latency = time.time() - t0

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    n_tokens = len(new_tokens)
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Outlines JSON-mode post-processing (JSON -> XML). GBNF returns XML as-is.
    if constraint is not None and constraint.post_process is not None:
        xml_str = constraint.post_process(raw)
    else:
        xml_str = raw

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
    from src.data.prompt_builder import build_prompt, build_system_prompt
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
    if training_method in (
        "zero_shot", "few_shot", "schema_guided", "chain_of_thought",
        "pot", "react_agent",
    ):
        prompt_mode = training_method
    else:
        prompt_mode = "zero_shot"  # Default for trained models

    # ── Grammar-constrained decoding handle (built once, reused per mission) ──
    constraint_mode = (
        cfg.get("eval", {}).get("constraint", {}).get("mode", "none") or "none"
    )
    constraint = None
    if constraint_mode != "none":
        from src.eval.constrained import build_constraint

        constraint = build_constraint(
            mode=constraint_mode,
            tokenizer=tokenizer,
            cfg=cfg,
            catalog=catalog,
        )
        logger.info(
            "Constrained decoding ACTIVE — mode=%s backend_version=%s",
            constraint.mode, constraint.backend_version,
        )

    # ── Code-as-Reasoning agents (single construction, reused per mission) ──
    agent = None
    if training_method == "pot":
        from src.agents.pot_agent import PoTAgent

        agent = PoTAgent(
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            pot_cfg=cfg.get("pot", {}),
            catalog=catalog,
            safety_rules=safety_rules,
        )
    elif training_method == "react_agent":
        from src.agents.react_agent import ReActAgent

        agent = ReActAgent(
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            react_cfg=cfg.get("react_agent", {}),
            catalog=catalog,
            safety_rules=safety_rules,
        )

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

        agent_meta: dict[str, Any] = {}

        if agent is not None:
            agent_result = agent.run(mission_text)
            xml_str = agent_result.xml
            latency_s = agent_result.total_latency_s
            n_tokens = agent_result.n_tokens
            agent_meta = {
                "agent_success": agent_result.success,
                "agent_code": agent_result.code,
                "agent_n_iterations": agent_result.n_iterations,
                "agent_llm_latency_s": agent_result.llm_latency_s,
                "agent_sandbox_latency_s": agent_result.sandbox_latency_s,
                "agent_error_type": agent_result.error_type,
                "agent_error_message": agent_result.error_message,
            }
        else:
            # Build prompt
            prompt = build_prompt(
                mode=prompt_mode,
                mission=mission_text,
                model_config=model_config,
                catalog=catalog,
                safety_rules=safety_rules,
            )

            # Generate (constraint may be None for baseline)
            xml_str, latency_s, n_tokens = _generate_xml(
                model, tokenizer, prompt, eval_cfg, constraint=constraint,
            )

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
            **agent_meta,
        })

        # Stream per-sample metrics to W&B (lightweight: no raw XML to avoid cost)
        if wandb_mod is not None:
            m_dict = asdict(metrics)
            log_entry = {
                "eval/step": i,
                "eval/score": m_dict.get("score", 0.0),
                "eval/valid": int(bool(m_dict.get("valid", False))),
                "eval/latency_s": m_dict.get("latency_s", 0.0),
                "eval/n_tokens": m_dict.get("n_tokens", 0),
                f"eval/by_cat/{category}/score": m_dict.get("score", 0.0),
            }
            if agent_meta:
                log_entry["eval/agent_iterations"] = agent_meta.get("agent_n_iterations", 0)
                log_entry["eval/agent_llm_latency_s"] = agent_meta.get("agent_llm_latency_s", 0.0)
                log_entry["eval/agent_sandbox_latency_s"] = agent_meta.get(
                    "agent_sandbox_latency_s", 0.0
                )
            wandb_mod.log(log_entry)

        if (i + 1) % 10 == 0:
            logger.info(f"  [{i + 1}/{len(test_missions)}] score={metrics.score:.2f}")

    # Aggregate
    summary = aggregate_metrics(all_metrics)
    summary["model_key"] = cfg.get("model", {}).get("key", "unknown")
    summary["training_method"] = training_method
    summary["peft_method"] = cfg.get("peft", {}).get("method", "none")
    summary["adapter_path"] = adapter_path
    summary["constraint_mode"] = constraint_mode
    if constraint is not None:
        summary["constraint_backend_version"] = constraint.backend_version
        # Hash the grammar / schema source to make runs reproducibly traceable.
        import hashlib
        try:
            if constraint.mode == "gbnf":
                gp = Path(
                    cfg["eval"]["constraint"].get("gbnf_path", "src/eval/bt_grammar.gbnf")
                )
                if not gp.is_absolute():
                    gp = Path(__file__).parent.parent.parent / gp
                if gp.is_file():
                    summary["grammar_sha256"] = hashlib.sha256(gp.read_bytes()).hexdigest()[:16]
            elif constraint.mode == "outlines":
                summary["outlines_schema_spec"] = cfg["eval"]["constraint"].get("outlines_schema")
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to compute constraint source hash: %s", e)

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

    with open(out_path / "system_prompt.txt", "w", encoding="utf-8") as f:
        f.write(build_system_prompt(safety_rules=safety_rules).rstrip() + "\n")

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
                        choices=["zero_shot", "few_shot", "schema_guided", "chain_of_thought",
                                 "pot", "react_agent",
                                 "gbnf", "outlines", "all_constraints"])
    parser.add_argument("--constraint", default=None,
                        choices=["none", "gbnf", "outlines", "all"],
                        help="Grammar-constrained decoding backend (overrides cfg.eval.constraint.mode).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from src.utils.config import load_config

    # Method configs that live under configs/methods/ and should be merged via load_config.
    # Agents (pot, react_agent) change prompt building; constraint presets
    # (gbnf, outlines, all_constraints) only set eval.constraint.*.
    _MERGEABLE_METHODS = ("pot", "react_agent", "gbnf", "outlines", "all_constraints")
    method_override = args.prompt_mode if args.prompt_mode in _MERGEABLE_METHODS else None
    cfg = load_config(args.config, model=args.model, method=method_override)
    if args.prompt_mode:
        cfg.setdefault("training", {})["method"] = args.prompt_mode

    if args.constraint:
        cfg.setdefault("eval", {}).setdefault("constraint", {})["mode"] = args.constraint

    # ── Constraint matrix: mode == "all" iterates over none/gbnf/outlines ───
    # Model is loaded once and reused across constraint modes — cheap comparison.
    requested = cfg.get("eval", {}).get("constraint", {}).get("mode", "none") or "none"
    if requested == "all":
        from src.utils.model_loader import load_for_inference

        model, tokenizer = load_for_inference(cfg, adapter_path=args.adapter)
        logger.info("Constraint matrix mode: iterating over [none, gbnf, outlines]")
        # One W&B run per constraint pass; default group ties them together in the UI.
        if not os.environ.get("WANDB_RUN_GROUP"):
            _mk = cfg.get("model", {}).get("key", "unknown")
            _mt = cfg.get("training", {}).get("method", "zero_shot")
            os.environ["WANDB_RUN_GROUP"] = f"eval_matrix_{_mt}_{_mk}"

        base_output = args.output
        results = {}
        for mode in ("none", "gbnf", "outlines"):
            logger.info("─" * 60)
            logger.info("Constraint pass: %s", mode)
            logger.info("─" * 60)
            import copy as _copy

            cfg_mode = _copy.deepcopy(cfg)
            cfg_mode["eval"]["constraint"]["mode"] = mode

            if base_output:
                out_dir = f"{base_output.rstrip('/')}_{mode}"
            else:
                import time as _time
                model_key = cfg.get("model", {}).get("key", "unknown")
                method = cfg.get("training", {}).get("method", "zero_shot")
                out_dir = f"runs/{method}_{mode}_{model_key}_{int(_time.time())}"

            try:
                results[mode] = run_benchmark(
                    cfg_mode,
                    model=model,
                    tokenizer=tokenizer,
                    adapter_path=args.adapter,
                    output_dir=out_dir,
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("Constraint mode %s failed: %s", mode, e)
                results[mode] = {"error": str(e)}

        logger.info("─" * 60)
        logger.info("Matrix summary:")
        for mode, summary in results.items():
            if "error" in summary:
                logger.info("  %-10s FAILED: %s", mode, summary["error"])
            else:
                logger.info(
                    "  %-10s validity=%.1f%% mean_score=%.3f",
                    mode,
                    100 * summary.get("xml_validity_rate", 0.0),
                    summary.get("mean_score", 0.0),
                )
        return

    run_benchmark(cfg, adapter_path=args.adapter, output_dir=args.output)


if __name__ == "__main__":
    main()
