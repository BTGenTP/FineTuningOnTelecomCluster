from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from scripts._paths import ensure_sys_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run prompt-based benchmark (batch) and write per-run artifacts.")
    p.add_argument("--config", type=str, default="configs/prompt_zero_shot.yaml", help="Experiment YAML config.")
    p.add_argument("--missions", type=str, default=None, help="Optional JSONL with {'mission': ...} rows.")
    p.add_argument("--n", type=int, default=10, help="Number of missions to evaluate (if missions file absent).")
    p.add_argument("--fewshot-pool", type=str, default=None, help="Optional JSONL with {'mission','xml'} for dynamic few-shot selection.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", type=str, default=None, help="Override config.output_root (e.g. runs/prompt_eval).")
    return p.parse_args()


def _load_missions(args: argparse.Namespace) -> list[str]:
    if args.missions:
        from scripts._jsonl import read_jsonl

        rows = list(read_jsonl(args.missions))
        missions = [str(r.get("mission", "")).strip() for r in rows]
        return [m for m in missions if m]

    from src.data.synthetic_generator import PROMPT_TEMPLATES

    random.seed(args.seed)
    return [random.choice(PROMPT_TEMPLATES) for _ in range(args.n)]


def main() -> int:
    root = ensure_sys_path()
    args = parse_args()

    from src.config_loader import load_experiment_config
    from src.evaluation.metrics import write_reports
    from src.evaluation.runner import ExperimentRunner
    from src.models.factory import load_model_bundle

    cfg = load_experiment_config(Path(args.config))
    if args.output_root:
        cfg.output_root = str(Path(args.output_root))

    runner = ExperimentRunner(cfg)
    model, tokenizer = load_model_bundle(cfg.model, cfg.peft)
    device = getattr(model, "device", None)
    device_str = str(device) if device is not None else None

    from scripts._hf_generate import HFChatGenerator

    generator = HFChatGenerator(model=model, tokenizer=tokenizer, device=device_str)

    fewshot_pool: list[Any] = []
    pool_path = args.fewshot_pool or cfg.prompt.few_shot_pool_path
    if pool_path and cfg.prompt.few_shot_k:
        from scripts._fewshot import load_pool
        from scripts._jsonl import read_jsonl

        fewshot_pool = load_pool(read_jsonl(pool_path))

    missions = _load_missions(args)
    rows: list[dict[str, Any]] = []
    static_examples = []
    if fewshot_pool and cfg.prompt.few_shot_k and not cfg.prompt.few_shot_dynamic:
        static_examples = [row.as_prompt_example() for row in fewshot_pool[: cfg.prompt.few_shot_k]]
    for mission in missions:
        examples = []
        if static_examples:
            examples = static_examples
        elif fewshot_pool and cfg.prompt.few_shot_k:
            from scripts._fewshot import select_top_k

            examples = select_top_k(mission, fewshot_pool, cfg.prompt.few_shot_k)
        result = runner.run_prompt_experiment(
            mission=mission,
            generate_fn=lambda messages: generator.generate(
                messages,
                max_new_tokens=cfg.generation.max_new_tokens,
                temperature=cfg.generation.temperature,
                top_p=cfg.generation.top_p,
                do_sample=cfg.generation.do_sample,
            ),
            few_shot_examples=examples,
        )
        rows.append(result["metrics"])

    reports = write_reports(cfg.output_root, rows)
    print("Wrote reports:", {k: str(v) for k, v in reports.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

