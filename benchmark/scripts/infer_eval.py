from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from _paths import ensure_sys_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference+validation on a batch of missions (base or adapter).")
    p.add_argument("--config", type=str, required=True, help="Experiment YAML config (prompting mode).")
    p.add_argument("--missions", type=str, default=None, help="Optional JSONL with {'mission': ...} rows.")
    p.add_argument("--n", type=int, default=20, help="Number of missions to evaluate (if missions file absent).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adapter", type=str, default=None, help="Optional PEFT adapter path override.")
    p.add_argument("--output-root", type=str, default=None, help="Override config.output_root.")
    return p.parse_args()


def _load_missions(args: argparse.Namespace) -> list[str]:
    if args.missions:
        from _jsonl import read_jsonl

        rows = list(read_jsonl(args.missions))
        missions = [str(r.get("mission", "")).strip() for r in rows]
        return [m for m in missions if m]
    from src.data.synthetic_generator import PROMPT_TEMPLATES

    random.seed(args.seed)
    return [random.choice(PROMPT_TEMPLATES) for _ in range(args.n)]


def main() -> int:
    ensure_sys_path()
    args = parse_args()

    from src.config_loader import load_experiment_config
    from src.evaluation.metrics import write_reports
    from src.evaluation.runner import ExperimentRunner
    from src.models.factory import load_model_bundle

    cfg = load_experiment_config(Path(args.config))
    if args.output_root:
        cfg.output_root = str(Path(args.output_root))
    if args.adapter:
        cfg.peft.adapter_path = args.adapter

    runner = ExperimentRunner(cfg)
    model, tokenizer = load_model_bundle(cfg.model, cfg.peft)
    device = getattr(model, "device", None)
    device_str = str(device) if device is not None else None

    from _hf_generate import HFChatGenerator

    generator = HFChatGenerator(model=model, tokenizer=tokenizer, device=device_str)

    missions = _load_missions(args)
    rows: list[dict[str, Any]] = []
    for mission in missions:
        result = runner.run_prompt_experiment(
            mission=mission,
            generate_fn=lambda messages: generator.generate(
                messages,
                max_new_tokens=cfg.generation.max_new_tokens,
                temperature=cfg.generation.temperature,
                top_p=cfg.generation.top_p,
                do_sample=cfg.generation.do_sample,
            ),
        )
        rows.append(result["metrics"])

    reports = write_reports(cfg.output_root, rows)
    print("Wrote reports:", {k: str(v) for k, v in reports.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

