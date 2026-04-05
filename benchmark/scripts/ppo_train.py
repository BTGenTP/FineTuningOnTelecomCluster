from __future__ import annotations

import argparse
import random
from pathlib import Path

from _paths import ensure_sys_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train with PPO (TRL) using deterministic validator reward.")
    p.add_argument("--config", type=str, default="configs/ppo.yaml")
    p.add_argument("--prompts", type=str, default=None, help="Optional JSONL with {'mission': ...} rows.")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", type=str, default=None, help="Override config.output_root.")
    return p.parse_args()


def _load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts:
        from _jsonl import read_jsonl

        rows = list(read_jsonl(args.prompts))
        missions = [str(r.get("mission", "")).strip() for r in rows]
        return [m for m in missions if m]
    from src.data.synthetic_generator import PROMPT_TEMPLATES

    random.seed(args.seed)
    return [random.choice(PROMPT_TEMPLATES) for _ in range(args.n)]


def main() -> int:
    ensure_sys_path()
    args = parse_args()

    import torch

    from src.config_loader import load_experiment_config
    from src.data.catalog import default_catalog_path, load_catalog
    from src.data.formatting import render_system_prompt
    from src.evaluation.runner import create_run_paths
    from src.rewards.reward_fn import reward_from_xml
    from src.xml_utils import extract_root_xml

    cfg = load_experiment_config(Path(args.config))
    if args.output_root:
        cfg.output_root = str(Path(args.output_root))

    paths = create_run_paths(cfg.output_root)
    paths.experiment_json.write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")

    catalog = load_catalog(cfg.catalog_path or default_catalog_path())
    system_prompt = render_system_prompt(catalog, include_schema=cfg.prompt.include_schema)

    def build_prompt(mission: str) -> str:
        return f"{system_prompt}\n\nMission:\n{mission.strip()}\n\nXML:\n"

    try:
        from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
    except Exception as exc:
        raise SystemExit(f"TRL PPO components not available: {exc}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name_or_path or cfg.model.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(cfg.model.model_name_or_path).to(device)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(cfg.model.model_name_or_path).to(device)

    ppo_cfg = PPOConfig(
        learning_rate=cfg.training.learning_rate,
        batch_size=cfg.training.batch_size,
        mini_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        log_with=None,
    )
    trainer = PPOTrainer(config=ppo_cfg, model=model, ref_model=ref_model, tokenizer=tokenizer, dataset=None)

    missions = _load_prompts(args)
    random.seed(args.seed)

    rows = []
    for epoch in range(args.epochs):
        random.shuffle(missions)
        for mission in missions:
            query = build_prompt(mission)
            query_t = tokenizer(query, return_tensors="pt").input_ids[0].to(device)
            response_t = trainer.generate(
                query_t,
                max_new_tokens=cfg.generation.max_new_tokens,
                do_sample=cfg.generation.do_sample,
                temperature=cfg.generation.temperature,
                top_p=cfg.generation.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            response_text = tokenizer.decode(response_t, skip_special_tokens=True)
            xml = extract_root_xml(response_text) or response_text.strip()

            score, breakdown, _report = reward_from_xml(
                xml_text=xml,
                catalog_path=cfg.catalog_path,
                xsd_path=cfg.xsd_path,
                strict=True,
                constraints_dir=str(Path("constraints").resolve()),
            )
            stats = trainer.step([query_t], [response_t], [torch.tensor(score, device=device)])
            rows.append(
                {
                    "epoch": epoch,
                    "mission_preview": mission[:80],
                    "reward": float(score),
                    "pass_l1": breakdown.pass_l1,
                    "pass_l2": breakdown.pass_l2,
                    "pass_l3": breakdown.pass_l3,
                    "ppo_stats": {k: float(v) for k, v in (stats or {}).items() if isinstance(v, (int, float))},
                }
            )

    out_dir = Path(cfg.training.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    from src.evaluation.metrics import render_markdown_table
    import json

    paths.metrics_json.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths.summary_md.write_text(render_markdown_table(rows[-50:]) + "\n", encoding="utf-8")
    print("run_dir:", paths.run_dir)
    print("saved_model:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

