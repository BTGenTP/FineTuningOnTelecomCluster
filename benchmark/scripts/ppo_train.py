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
    p.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional PEFT adapter path (e.g. SFT LoRA) to initialize policy and frozen ref.",
    )
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
    from src.evaluation.run_manifest import write_manifest_for_run
    from src.evaluation.runner import create_run_paths
    from src.models.factory import apply_peft, load_base_model, load_tokenizer
    from src.rewards.reward_fn import reward_from_xml
    from src.xml_utils import extract_root_xml

    cfg = load_experiment_config(Path(args.config))
    if args.output_root:
        cfg.output_root = str(Path(args.output_root))
    if args.adapter:
        cfg.peft.adapter_path = args.adapter

    paths = create_run_paths(cfg.output_root)
    paths.experiment_json.write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    write_manifest_for_run(
        paths,
        config_path=Path(args.config).resolve(),
        cfg=cfg,
        extra={"adapter_cli": args.adapter, "prompts_file": args.prompts, "script": "ppo_train.py"},
    )

    catalog = load_catalog(cfg.catalog_path or default_catalog_path())
    system_prompt = render_system_prompt(catalog, include_schema=cfg.prompt.include_schema)

    def build_prompt(mission: str) -> str:
        return f"{system_prompt}\n\nMission:\n{mission.strip()}\n\nXML:\n"

    try:
        from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
    except Exception as exc:
        raise SystemExit(
            "TRL PPO components not available: "
            f"{exc}\n"
            "Install a TRL release that still exports the classic PPO API, e.g. "
            "`pip install 'trl>=0.26.0,<0.29.0'` (0.29+ removed PPO from `trl` root; "
            "this script is not ported to `trl.experimental.ppo` yet)."
        )

    if not cfg.peft.method and not cfg.peft.adapter_path:
        raise SystemExit("PPO LoRA: set peft.method (e.g. lora) and/or peft.adapter_path / --adapter for SFT init.")

    tokenizer = load_tokenizer(cfg.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_inner = apply_peft(load_base_model(cfg.model), cfg.peft)
    model = AutoModelForCausalLMWithValueHead(policy_inner).to(device)

    ref_inner = apply_peft(load_base_model(cfg.model), cfg.peft)
    ref_inner.load_state_dict(policy_inner.state_dict(), strict=True)
    ref_model = AutoModelForCausalLMWithValueHead(ref_inner).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

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
            gen_kw: dict = {
                "max_new_tokens": cfg.generation.max_new_tokens,
                "do_sample": cfg.generation.do_sample,
                "temperature": cfg.generation.temperature,
                "top_p": cfg.generation.top_p,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if int(getattr(cfg.generation, "top_k", -1)) > 0:
                gen_kw["top_k"] = int(cfg.generation.top_k)
            response_t = trainer.generate(
                query_t,
                **gen_kw,
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
    inner = getattr(model, "pretrained_model", None)
    if inner is not None:
        inner.save_pretrained(out_dir)
    else:
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
