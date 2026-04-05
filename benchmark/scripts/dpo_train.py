from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from _paths import ensure_sys_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build preference pairs (chosen/rejected) and run DPO training.")
    p.add_argument("--config", type=str, default="configs/dpo.yaml")
    p.add_argument("--prompts", type=str, default=None, help="Optional JSONL with {'mission': ...} rows.")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chosen-adapter", type=str, default=None, help="Adapter path for chosen model (e.g. artifacts/sft_lora).")
    p.add_argument("--rejected-adapter", type=str, default=None, help="Adapter path for rejected model (optional).")
    p.add_argument("--output-root", type=str, default=None, help="Override config.output_root.")
    p.add_argument("--max-new-tokens", type=int, default=None, help="Override generation.max_new_tokens.")
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

    from src.config_loader import load_experiment_config
    from src.data.catalog import default_catalog_path, load_catalog
    from src.methods.prompting import render_prompt_bundle
    from src.models.factory import load_model_bundle
    from src.rewards.reward_fn import reward_from_xml
    from src.xml_utils import extract_root_xml

    cfg = load_experiment_config(Path(args.config))
    if args.output_root:
        cfg.output_root = str(Path(args.output_root))
    if args.max_new_tokens is not None:
        cfg.generation.max_new_tokens = int(args.max_new_tokens)

    # Catalog for prompt rendering (system rules).
    catalog = load_catalog(cfg.catalog_path or default_catalog_path())

    # Two generators: chosen (usually SFT/PEFT), rejected (usually base).
    chosen_cfg = cfg
    rejected_cfg = load_experiment_config(Path(args.config))  # fresh copy
    if args.chosen_adapter:
        chosen_cfg.peft.adapter_path = args.chosen_adapter
    if args.rejected_adapter:
        rejected_cfg.peft.adapter_path = args.rejected_adapter
    else:
        rejected_cfg.peft.adapter_path = None

    chosen_model, chosen_tok = load_model_bundle(chosen_cfg.model, chosen_cfg.peft)
    rejected_model, rejected_tok = load_model_bundle(rejected_cfg.model, rejected_cfg.peft)
    chosen_device = str(getattr(chosen_model, "device", "")) or None
    rejected_device = str(getattr(rejected_model, "device", "")) or None

    from _hf_generate import HFChatGenerator

    chosen_gen = HFChatGenerator(model=chosen_model, tokenizer=chosen_tok, device=chosen_device)
    rejected_gen = HFChatGenerator(model=rejected_model, tokenizer=rejected_tok, device=rejected_device)

    def _generate_xml(gen: HFChatGenerator, mission: str) -> str:
        bundle = render_prompt_bundle(mission=mission, catalog=catalog, prompt_config=cfg.prompt, xsd_path=cfg.xsd_path)
        raw = gen.generate(
            bundle["messages"],
            max_new_tokens=cfg.generation.max_new_tokens,
            temperature=cfg.generation.temperature,
            top_p=cfg.generation.top_p,
            do_sample=cfg.generation.do_sample,
        )
        return extract_root_xml(raw) or raw.strip()

    def _reward(_prompt: str, xml: str) -> float:
        score, _breakdown, _report = reward_from_xml(
            xml_text=xml,
            catalog_path=cfg.catalog_path,
            xsd_path=cfg.xsd_path,
            strict=True,
            constraints_dir=str(Path("constraints").resolve()),
        )
        return score

    from src.methods.rl.dpo import build_preference_dataset
    from src.evaluation.runner import ExperimentRunner

    prompts = _load_prompts(args)
    pairs = build_preference_dataset(
        prompts=prompts,
        chosen_fn=lambda p: _generate_xml(chosen_gen, p),
        rejected_fn=lambda p: _generate_xml(rejected_gen, p),
        reward_fn=_reward,
    )

    runner = ExperimentRunner(cfg)
    res = runner.run_dpo_experiment(preference_pairs=pairs)
    print("run_dir:", res["run_dir"])
    print("training:", res["training"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

