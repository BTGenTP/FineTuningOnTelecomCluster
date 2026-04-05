#!/usr/bin/env python3
"""Minimal prompt experiment: replay XML (smoke test for ExperimentRunner)."""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-run prompt experiment with a fixed XML replay.")
    p.add_argument("--mission", type=str, default="Mission: inspection...")
    p.add_argument(
        "--xml",
        type=str,
        default="real_inspection_mission.xml",
        help="XML file to replay as model output.",
    )
    p.add_argument(
        "--catalog",
        type=str,
        default=None,
        help="Catalog path (default: merged if present).",
    )
    return p.parse_args()


def main() -> int:
    from _paths import ensure_sys_path

    ensure_sys_path()
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    if args.catalog:
        cat = Path(args.catalog)
        catalog_path = str(cat if cat.is_absolute() else (root / cat).resolve())
    else:
        merged = root / "data" / "nav4rail_catalog_merged.json"
        base = root / "data" / "nav4rail_catalog.json"
        catalog_path = str(merged if merged.exists() else base)

    xml_path = Path(args.xml)
    if not xml_path.is_absolute():
        xml_path = (root / xml_path).resolve()
    xml = xml_path.read_text(encoding="utf-8")

    from src.contracts import ExperimentConfig, ModelConfig, PromptConfig
    from src.evaluation.runner import ExperimentRunner

    cfg = ExperimentConfig(
        name="prompt_smoke",
        task="xml_generation",
        output_root=str(root / "runs" / "prompt_smoke"),
        method="zero_shot",
        model=ModelConfig(model_name_or_path="dummy"),
        prompt=PromptConfig(mode="zero_shot"),
        catalog_path=catalog_path,
    )
    runner = ExperimentRunner(cfg)
    res = runner.run_prompt_experiment(mission=args.mission, generate_fn=lambda _m: xml)
    print(res["run_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
