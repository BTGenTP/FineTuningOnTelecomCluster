from __future__ import annotations

import argparse
from pathlib import Path

from scripts._paths import ensure_sys_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SFT (optionally PEFT) on synthetic or JSONL dataset.")
    p.add_argument("--config", type=str, default="configs/sft_lora.yaml", help="Experiment YAML config.")
    p.add_argument("--dataset", type=str, default=None, help="Optional JSONL with {'mission','xml'} rows.")
    p.add_argument("--generate-synthetic", type=int, default=200, help="If --dataset absent, generate N synthetic records.")
    p.add_argument("--write-dataset", type=str, default="data/dataset_synthetic.jsonl", help="Where to write generated dataset (optional).")
    p.add_argument("--output-root", type=str, default=None, help="Override config.output_root (e.g. runs/sft).")
    return p.parse_args()


def main() -> int:
    ensure_sys_path()
    args = parse_args()

    from src.config_loader import load_experiment_config
    from src.evaluation.runner import ExperimentRunner

    cfg = load_experiment_config(Path(args.config))
    if args.output_root:
        cfg.output_root = str(Path(args.output_root))
    runner = ExperimentRunner(cfg)

    rows: list[dict] = []
    if args.dataset:
        from scripts._jsonl import read_jsonl

        rows = list(read_jsonl(args.dataset))
    else:
        from src.data.synthetic_generator import iter_dataset
        from scripts._jsonl import write_jsonl

        rows = list(iter_dataset(args.generate_synthetic))
        if args.write_dataset:
            write_jsonl(args.write_dataset, rows)

    result = runner.run_sft_experiment(train_rows=rows)
    print("run_dir:", result["run_dir"])
    print("training:", result["training"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

