#!/usr/bin/env python3
"""Generate synthetic NAV4RAIL dataset JSONL (wrapper for README quick start)."""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write synthetic dataset JSONL.")
    p.add_argument("--n", type=int, default=100, help="Number of records.")
    p.add_argument(
        "--output",
        type=str,
        default="data/dataset_synthetic.jsonl",
        help="Output JSONL path (relative to benchmark root).",
    )
    return p.parse_args()


def main() -> int:
    from _paths import ensure_sys_path

    ensure_sys_path()
    args = parse_args()
    from src.data.synthetic_generator import iter_dataset, write_jsonl

    root = Path(__file__).resolve().parents[1]
    out = (root / args.output).resolve()
    write_jsonl(out, iter_dataset(args.n))
    print(f"wrote {args.n} records -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
