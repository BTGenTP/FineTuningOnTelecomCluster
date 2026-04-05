#!/usr/bin/env python3
"""Validate a BehaviorTree XML file (or stdin) with the deterministic validator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate BT XML against catalog + constraints.")
    p.add_argument(
        "--xml",
        type=str,
        default=None,
        help="Path to XML file (default: real_inspection_mission.xml in benchmark root).",
    )
    p.add_argument(
        "--catalog",
        type=str,
        default=None,
        help="Catalog JSON (default: data/nav4rail_catalog_merged.json if present else data/nav4rail_catalog.json).",
    )
    p.add_argument("--constraints", type=str, default="constraints", help="Constraints directory.")
    p.add_argument("--strict", action="store_true", default=True)
    return p.parse_args()


def main() -> int:
    from _paths import ensure_sys_path

    ensure_sys_path()
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    if args.xml:
        xml_path = Path(args.xml).expanduser()
        if not xml_path.is_absolute():
            xml_path = (root / xml_path).resolve()
        xml_text = xml_path.read_text(encoding="utf-8")
    else:
        default_xml = root / "real_inspection_mission.xml"
        xml_text = default_xml.read_text(encoding="utf-8")

    if args.catalog:
        catalog = Path(args.catalog).expanduser()
        if not catalog.is_absolute():
            catalog = (root / catalog).resolve()
        catalog_path = str(catalog)
    else:
        merged = root / "data" / "nav4rail_catalog_merged.json"
        base = root / "data" / "nav4rail_catalog.json"
        catalog_path = str(merged if merged.exists() else base)

    constraints_dir = Path(args.constraints)
    if not constraints_dir.is_absolute():
        constraints_dir = (root / constraints_dir).resolve()

    from src.rewards.validator import validate

    report = validate(
        xml_text=xml_text,
        catalog_path=catalog_path,
        constraints_dir=str(constraints_dir),
        strict=args.strict,
    )
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
