from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from finetune_Nav2_XML.catalog.catalog_io import default_catalog_path
from finetune_Nav2_XML.eval.bt_validation import compute_bt_structure_metrics, validate_bt_xml
from finetune_Nav2_XML.eval.run_artifacts import create_run_dir, next_run_id, now_iso_z, write_json, write_text
from finetune_Nav2_XML.eval.xml_extraction import extract_root_xml


def _load_dataset_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Static eval (oracle): dataset XML -> validate -> write runs/ artifacts.")
    p.add_argument("--dataset", type=str, required=True, help="Path to dataset JSONL with fields mission + xml.")
    p.add_argument("--catalog", type=str, default=str(default_catalog_path()))
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strict-attrs", action="store_true")
    p.add_argument("--strict-blackboard", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    dataset_path = Path(args.dataset).expanduser().resolve()
    rows = _load_dataset_jsonl(dataset_path)
    if not rows:
        raise SystemExit(f"Empty dataset: {dataset_path}")

    import random

    random.seed(int(args.seed))
    random.shuffle(rows)
    rows = rows[: int(args.n)]

    for row in rows:
        mission = str(row.get("mission") or "").strip()
        if not mission:
            raise SystemExit("Dataset row missing mission.")

        llm_raw = str(row.get("xml") or "")
        xml = extract_root_xml(llm_raw) or llm_raw.strip()

        t0 = time.perf_counter()
        run_id = next_run_id()
        rp = create_run_dir(run_id)

        experiment = {
            "id": run_id,
            "llm": {"provider": "oracle_dataset", "model": None, "temperature": None},
            "generator": {"mode": "oracle", "constraints": {"enabled": False, "kind": None}},
            "validation": {"strict_attrs": bool(args.strict_attrs), "strict_blackboard": bool(args.strict_blackboard)},
            "inputs": {"dataset": str(dataset_path)},
        }

        write_text(rp.mission_txt, mission)
        write_json(rp.experiment_json, experiment)
        write_text(rp.prompt_rendered_txt, mission)
        write_text(rp.llm_xml_raw_txt, llm_raw)
        write_text(rp.generated_bt_xml, xml)

        report = validate_bt_xml(
            xml_path=rp.generated_bt_xml,
            strict_attrs=bool(args.strict_attrs),
            strict_blackboard=bool(args.strict_blackboard),
            catalog_path=Path(args.catalog).expanduser().resolve(),
        )
        write_json(rp.validation_report_json, report)

        latency_ms = int((time.perf_counter() - t0) * 1000.0)
        ok = bool(report.get("ok"))
        summary = report.get("summary") or {}

        metrics: Dict[str, Any] = {
            "schema_version": "0.1",
            "run_id": run_id,
            "timestamps": {"run_started_at": now_iso_z(), "run_finished_at": None},
            "llm": {
                "provider": "oracle_dataset",
                "api_base": None,
                "model": None,
                "temperature": None,
                "latency_ms": latency_ms,
                "tokens": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                "errors": {},
            },
            "bt": {
                "xml_valid": ok,
                "validator": {
                    "errors": (summary.get("errors") if isinstance(summary, dict) else None),
                    "warnings": (summary.get("warnings") if isinstance(summary, dict) else None),
                    "issues_total": (summary.get("issues_total") if isinstance(summary, dict) else None),
                },
                "structure": compute_bt_structure_metrics(rp.generated_bt_xml),
            },
            "simulation": {"enabled": False},
        }

        metrics["timestamps"]["run_finished_at"] = now_iso_z()
        write_json(rp.metrics_json, metrics)
        print(f"[{run_id}] xml_valid={ok} wrote: {rp.run_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

