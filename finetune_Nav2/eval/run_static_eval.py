from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from finetune_Nav2.catalog.catalog_io import default_catalog_path, load_catalog
from finetune_Nav2.eval.bt_validation import compute_bt_structure_metrics, validate_bt_xml
from finetune_Nav2.eval.json_to_xml import build_bt_xml, steps_from_dicts
from finetune_Nav2.eval.run_artifacts import (
    create_run_dir,
    next_run_id,
    now_iso_z,
    write_json,
    write_text,
)
from finetune_Nav2.eval.steps_parsing import parse_steps_strict


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
    p = argparse.ArgumentParser(description="Static evaluation: steps JSON -> BT XML -> validate -> write runs/ artifacts.")
    p.add_argument("--dataset", type=str, required=True, help="Path to dataset JSONL with fields mission + steps/steps_json.")
    p.add_argument("--catalog", type=str, default=str(default_catalog_path()))
    p.add_argument("--n", type=int, default=1, help="Number of samples to run (each sample creates a run).")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed (sampling).")
    p.add_argument("--strict-attrs", action="store_true")
    p.add_argument("--strict-blackboard", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    dataset_path = Path(args.dataset).expanduser().resolve()
    rows = _load_dataset_jsonl(dataset_path)
    if not rows:
        raise SystemExit(f"Empty dataset: {dataset_path}")

    # Deterministic sampling
    import random

    random.seed(int(args.seed))
    random.shuffle(rows)
    rows = rows[: int(args.n)]

    catalog = load_catalog(args.catalog)

    for row in rows:
        mission = str(row.get("mission") or "").strip()
        if not mission:
            raise SystemExit("Dataset row missing mission.")

        # Oracle mode: use provided steps_json/steps as if it was the LLM output.
        llm_raw = row.get("steps_json")
        if not isinstance(llm_raw, str) or not llm_raw.strip():
            # fallback to serializing `steps`
            llm_raw = json.dumps(row.get("steps"), ensure_ascii=False)

        t0 = time.perf_counter()
        parsed = parse_steps_strict(llm_raw, catalog)
        latency_ms = int((time.perf_counter() - t0) * 1000.0)

        run_id = next_run_id()
        rp = create_run_dir(run_id)

        # Minimal experiment config snapshot
        experiment = {
            "id": run_id,
            "llm": {
                "provider": "oracle_dataset",
                "model": None,
                "temperature": None,
            },
            "generator": {
                "mode": "fail-fast",
                "constraints": {"enabled": False, "kind": None},
            },
            "validation": {
                "strict_attrs": bool(args.strict_attrs),
                "strict_blackboard": bool(args.strict_blackboard),
            },
            "inputs": {
                "dataset": str(dataset_path),
            },
        }

        write_text(rp.mission_txt, mission)
        write_json(rp.experiment_json, experiment)
        write_text(rp.prompt_rendered_txt, mission)  # oracle has no extra prompt rendering
        write_text(rp.llm_steps_raw_txt, llm_raw)

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
                "errors": parsed.errors,
            },
            "bt": {"xml_valid": False, "validator": {"errors": None, "warnings": None, "issues_total": None}, "structure": {}},
            "simulation": {"enabled": False, "nav2_result": "UNKNOWN", "mission_success": False, "duration_s": None, "recovery_events": None, "replan_events": None, "distance_m": None},
        }

        if not parsed.ok or not parsed.steps:
            # Still write llm_steps.json as empty to keep run structure stable
            write_json(rp.llm_steps_json, [])
            write_json(
                rp.validation_report_json,
                {"ok": False, "issues": [{"level": "error", "code": "steps_invalid", "message": parsed.error_message}]},
            )
            metrics["timestamps"]["run_finished_at"] = now_iso_z()
            write_json(rp.metrics_json, metrics)
            print(f"[{run_id}] steps invalid; wrote run artifacts.")
            continue

        write_json(rp.llm_steps_json, parsed.steps)

        # Build BT XML
        xml_tree = build_bt_xml(steps_from_dicts(parsed.steps), catalog=catalog)
        xml_tree.write(rp.generated_bt_xml, encoding="utf-8", xml_declaration=False)

        # Validate BT XML
        report = validate_bt_xml(
            xml_path=rp.generated_bt_xml,
            strict_attrs=bool(args.strict_attrs),
            strict_blackboard=bool(args.strict_blackboard),
        )
        write_json(rp.validation_report_json, report)

        ok = bool(report.get("ok"))
        metrics["bt"]["xml_valid"] = ok
        summary = report.get("summary") or {}
        if isinstance(summary, dict):
            metrics["bt"]["validator"] = {
                "errors": summary.get("errors"),
                "warnings": summary.get("warnings"),
                "issues_total": summary.get("issues_total"),
            }

        # Structure metrics
        struct = compute_bt_structure_metrics(rp.generated_bt_xml)
        metrics["bt"]["structure"] = struct

        metrics["timestamps"]["run_finished_at"] = now_iso_z()
        write_json(rp.metrics_json, metrics)
        print(f"[{run_id}] xml_valid={ok} wrote: {rp.run_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

