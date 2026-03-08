from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore

from finetune_Nav2.catalog.catalog_io import default_catalog_path, load_catalog
from finetune_Nav2.constraints.steps_prefix_fn import build_prefix_allowed_tokens_fn
from finetune_Nav2.eval.bt_validation import compute_bt_structure_metrics, validate_bt_xml
from finetune_Nav2.eval.json_to_xml import build_bt_xml, steps_from_dicts
from finetune_Nav2.eval.run_artifacts import create_run_dir, next_run_id, now_iso_z, write_json, write_text
from finetune_Nav2.eval.steps_parsing import parse_steps_strict
from finetune_Nav2.train.model_registry import MODELS
from finetune_Nav2.train.prompting import build_chat_messages, build_mistral_inst_prompt, build_phi2_prompt


def _load_dataset_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def _make_prompt(model_key: str, tokenizer, catalog: Dict[str, Any], mission: str) -> str:
    spec = MODELS[model_key]
    if spec.chat_template:
        msgs = build_chat_messages(mission=mission, catalog=catalog)
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "\n### Steps JSON:\n"
    if model_key == "phi2":
        prompt, _ = build_phi2_prompt(mission=mission, catalog=catalog)
        return prompt
    prompt, _ = build_mistral_inst_prompt(mission=mission, catalog=catalog)
    return prompt


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HF eval: model/adpater -> steps JSON -> BT XML -> validate -> runs/.")
    p.add_argument("--model-key", choices=sorted(MODELS.keys()), required=True)
    p.add_argument("--adapter-dir", type=str, required=True, help="Path to LoRA adapter directory.")
    p.add_argument("--dataset", type=str, required=True, help="Dataset JSONL path (missions).")
    p.add_argument("--catalog", type=str, default=str(default_catalog_path()))
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--constrained", choices=["off", "jsonschema"], default="off")
    p.add_argument("--strict-attrs", action="store_true")
    p.add_argument("--strict-blackboard", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    spec = MODELS[str(args.model_key)]

    dataset_path = Path(args.dataset).expanduser().resolve()
    rows = _load_dataset_jsonl(dataset_path)
    if not rows:
        raise SystemExit(f"Empty dataset: {dataset_path}")

    # Deterministic sampling
    import random

    random.seed(42)
    random.shuffle(rows)
    rows = rows[: int(args.n)]

    catalog = load_catalog(args.catalog)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, Path(args.adapter_dir).expanduser().resolve())
    model.eval()

    prefix_fn = None
    if args.constrained == "jsonschema":
        prefix_fn = build_prefix_allowed_tokens_fn(tokenizer, catalog)

    for row in rows:
        mission = str(row.get("mission") or "").strip()
        if not mission:
            continue

        prompt = _make_prompt(str(args.model_key), tokenizer, catalog, mission)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        t0 = time.perf_counter()
        out = model.generate(
            **inputs,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=bool(float(args.temperature) > 0.0),
            temperature=float(args.temperature) if float(args.temperature) > 0.0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=prefix_fn,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000.0)

        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        llm_raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        parsed = parse_steps_strict(llm_raw, catalog)

        run_id = next_run_id()
        rp = create_run_dir(run_id)

        experiment = {
            "id": run_id,
            "llm": {
                "provider": "hf_local_peft",
                "api_base": None,
                "model": spec.hf_id,
                "temperature": float(args.temperature),
                "adapter_dir": str(Path(args.adapter_dir).expanduser().resolve()),
                "constrained": str(args.constrained),
            },
            "generator": {
                "mode": "fail-fast",
                "constraints": {"enabled": args.constrained != "off", "kind": str(args.constrained)},
            },
            "validation": {
                "strict_attrs": bool(args.strict_attrs),
                "strict_blackboard": bool(args.strict_blackboard),
            },
            "inputs": {"dataset": str(dataset_path)},
        }

        write_text(rp.mission_txt, mission)
        write_json(rp.experiment_json, experiment)
        write_text(rp.prompt_rendered_txt, prompt)
        write_text(rp.llm_steps_raw_txt, llm_raw)

        metrics: Dict[str, Any] = {
            "schema_version": "0.1",
            "run_id": run_id,
            "timestamps": {"run_started_at": now_iso_z(), "run_finished_at": None},
            "llm": {
                "provider": "hf_local_peft",
                "api_base": None,
                "model": spec.hf_id,
                "temperature": float(args.temperature),
                "latency_ms": latency_ms,
                "tokens": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                "errors": parsed.errors,
            },
            "bt": {"xml_valid": False, "validator": {"errors": None, "warnings": None, "issues_total": None}, "structure": {}},
            "simulation": {"enabled": False, "nav2_result": "UNKNOWN", "mission_success": False, "duration_s": None, "recovery_events": None, "replan_events": None, "distance_m": None},
        }

        if not parsed.ok or not parsed.steps:
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

        xml_tree = build_bt_xml(steps_from_dicts(parsed.steps), catalog=catalog)
        xml_tree.write(rp.generated_bt_xml, encoding="utf-8", xml_declaration=False)

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
        metrics["bt"]["structure"] = compute_bt_structure_metrics(rp.generated_bt_xml)
        metrics["timestamps"]["run_finished_at"] = now_iso_z()
        write_json(rp.metrics_json, metrics)
        print(f"[{run_id}] xml_valid={ok} wrote: {rp.run_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

