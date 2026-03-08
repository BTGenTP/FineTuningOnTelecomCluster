from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from finetune_Nav2.catalog.catalog_io import default_catalog_path, load_catalog
from finetune_Nav2.eval.json_to_xml import build_bt_xml, steps_from_dicts


def _load_steps_dataset(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


SYSTEM_PROMPT = (
    "Tu es un assistant spécialisé Nav2 / BehaviorTree.CPP.\n"
    "Ta tâche: générer un Behavior Tree XML compatible Nav2.\n"
    "\n"
    "Règles STRICTES:\n"
    "- La sortie doit être UNIQUEMENT du XML (aucun markdown, aucun texte).\n"
    "- Structure: <root main_tree_to_execute=\"MainTree\"> puis <BehaviorTree ID=\"MainTree\">.\n"
    "- N'utilise que des tags autorisés par le catalogue.\n"
)


def build_prompt_mistral(mission: str) -> str:
    instruction = f"{SYSTEM_PROMPT}\nMission: {mission}\n\n### BT XML:\n"
    return f"<s>[INST] {instruction} [/INST]\n"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate XML-direct dataset from steps dataset (oracle conversion).")
    p.add_argument("--steps-dataset", type=str, required=True, help="Input steps dataset JSONL.")
    p.add_argument("--out", type=str, required=True, help="Output XML-direct dataset JSONL.")
    p.add_argument("--catalog", type=str, default=str(default_catalog_path()))
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    inp = Path(args.steps_dataset).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(args.catalog)
    rows = _load_steps_dataset(inp)
    if args.limit is not None:
        rows = rows[: int(args.limit)]

    wrote = 0
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            mission = str(r.get("mission") or "").strip()
            steps = r.get("steps")
            if not mission or not isinstance(steps, list):
                continue
            xml_tree = build_bt_xml(steps_from_dicts(steps), catalog=catalog)
            import io

            buf = io.BytesIO()
            xml_tree.write(buf, encoding="utf-8", xml_declaration=False)
            xml = buf.getvalue().decode("utf-8")

            obj = {
                "mission": mission,
                "xml": xml,
                "prompt": build_prompt_mistral(mission) + xml + " </s>",
                "meta": {"source": "oracle_steps_to_xml", "dataset_version": "nav2_xml_direct_v0"},
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"Wrote XML-direct dataset: {out} ({wrote} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

