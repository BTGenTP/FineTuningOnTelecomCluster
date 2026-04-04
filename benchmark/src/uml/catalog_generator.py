from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..constraints.loader import load_constraints
from .xmi_parser import UmlOperation, infer_bt_tags_from_behaviortreeschema, parse_skills_uml, sha256_path


def _map_type_href(type_href: Optional[str]) -> str:
    if not type_href:
        return "string"
    low = type_href.lower()
    if "float" in low or "float64" in low or "double" in low:
        return "float"
    if "bool" in low:
        return "bool"
    if "uint" in low or "int" in low:
        return "integer"
    if "string" in low:
        return "string"
    return "string"


def _apply_enums(skill_id: str, port_name: str, enums: Mapping[str, Any]) -> Optional[str]:
    key = f"{skill_id}.{port_name}"
    enum_spec = (enums.get("enums") or {}).get(key)
    if isinstance(enum_spec, Mapping) and "values" in enum_spec:
        values = [str(v) for v in (enum_spec.get("values") or [])]
        if values:
            return "enum:" + "|".join(values)
    return None


def _tag_override(skill_id: str, enums: Mapping[str, Any]) -> Optional[str]:
    overrides = enums.get("tag_overrides") or {}
    if isinstance(overrides, Mapping) and skill_id in overrides:
        return str(overrides[skill_id])
    return None


def build_catalog_from_uml(
    *,
    skills_uml_paths: Iterable[Path],
    behaviortreeschema_paths: Iterable[Path],
    constraints_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    constraints = load_constraints(constraints_dir)
    inferred_tags = infer_bt_tags_from_behaviortreeschema(list(behaviortreeschema_paths))

    atomic: List[Dict[str, Any]] = []
    source_files: List[Dict[str, str]] = []

    for p in skills_uml_paths:
        source_files.append({"path": str(p), "sha256": sha256_path(p)})
        for op in parse_skills_uml(p):
            skill_id = op.name
            bt_tag = _tag_override(skill_id, constraints.enums) or inferred_tags.get(skill_id) or "Action"

            # Map UML params into a “ports” representation.
            attributes: Dict[str, str] = {"ID": f"const:{skill_id}", "name": "string"}
            required: List[str] = ["ID"]
            bb_inputs: List[str] = []
            bb_outputs: List[str] = []

            for prm in op.parameters:
                if prm.direction == "return":
                    continue
                if prm.name in {"run", "ok", "fail"}:
                    continue
                enum_type = _apply_enums(skill_id, prm.name, constraints.enums)
                prim_type = enum_type or _map_type_href(prm.type_href)

                # Heuristic: out params are blackboard outputs; other params are inputs.
                if prm.direction == "out":
                    attributes[prm.name] = "blackboard_port_output"
                    required.append(prm.name)
                    bb_outputs.append(prm.name)
                else:
                    # Inputs are usually blackboard ports in BT XML; keep primitive typing in constraints/enums.
                    attributes[prm.name] = "blackboard_port_input" if prim_type in {"integer", "float", "bool", "string"} else "blackboard_port_input"
                    required.append(prm.name)
                    bb_inputs.append(prm.name)

            atomic.append(
                {
                    "id": skill_id,
                    "bt_tag": bt_tag,
                    "node_type": bt_tag,
                    "attributes": attributes,
                    "required_attributes": sorted(set(required)),
                    "blackboard_inputs": sorted(set(bb_inputs)),
                    "blackboard_outputs": sorted(set(bb_outputs)),
                    "semantic_description": op.description,
                    "provenance": {"skills_uml": str(p)},
                }
            )

    catalog = {
        "version": "v4",
        "domain": "NAV4RAIL",
        "description": "Generated from Papyrus Robotics UML (skills) + behaviortreeschema (bt_tag inference).",
        "generated_at_utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "source_files": source_files,
        "atomic_skills": sorted(atomic, key=lambda x: x["id"]),
    }
    return catalog


def write_catalog(path: str | Path, catalog: Mapping[str, Any]) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(catalog), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out

