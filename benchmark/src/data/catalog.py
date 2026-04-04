from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class SkillSpec:
    id: str
    bt_tag: str
    node_type: str
    attributes: Dict[str, str]
    required_attributes: list[str]
    blackboard_inputs: list[str]
    blackboard_outputs: list[str]
    semantic_description: str


def default_catalog_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "nav4rail_catalog.json"


def load_catalog(path: str | Path | None = None) -> Dict[str, Any]:
    catalog_path = Path(path).expanduser().resolve() if path else default_catalog_path()
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def iter_atomic_skills(catalog: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for skill in catalog.get("atomic_skills", []) or []:
        if isinstance(skill, Mapping):
            yield skill


def skill_map(catalog: Mapping[str, Any]) -> Dict[str, SkillSpec]:
    out: Dict[str, SkillSpec] = {}
    for raw in iter_atomic_skills(catalog):
        skill_id = str(raw["id"])
        out[skill_id] = SkillSpec(
            id=skill_id,
            bt_tag=str(raw["bt_tag"]),
            node_type=str(raw.get("node_type", raw["bt_tag"])),
            attributes=dict(raw.get("attributes", {})),
            required_attributes=list(raw.get("required_attributes", [])),
            blackboard_inputs=list(raw.get("blackboard_inputs", [])),
            blackboard_outputs=list(raw.get("blackboard_outputs", [])),
            semantic_description=str(raw.get("semantic_description", "")),
        )
    return out


def control_nodes(catalog: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    nodes: Dict[str, Dict[str, Any]] = {}
    for node in catalog.get("control_nodes", []) or []:
        if isinstance(node, Mapping) and "bt_tag" in node:
            nodes[str(node["bt_tag"])] = dict(node)
    return nodes


def allowed_skill_ids(catalog: Mapping[str, Any]) -> set[str]:
    return set(skill_map(catalog).keys())


def required_attributes_by_skill(catalog: Mapping[str, Any]) -> Dict[str, set[str]]:
    return {skill_id: set(spec.required_attributes) for skill_id, spec in skill_map(catalog).items()}


def blackboard_contract(catalog: Mapping[str, Any]) -> Dict[str, Dict[str, set[str]]]:
    contract: Dict[str, Dict[str, set[str]]] = {}
    for skill_id, spec in skill_map(catalog).items():
        contract[skill_id] = {
            "inputs": set(spec.blackboard_inputs),
            "outputs": set(spec.blackboard_outputs),
        }
    return contract


def nav4rail_system_rules(catalog: Mapping[str, Any]) -> list[str]:
    return [str(rule) for rule in catalog.get("system_rules", []) or []]
