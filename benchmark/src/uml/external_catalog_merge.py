from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping


def merge_bt_navigator_port_semantics(nav4rail_catalog: Dict[str, Any], bt_navigator_json: Path) -> Dict[str, Any]:
    """
    Enrich NAV4RAIL atomic_skills with input_ports / output_ports from BT_Navigator-style JSON
    (specific port names + type hints). Does not change validator attribute types on the base catalog.
    """
    raw = json.loads(Path(bt_navigator_json).expanduser().resolve().read_text(encoding="utf-8"))
    ext_by_id: Dict[str, Mapping[str, Any]] = {}
    for row in raw.get("atomic_skills") or []:
        if isinstance(row, Mapping) and row.get("id"):
            ext_by_id[str(row["id"])] = row

    skills = nav4rail_catalog.get("atomic_skills") or []
    for skill in skills:
        if not isinstance(skill, Mapping):
            continue
        sid = str(skill.get("id", ""))
        ext = ext_by_id.get(sid)
        if not ext:
            continue
        skill["port_semantics"] = {
            "source": "bt_navigator_catalog",
            "input_ports": dict(ext.get("input_ports") or {}),
            "output_ports": dict(ext.get("output_ports") or {}),
            "semantic_description_external": ext.get("semantic_description"),
        }

    prov = dict(nav4rail_catalog.get("provenance") or {})
    prov["bt_navigator_catalog"] = str(Path(bt_navigator_json).resolve())
    nav4rail_catalog["provenance"] = prov
    return nav4rail_catalog
