from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge base NAV4RAIL catalog with UML-derived atomic skills overlay.")
    p.add_argument("--base", default="data/nav4rail_catalog.json", help="Base catalog (has control nodes + rules).")
    p.add_argument("--uml", default="data/nav4rail_skills_from_uml.json", help="UML-derived skills snapshot.")
    p.add_argument("--output", default="data/nav4rail_catalog_merged.json", help="Output merged catalog path.")
    p.add_argument("--prefer-strict", action="store_true", help="Prefer stricter attribute typing when conflicting.")
    return p.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def _skill_map(catalog: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for s in (catalog.get("atomic_skills") or []):
        if isinstance(s, Mapping) and "id" in s:
            out[str(s["id"])] = dict(s)
    return out


def _strictness_rank(attr_type: str) -> int:
    """
    Higher is stricter.
    We keep this deliberately simple to avoid accidental relaxations.
    """
    t = (attr_type or "").strip()
    if t.startswith("const:"):
        return 5
    if t.startswith("enum:"):
        return 4
    if t in {"integer", "float", "bool"}:
        return 3
    if t in {"blackboard_port_input", "blackboard_port_output"}:
        return 2
    if t == "string_or_blackboard":
        return 1
    if t == "string":
        return 0
    return 0


@dataclass(slots=True)
class MergeStats:
    base_skills: int = 0
    uml_skills: int = 0
    updated: int = 0
    added_from_uml: int = 0


def merge_catalogs(base: Mapping[str, Any], uml: Mapping[str, Any], *, prefer_strict: bool) -> tuple[dict[str, Any], MergeStats]:
    base_out = dict(base)
    base_skills = _skill_map(base)
    uml_skills = _skill_map(uml)

    stats = MergeStats(base_skills=len(base_skills), uml_skills=len(uml_skills))

    merged_atomic: list[dict[str, Any]] = []
    for skill_id, base_skill in sorted(base_skills.items()):
        overlay = uml_skills.get(skill_id)
        if not overlay:
            merged_atomic.append(base_skill)
            continue

        merged = dict(base_skill)

        # Prefer UML semantic_description if it is non-empty (often richer).
        uml_desc = str(overlay.get("semantic_description") or "").strip()
        if uml_desc:
            merged["semantic_description"] = uml_desc

        # Attributes: keep base (benchmark contracts) as canonical, but add missing attrs from UML.
        base_attrs = dict(base_skill.get("attributes", {}))
        uml_attrs = dict(overlay.get("attributes", {}))
        for attr_name, uml_type in uml_attrs.items():
            if attr_name not in base_attrs:
                base_attrs[attr_name] = uml_type
            elif prefer_strict:
                # Keep stricter of the two types.
                base_t = str(base_attrs[attr_name])
                uml_t = str(uml_type)
                if _strictness_rank(uml_t) > _strictness_rank(base_t):
                    base_attrs[attr_name] = uml_t
        merged["attributes"] = base_attrs

        # Required attributes: keep base as canonical.
        # Rationale: the benchmark catalog encodes the validator contracts;
        # UML snapshots can add params that are not yet used in our reference BTs.
        merged["required_attributes"] = list(base_skill.get("required_attributes", []) or [])

        # Blackboard contracts: keep base as canonical (UML snapshot cannot infer outputs reliably).
        merged_atomic.append(merged)
        stats.updated += 1

    # Skills present in UML but missing from base: add them (they will be validated but may be incomplete).
    base_ids = set(base_skills.keys())
    for skill_id, overlay in sorted(uml_skills.items()):
        if skill_id in base_ids:
            continue
        merged_atomic.append(dict(overlay))
        stats.added_from_uml += 1

    base_out["atomic_skills"] = merged_atomic
    base_out["description"] = str(base.get("description", "")).rstrip() + " (merged with UML snapshot)"
    base_out["provenance"] = {
        "base_catalog": base.get("description"),
        "uml_snapshot": {
            "generated_at_utc": uml.get("generated_at_utc"),
            "source_files": uml.get("source_files", []),
        },
        "merge_policy": {
            "semantic_description": "prefer_uml_if_non_empty",
            "attributes": "base_canonical_add_missing_from_uml" + ("_prefer_strict" if prefer_strict else ""),
            "required_attributes": "base_canonical",
            "blackboard_contracts": "base_canonical",
        },
    }
    return base_out, stats


def main() -> int:
    args = parse_args()
    base = _load(args.base)
    uml = _load(args.uml)
    merged, stats = merge_catalogs(base, uml, prefer_strict=bool(args.prefer_strict))
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote: {out}")
    print(f"base_skills={stats.base_skills} uml_skills={stats.uml_skills} updated={stats.updated} added_from_uml={stats.added_from_uml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

