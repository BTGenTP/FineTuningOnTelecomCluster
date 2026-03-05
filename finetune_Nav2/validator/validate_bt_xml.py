#!/usr/bin/env python3
# flake8: noqa
"""
Static post-generation validator for Nav2 / BehaviorTree.CPP XML (vendored for finetune_Nav2).

Design goals:
- Self-contained: no dependency on external repository files
- Strict allowlist: tags/attrs must come from the local catalog + (optional) local reference BTs
- Useful errors for debugging training/inference
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET


SCRIPT_DIR = Path(__file__).resolve().parent
FINETUNE_NAV2_ROOT = SCRIPT_DIR.parent

DEFAULT_CATALOG_PATH = FINETUNE_NAV2_ROOT / "catalog" / "bt_nodes_catalog.json"
DEFAULT_REFERENCE_DIR = FINETUNE_NAV2_ROOT / "reference_behavior_trees"

BB_VAR_RE = re.compile(r"\{([^{}]+)\}")


# Minimal known blackboard directions for common Nav2 nodes (heuristic).
# Values: "in" / "out" / "inout".
KNOWN_PORT_DIRECTIONS: Dict[str, Dict[str, str]] = {
    "ComputePathToPose": {"goal": "in", "path": "out"},
    "ComputePathThroughPoses": {"goals": "in", "path": "out"},
    "FollowPath": {"path": "in"},
    "GoalUpdater": {"input_goal": "in", "output_goal": "out"},
    "TruncatePath": {"input_path": "in", "output_path": "out"},
    "RemovePassedGoals": {"input_goals": "in", "output_goals": "inout"},
}

CONTROL_ATTR_TYPES: Dict[str, Dict[str, str]] = {
    # Best-effort typing for common BT control node attributes.
    "RateController": {"hz": "float"},
    "DistanceController": {"distance": "float"},
    "SpeedController": {"max_speed": "float"},
    "Repeat": {"num_cycles": "int"},
    "RecoveryNode": {"number_of_retries": "int"},
}


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    code: str
    message: str
    file: Optional[str] = None
    xpath: Optional[str] = None
    tag: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"level": self.level, "code": self.code, "message": self.message}
        if self.file:
            out["file"] = self.file
        if self.xpath:
            out["xpath"] = self.xpath
        if self.tag:
            out["tag"] = self.tag
        return out


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _print_checklist(report: Dict[str, Any]) -> None:
    import sys

    issues = report.get("issues") or []
    if not isinstance(issues, list):
        issues = []

    def has(code: str, *, level: Optional[str] = None) -> bool:
        for it in issues:
            if not isinstance(it, dict):
                continue
            if it.get("code") != code:
                continue
            if level is None or it.get("level") == level:
                return True
        return False

    def line(ok: bool, label: str) -> str:
        return f"{'✅' if ok else '❌'} {label}"

    checks = [
        (not has("xml_parse", level="error"), "XML parse (well-formed)"),
        (not has("root_tag", level="error"), "Root tag == <root>"),
        (not has("root_main_tree", level="error"), "root@main_tree_to_execute present"),
        (not has("missing_main_tree_def", level="error"), "MainTree definition exists"),
        (not has("bt_missing_id", level="error"), "All <BehaviorTree> have ID"),
        (not has("bt_duplicate_id", level="error"), "BehaviorTree IDs unique"),
        (not has("subtree_missing_definition", level="error"), "All SubTree IDs defined"),
        (not has("subtree_cycle", level="error"), "No SubTree cycles"),
        (not has("tag_not_allowed", level="error"), "All tags allowed (catalog + refs)"),
        (not has("missing_required_attr", level="error"), "Required attributes present"),
        (not has("unknown_attr", level="error"), "No unknown attributes (strict mode)"),
        (not has("empty_control_node", level="error"), "No empty control nodes"),
        (not has("blackboard_unproduced", level="error"), "Blackboard vars consistent (strict mode)"),
    ]

    sys.stderr.write("\n[validate_bt_xml] Checklist\n")
    for ok, label in checks:
        sys.stderr.write(f"- {line(bool(ok), label)}\n")


def _catalog_allowlist(catalog: Dict[str, Any]) -> Tuple[Set[str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    allowed_tags: Set[str] = set()
    allowed_attrs_by_tag: Dict[str, Set[str]] = {}
    required_attrs_by_tag: Dict[str, Set[str]] = {}

    # Control nodes
    for item in catalog.get("control_nodes_allowed", []) or []:
        tag = item.get("bt_tag")
        attrs = item.get("attributes") or []
        if isinstance(tag, str) and tag:
            allowed_tags.add(tag)
            allowed_attrs_by_tag.setdefault(tag, set()).update({a for a in attrs if isinstance(a, str)})

    # Atomic skills
    for item in catalog.get("atomic_skills", []) or []:
        tag = item.get("bt_tag")
        input_ports = item.get("input_ports", {}) or {}
        output_ports = item.get("output_ports", {}) or {}
        if not (isinstance(tag, str) and tag):
            continue

        allowed_tags.add(tag)
        attrs: Set[str] = set()
        req: Set[str] = set()

        for k, v in dict(input_ports).items():
            if not isinstance(k, str):
                continue
            if k in ("ID", "__shared_blackboard"):
                continue
            attrs.add(k)
            is_optional = isinstance(v, str) and ("optional" in v.lower())
            if not is_optional:
                req.add(k)

        for k in dict(output_ports).keys():
            if isinstance(k, str) and k:
                attrs.add(k)

        allowed_attrs_by_tag.setdefault(tag, set()).update(attrs)
        if req:
            required_attrs_by_tag.setdefault(tag, set()).update(req)

    # Base BT tags always allowed
    for base in ("root", "BehaviorTree", "SubTree"):
        allowed_tags.add(base)
        allowed_attrs_by_tag.setdefault(base, set())

    # Known required attrs for base tags
    required_attrs_by_tag.setdefault("root", set()).add("main_tree_to_execute")
    required_attrs_by_tag.setdefault("BehaviorTree", set()).add("ID")

    # SubTree commonly uses these attrs
    allowed_attrs_by_tag.setdefault("SubTree", set()).update({"ID", "__shared_blackboard"})
    required_attrs_by_tag.setdefault("SubTree", set()).add("ID")

    return allowed_tags, allowed_attrs_by_tag, required_attrs_by_tag


def _infer_port_type(desc: Any) -> str:
    if not isinstance(desc, str):
        return "string"
    d = desc.lower()
    if "bool" in d:
        return "bool"
    if "int" in d:
        return "int"
    if "float" in d or "double" in d:
        return "float"
    return "string"


def _catalog_attr_types(catalog: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {
        "root": {"main_tree_to_execute": "string"},
        "BehaviorTree": {"ID": "string"},
        "SubTree": {"ID": "string", "__shared_blackboard": "bool"},
    }

    for item in catalog.get("atomic_skills", []) or []:
        tag = item.get("bt_tag")
        if not (isinstance(tag, str) and tag):
            continue
        out.setdefault(tag, {})
        input_ports = item.get("input_ports", {}) or {}
        for k, v in dict(input_ports).items():
            if not isinstance(k, str):
                continue
            if k in ("ID", "__shared_blackboard"):
                continue
            out[tag][k] = _infer_port_type(v)

    for tag, attrs in CONTROL_ATTR_TYPES.items():
        out.setdefault(tag, {}).update(attrs)

    return out


def _is_blackboard_value(v: str) -> bool:
    if not v:
        return False
    return BB_VAR_RE.search(v) is not None


def _check_type(expected: str, value: str) -> bool:
    v = (value or "").strip()
    if expected == "bool":
        return v.lower() in ("true", "false")
    if expected == "int":
        return re.fullmatch(r"-?\d+", v) is not None
    if expected == "float":
        try:
            float(v)
            return True
        except Exception:
            return False
    return True


def _scan_reference_allowlist(reference_dir: Path) -> Tuple[Set[str], Dict[str, Set[str]], Set[str]]:
    tags: Set[str] = set()
    attrs_by_tag: Dict[str, Set[str]] = {}
    bb_vars: Set[str] = set()

    for xml_path in sorted(reference_dir.rglob("*.xml")):
        try:
            tree = ET.parse(str(xml_path))
        except Exception:
            continue
        root = tree.getroot()
        for el in root.iter():
            tags.add(el.tag)
            attrs_by_tag.setdefault(el.tag, set()).update(el.attrib.keys())
            for v in el.attrib.values():
                for var in BB_VAR_RE.findall(v or ""):
                    bb_vars.add(var)

    return tags, attrs_by_tag, bb_vars


def _xpath_of(el: ET.Element, parent_map: Dict[ET.Element, ET.Element]) -> str:
    parts: List[str] = []
    cur: Optional[ET.Element] = el
    while cur is not None:
        p = parent_map.get(cur)
        if p is None:
            parts.append(f"/{cur.tag}")
            break
        same = [c for c in list(p) if c.tag == cur.tag]
        idx = same.index(cur) + 1
        parts.append(f"/{cur.tag}[{idx}]")
        cur = p
    return "".join(reversed(parts))


def _build_parent_map(root: ET.Element) -> Dict[ET.Element, ET.Element]:
    parent: Dict[ET.Element, ET.Element] = {}
    for p in root.iter():
        for c in list(p):
            parent[c] = p
    return parent


def _collect_bt_definitions(root: ET.Element) -> Dict[str, ET.Element]:
    out: Dict[str, ET.Element] = {}
    for bt in root.findall("BehaviorTree"):
        bt_id = bt.get("ID")
        if bt_id:
            out[bt_id] = bt
    return out


def _collect_subtree_refs(root: ET.Element) -> Set[str]:
    refs: Set[str] = set()
    for st in root.iter("SubTree"):
        sid = st.get("ID")
        if sid:
            refs.add(sid)
    return refs


def _detect_cycles(bt_defs: Dict[str, ET.Element]) -> List[List[str]]:
    graph: Dict[str, Set[str]] = {}
    for bt_id, bt_el in bt_defs.items():
        graph[bt_id] = set()
        for st in bt_el.iter("SubTree"):
            sid = st.get("ID")
            if sid:
                graph[bt_id].add(sid)

    visited: Set[str] = set()
    stack: Set[str] = set()
    path: List[str] = []
    cycles: List[List[str]] = []

    def dfs(n: str) -> None:
        visited.add(n)
        stack.add(n)
        path.append(n)
        for m in graph.get(n, set()):
            if m not in bt_defs:
                continue
            if m not in visited:
                dfs(m)
            elif m in stack:
                if m in path:
                    i = path.index(m)
                    cycles.append(path[i:] + [m])
        path.pop()
        stack.remove(n)

    for node in graph.keys():
        if node not in visited:
            dfs(node)

    return cycles


def _validate_tree(
    *,
    xml_path: Path,
    reference_dir: Optional[Path],
    catalog_path: Path,
    strict_attrs: bool,
    strict_blackboard: bool,
    external_bb_vars: Optional[List[str]] = None,
) -> Dict[str, Any]:
    issues: List[Issue] = []

    catalog = _load_json(catalog_path)
    cat_tags, cat_attrs_by_tag, required_attrs_by_tag = _catalog_allowlist(catalog)
    expected_types_by_tag = _catalog_attr_types(catalog)

    ref_tags: Set[str] = set()
    ref_attrs_by_tag: Dict[str, Set[str]] = {}
    ref_bb_vars: Set[str] = set()
    if reference_dir and reference_dir.exists():
        ref_tags, ref_attrs_by_tag, ref_bb_vars = _scan_reference_allowlist(reference_dir)

    allowed_tags = cat_tags | ref_tags
    allowed_attrs_by_tag: Dict[str, Set[str]] = {}
    for tag in allowed_tags:
        allowed_attrs_by_tag[tag] = set()
        allowed_attrs_by_tag[tag].update(cat_attrs_by_tag.get(tag, set()))
        allowed_attrs_by_tag[tag].update(ref_attrs_by_tag.get(tag, set()))

    try:
        tree = ET.parse(str(xml_path))
    except Exception as exc:
        issues.append(Issue(level="error", code="xml_parse", message=f"XML parse error: {exc}", file=str(xml_path)))
        return {"ok": False, "issues": [i.as_dict() for i in issues]}

    root = tree.getroot()
    parent_map = _build_parent_map(root)

    if root.tag != "root":
        issues.append(
            Issue(
                level="error",
                code="root_tag",
                message=f"Expected root tag <root>, got <{root.tag}>",
                file=str(xml_path),
                xpath=_xpath_of(root, parent_map),
                tag=root.tag,
            )
        )
    if not root.get("main_tree_to_execute"):
        issues.append(
            Issue(
                level="error",
                code="root_main_tree",
                message="Missing attribute root@main_tree_to_execute",
                file=str(xml_path),
                xpath=_xpath_of(root, parent_map),
                tag=root.tag,
            )
        )

    bt_defs = _collect_bt_definitions(root)
    main_id = root.get("main_tree_to_execute") or ""
    if main_id and main_id not in bt_defs:
        issues.append(
            Issue(
                level="error",
                code="missing_main_tree_def",
                message=f"main_tree_to_execute='{main_id}' has no <BehaviorTree ID='{main_id}'> definition.",
                file=str(xml_path),
                xpath=_xpath_of(root, parent_map),
            )
        )

    seen_bt_ids: Set[str] = set()
    for bt in root.findall("BehaviorTree"):
        bt_id = bt.get("ID")
        if not bt_id:
            issues.append(
                Issue(
                    level="error",
                    code="bt_missing_id",
                    message="BehaviorTree missing required attribute ID.",
                    file=str(xml_path),
                    xpath=_xpath_of(bt, parent_map),
                    tag="BehaviorTree",
                )
            )
            continue
        if bt_id in seen_bt_ids:
            issues.append(
                Issue(
                    level="error",
                    code="bt_duplicate_id",
                    message=f"Duplicate BehaviorTree ID: {bt_id}",
                    file=str(xml_path),
                    xpath=_xpath_of(bt, parent_map),
                    tag="BehaviorTree",
                )
            )
        seen_bt_ids.add(bt_id)

    subtree_refs = _collect_subtree_refs(root)
    missing_defs = sorted([sid for sid in subtree_refs if sid not in bt_defs])
    for sid in missing_defs:
        issues.append(
            Issue(
                level="error",
                code="subtree_missing_definition",
                message=f"SubTree reference ID='{sid}' has no matching <BehaviorTree ID='{sid}'> definition.",
                file=str(xml_path),
            )
        )

    cycles = _detect_cycles(bt_defs)
    for cyc in cycles:
        issues.append(
            Issue(
                level="error",
                code="subtree_cycle",
                message=f"SubTree cycle detected: {' -> '.join(cyc)}",
                file=str(xml_path),
            )
        )

    referenced = set(subtree_refs)
    unused = sorted([bid for bid in bt_defs.keys() if bid not in referenced and bid != main_id])
    for bid in unused:
        issues.append(
            Issue(
                level="warning",
                code="bt_unused_definition",
                message=f"BehaviorTree ID='{bid}' is defined but never referenced (dead subtree).",
                file=str(xml_path),
            )
        )

    for bt_id, bt_el in bt_defs.items():
        if len(list(bt_el)) == 0:
            issues.append(
                Issue(
                    level="warning",
                    code="bt_empty",
                    message=f"BehaviorTree ID='{bt_id}' has no children.",
                    file=str(xml_path),
                    tag="BehaviorTree",
                )
            )

    external_set: Set[str] = set()
    if external_bb_vars:
        for v in external_bb_vars:
            if isinstance(v, str) and v.strip():
                external_set.add(v.strip())

    produced_bb: Set[str] = set(external_set)
    consumed_bb: Set[str] = set()
    all_bb_vars: Set[str] = set()

    for el in root.iter():
        xp = _xpath_of(el, parent_map)

        if el.tag not in allowed_tags:
            issues.append(
                Issue(
                    level="error",
                    code="tag_not_allowed",
                    message=f"Tag <{el.tag}> is not in allowlist (catalog + reference BTs).",
                    file=str(xml_path),
                    xpath=xp,
                    tag=el.tag,
                )
            )
            continue

        req = required_attrs_by_tag.get(el.tag, set())
        for r in sorted(req):
            if r not in el.attrib:
                issues.append(
                    Issue(
                        level="error",
                        code="missing_required_attr",
                        message=f"Missing required attribute '{r}' on <{el.tag}>.",
                        file=str(xml_path),
                        xpath=xp,
                        tag=el.tag,
                    )
                )

        allowed_attrs = allowed_attrs_by_tag.get(el.tag, set())
        unknown = sorted([a for a in el.attrib.keys() if a not in allowed_attrs])
        if unknown:
            level = "error" if strict_attrs else "warning"
            issues.append(
                Issue(
                    level=level,
                    code="unknown_attr",
                    message=f"Unknown attribute(s) on <{el.tag}>: {', '.join(unknown)}",
                    file=str(xml_path),
                    xpath=xp,
                    tag=el.tag,
                )
            )

        exp = expected_types_by_tag.get(el.tag, {})
        for attr, value in el.attrib.items():
            if attr not in exp:
                continue
            if _is_blackboard_value(value or ""):
                continue
            expected = exp.get(attr, "string")
            if not _check_type(expected, value or ""):
                issues.append(
                    Issue(
                        level="error",
                        code="attr_type_mismatch",
                        message=f"Attribute '{attr}' on <{el.tag}> expects {expected}, got: {value!r}",
                        file=str(xml_path),
                        xpath=xp,
                        tag=el.tag,
                    )
                )

        for attr, value in el.attrib.items():
            for var in BB_VAR_RE.findall(value or ""):
                all_bb_vars.add(var)
                dir_map = KNOWN_PORT_DIRECTIONS.get(el.tag, {})
                direction = dir_map.get(attr, "in")
                if direction in ("out", "inout"):
                    produced_bb.add(var)
                if direction in ("in", "inout"):
                    consumed_bb.add(var)

    CONTROL_NODES = {
        "Sequence",
        "Fallback",
        "ReactiveSequence",
        "ReactiveFallback",
        "RoundRobin",
        "PipelineSequence",
        "RateController",
        "DistanceController",
        "SpeedController",
        "KeepRunningUntilFailure",
        "Repeat",
        "Inverter",
        "RecoveryNode",
    }
    for el in root.iter():
        if el.tag not in CONTROL_NODES:
            continue
        xp = _xpath_of(el, parent_map)
        if len(list(el)) == 0:
            issues.append(
                Issue(
                    level="error",
                    code="empty_control_node",
                    message=f"Control node <{el.tag}> has no children.",
                    file=str(xml_path),
                    xpath=xp,
                    tag=el.tag,
                )
            )
        if el.tag == "Repeat" and "num_cycles" not in el.attrib:
            issues.append(
                Issue(
                    level="warning",
                    code="repeat_unbounded",
                    message="Repeat without num_cycles may be non-terminating (warning).",
                    file=str(xml_path),
                    xpath=xp,
                    tag="Repeat",
                )
            )

    missing_bb = sorted(consumed_bb - produced_bb)
    if missing_bb:
        level = "error" if strict_blackboard else "warning"
        issues.append(
            Issue(
                level=level,
                code="blackboard_unproduced",
                message="Blackboard variable(s) consumed but never produced (heuristic): " + ", ".join(missing_bb),
                file=str(xml_path),
            )
        )

    novel_bb = sorted(all_bb_vars - ref_bb_vars) if ref_bb_vars else []
    if novel_bb:
        issues.append(
            Issue(
                level="warning",
                code="blackboard_novel_vars",
                message="Blackboard variable(s) not seen in reference BTs: " + ", ".join(novel_bb),
                file=str(xml_path),
            )
        )

    ok = not any(i.level == "error" for i in issues)
    return {
        "ok": ok,
        "file": str(xml_path),
        "blackboard": {"external_vars": sorted(list(external_set))},
        "summary": {
            "issues_total": len(issues),
            "errors": sum(1 for i in issues if i.level == "error"),
            "warnings": sum(1 for i in issues if i.level == "warning"),
        },
        "issues": [i.as_dict() for i in issues],
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a Nav2 BehaviorTree.CPP XML file (static checks).")
    p.add_argument("xml", nargs="?", type=str, help="Path to the BT XML to validate.")
    p.add_argument("--xml-path", type=str, default=None, help="Alternative to positional xml path.")
    p.add_argument(
        "--catalog",
        type=str,
        default=str(DEFAULT_CATALOG_PATH),
        help="Path to bt_nodes_catalog.json (default: finetune_Nav2/catalog/bt_nodes_catalog.json).",
    )
    p.add_argument(
        "--reference-dir",
        type=str,
        default=str(DEFAULT_REFERENCE_DIR),
        help="Directory of reference BT XMLs used to learn allowlist/blackboard vars.",
    )
    p.add_argument("--no-reference-scan", action="store_true", help="Disable reference BT scanning.")
    p.add_argument("--strict-attrs", action="store_true", help="Unknown attributes are errors (default: warnings).")
    p.add_argument(
        "--strict-blackboard",
        action="store_true",
        help="Blackboard 'consumed but not produced' becomes an error (default: warning).",
    )
    p.add_argument(
        "--external-bb",
        action="append",
        default=[],
        help="Blackboard variable name provided externally at runtime (repeatable). Example: --external-bb goal",
    )
    p.add_argument(
        "--external-bb-file",
        type=str,
        default=None,
        help="Optional file with one external blackboard var per line.",
    )
    p.add_argument("--output", "-o", type=str, default="-", help="Output report path (default: '-' for stdout).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    xml_arg = args.xml or args.xml_path
    if not xml_arg:
        raise SystemExit("Missing BT XML path. Provide positional <xml> or --xml-path <xml>.")

    xml_path = Path(xml_arg).resolve()
    catalog_path = Path(args.catalog).resolve()
    reference_dir = None if args.no_reference_scan else Path(args.reference_dir).resolve()

    external_vars: List[str] = []
    for v in (args.external_bb or [])[:200]:
        if isinstance(v, str) and v.strip():
            external_vars.append(v.strip())
    if args.external_bb_file:
        p = Path(str(args.external_bb_file)).expanduser().resolve()
        if p.exists():
            for ln in p.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("#"):
                    external_vars.append(ln)

    report = _validate_tree(
        xml_path=xml_path,
        reference_dir=reference_dir,
        catalog_path=catalog_path,
        strict_attrs=bool(args.strict_attrs),
        strict_blackboard=bool(args.strict_blackboard),
        external_bb_vars=external_vars,
    )

    _print_checklist(report)

    out_text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output == "-":
        print(out_text)
    else:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text + "\n", encoding="utf-8")
        print(f"Wrote: {out_path}")

    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

