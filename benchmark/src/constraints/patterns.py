from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from xml.etree import ElementTree as ET


@dataclass(frozen=True, slots=True)
class PatternFinding:
    pattern: str
    ok: bool
    code: str
    details: str


def _iter_skill_nodes(root: ET.Element) -> Iterable[ET.Element]:
    for node in root.iter():
        if node.tag in {"Action", "Condition"} and "ID" in node.attrib:
            yield node


def _match_node(node: ET.Element, spec: Mapping[str, Any]) -> bool:
    tag = spec.get("tag")
    if tag and node.tag != tag:
        return False
    sid = spec.get("id")
    if sid and node.attrib.get("ID") != sid:
        return False
    any_ids = spec.get("id_any_of")
    if any_ids and node.attrib.get("ID") not in set(any_ids):
        return False
    attrs = spec.get("attrs") or {}
    if isinstance(attrs, Mapping):
        for k, v in attrs.items():
            if node.attrib.get(str(k)) != str(v):
                return False
    return True


def _contains_subsequence(skill_nodes: Sequence[ET.Element], subseq_specs: Sequence[Mapping[str, Any]]) -> bool:
    if not subseq_specs:
        return True
    i = 0
    for node in skill_nodes:
        if _match_node(node, subseq_specs[i]):
            i += 1
            if i == len(subseq_specs):
                return True
    return False


def check_preparation_pattern(root: ET.Element, patterns: Mapping[str, Any]) -> PatternFinding:
    spec = (patterns.get("patterns") or {}).get("PreparationPattern") or {}
    required_seq = spec.get("required_sequence") or []
    if not required_seq:
        return PatternFinding("PreparationPattern", True, "preparation_pattern_skipped", "No required_sequence specified.")
    # Scope: preparation-like BT IDs used in ground-truth missions; else global.
    scoped = root
    prep_ids = {"preparation", "base_preparation", "real_preparation", "get_mission", "Get_mission"}
    for bt in root.findall("BehaviorTree"):
        bid = bt.attrib.get("ID") or ""
        if bid in prep_ids and list(bt):
            scoped = bt
            break
    skill_nodes = list(_iter_skill_nodes(scoped))
    ok = _contains_subsequence(skill_nodes, required_seq)
    return PatternFinding(
        "PreparationPattern",
        ok,
        "nav4rail_preparation_pattern" if not ok else "preparation_pattern_ok",
        "Required subsequence found." if ok else "Missing required preparation subsequence.",
    )


def check_execution_loop_pattern(root: ET.Element, patterns: Mapping[str, Any]) -> PatternFinding:
    spec = (patterns.get("patterns") or {}).get("ExecutionLoopPattern") or {}
    root_spec = spec.get("root") or {}
    tag = root_spec.get("tag", "ReactiveFallback")
    # Scope to execute subtree when present.
    scoped = root
    for bt in root.findall("BehaviorTree"):
        if bt.attrib.get("ID") == "execute" and list(bt):
            scoped = bt
            break
    rf_nodes = [n for n in scoped.iter(tag)]
    if not rf_nodes:
        return PatternFinding("ExecutionLoopPattern", False, "nav4rail_execution_loop", f"Missing <{tag}> execution root.")
    for rf in rf_nodes:
        children = list(rf)
        required_children = root_spec.get("required_children") or []
        matched = 0
        for child_spec in required_children:
            if child_spec.get("tag") == "Condition" and child_spec.get("id"):
                if any(c.tag == "Condition" and c.attrib.get("ID") == child_spec["id"] for c in children):
                    matched += 1
            elif child_spec.get("tag") == "Repeat":
                attrs = (child_spec.get("attrs") or {})
                if any(c.tag == "Repeat" and all(c.attrib.get(k) == str(v) for k, v in attrs.items()) for c in children):
                    matched += 1
        if matched == len(required_children):
            return PatternFinding("ExecutionLoopPattern", True, "execution_loop_ok", "Execution loop root satisfies required children.")
    return PatternFinding("ExecutionLoopPattern", False, "nav4rail_execution_loop", "No ReactiveFallback satisfies Repeat(-1) + MissionTerminated.")


def check_inspection_pattern(root: ET.Element, patterns: Mapping[str, Any]) -> PatternFinding:
    spec = (patterns.get("patterns") or {}).get("InspectionPattern") or {}
    trigger = spec.get("trigger") or {}
    values = set(trigger.get("values") or [])
    has_trigger = any(
        n.tag == trigger.get("tag", "Condition")
        and n.attrib.get("ID") == trigger.get("id", "CheckCurrentStepType")
        and n.attrib.get(trigger.get("attr", "type_to_be_checked")) in values
        for n in _iter_skill_nodes(root)
    )
    if not has_trigger:
        return PatternFinding("InspectionPattern", True, "inspection_not_applicable", "No inspection trigger present (pattern not applicable).")

    required_subsequence = spec.get("required_subsequence") or []
    skill_nodes = list(_iter_skill_nodes(root))
    ids = [n.attrib.get("ID", "") for n in skill_nodes]
    if "AnalyseMeasurements" not in set(ids):
        return PatternFinding("InspectionPattern", False, "inspection_requires_analysis", "Inspection subtree is missing AnalyseMeasurements.")
    if not _contains_subsequence(skill_nodes, required_subsequence):
        return PatternFinding("InspectionPattern", False, "inspection_subsequence_missing", "Missing required inspection subsequence.")

    fb_spec = spec.get("required_fallback") or {}
    if not fb_spec or fb_spec.get("tag") != "Fallback":
        return PatternFinding("InspectionPattern", True, "inspection_ok", "Inspection subsequence satisfied (fallback check disabled).")

    fallbacks = [n for n in root.iter("Fallback")]
    if not fallbacks:
        return PatternFinding("InspectionPattern", False, "inspection_requires_fallback", "Missing Fallback for inspection recovery.")
    must_contain_any = fb_spec.get("must_contain_any") or []
    corrective_pair = set(n.get("id") for n in fb_spec.get("corrective_branch_must_contain") or [])
    for fb in fallbacks:
        ids = {n.attrib.get("ID") for n in fb.iter() if n.tag in {"Action", "Condition"}}
        any_ok = any((spec_item.get("id") in ids) for spec_item in must_contain_any if isinstance(spec_item, Mapping))
        pair_ok = corrective_pair.issubset(ids)
        if any_ok and pair_ok:
            return PatternFinding("InspectionPattern", True, "inspection_ok", "Inspection recovery fallback satisfied.")
    return PatternFinding("InspectionPattern", False, "inspection_requires_fallback", "Inspection recovery fallback is incomplete.")


def evaluate_patterns(root: ET.Element, patterns: Mapping[str, Any]) -> List[PatternFinding]:
    return [
        check_preparation_pattern(root, patterns),
        check_execution_loop_pattern(root, patterns),
        check_inspection_pattern(root, patterns),
    ]

