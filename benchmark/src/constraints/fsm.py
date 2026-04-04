from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from xml.etree import ElementTree as ET


@dataclass(frozen=True, slots=True)
class FsmIssue:
    code: str
    message: str
    state: str
    event: str


def _iter_skill_events(root: ET.Element) -> Iterable[str]:
    for node in root.iter():
        if node.tag in {"Action", "Condition"} and "ID" in node.attrib:
            yield str(node.attrib["ID"])


def evaluate_fsm(root: ET.Element, fsm_spec: Mapping[str, Any], *, satisfied_patterns: Set[str]) -> tuple[bool, str, List[FsmIssue]]:
    machines = fsm_spec.get("machines") or {}
    machine = machines.get("Nav4RailMissionFSM") or {}
    initial_state = str(machine.get("initial_state", "Start"))
    transitions = list(machine.get("transitions") or [])
    illegal = list(machine.get("illegal_if_missing") or [])

    # Build transition maps
    by_skill: Dict[str, List[Tuple[str, str]]] = {}
    by_pattern: Dict[str, List[Tuple[str, str]]] = {}
    for t in transitions:
        if not isinstance(t, Mapping):
            continue
        src = str(t.get("from", ""))
        dst = str(t.get("to", ""))
        if "on_skill" in t:
            by_skill.setdefault(str(t["on_skill"]), []).append((src, dst))
        if "on_pattern" in t:
            by_pattern.setdefault(str(t["on_pattern"]), []).append((src, dst))

    states: Set[str] = {initial_state}
    issues: List[FsmIssue] = []

    def apply_pattern_closure(current: Set[str]) -> Set[str]:
        out = set(current)
        changed = True
        while changed:
            changed = False
            for patt in satisfied_patterns:
                for src, dst in by_pattern.get(patt, []):
                    if src in out and dst not in out:
                        out.add(dst)
                        changed = True
        return out

    states = apply_pattern_closure(states)

    for event in _iter_skill_events(root):
        candidates = by_skill.get(event, [])
        if not candidates:
            continue
        next_states: Set[str] = set(states)
        progressed = False
        for src, dst in candidates:
            if src in states:
                next_states.add(dst)
                progressed = True
        if not progressed:
            # No valid transition from any current state.
            sample_state = sorted(states)[0] if states else initial_state
            issues.append(
                FsmIssue(
                    code="fsm_illegal_transition",
                    message=f"Event {event} not allowed from any of states={sorted(states)}.",
                    state=sample_state,
                    event=event,
                )
            )
        states = apply_pattern_closure(next_states)

    # Post-check illegal rules
    final_state = sorted(states)[0] if states else initial_state
    for rule in illegal:
        if not isinstance(rule, Mapping):
            continue
        required_state = str(rule.get("required_state", ""))
        before_skill = str(rule.get("before_skill", ""))
        msg = str(rule.get("message", ""))
        if before_skill and any(e == before_skill for e in _iter_skill_events(root)):
            if required_state not in states:
                issues.append(
                    FsmIssue(
                        code="fsm_prereq_missing",
                        message=msg or f"Missing prereq state {required_state} before {before_skill}.",
                        state=final_state,
                        event=before_skill,
                    )
                )

    ok = len([i for i in issues if i.code == "fsm_illegal_transition"]) == 0
    return ok, final_state, issues

