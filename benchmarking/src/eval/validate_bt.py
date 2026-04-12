"""
Validateur multi-niveaux pour Behavior Trees NAV4RAIL
================================================================================
Refactored from finetune/validate_bt.py to be catalog-driven.
All skill/port/constraint constants are loaded from skills_catalog.yaml.

Levels:
  L1 — Syntactic  : XML well-formed, valid tags/skills, MoveAndStop present
  L2 — Structural  : Non-empty control nodes, depth, Fallback branches
  L3 — Semantic    : Skill ordering, prerequisites, conditions in Fallback
  L4 — Ports       : Required ports, type checking, allowed values
  L5 — Safety      : Blackboard chaining, inspection safety, loop patterns

Score:
  1.0  -> all levels passed, no warnings
  0.5-0.9 -> valid with warnings (-0.1 per warning)
  0.0  -> invalid (blocking error)

Usage:
    from src.eval.validate_bt import validate_bt
    result = validate_bt(xml_str)
    print(result.summary())
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from src.data.skills_loader import SkillsCatalog

# ── Default catalog (loaded once, reused) ────────────────────────────────────
_DEFAULT_CATALOG: SkillsCatalog | None = None


def _get_catalog(catalog: SkillsCatalog | None = None) -> SkillsCatalog:
    global _DEFAULT_CATALOG
    if catalog is not None:
        return catalog
    if _DEFAULT_CATALOG is None:
        _DEFAULT_CATALOG = SkillsCatalog()
    return _DEFAULT_CATALOG


# ── Non-functional attributes (present on all nodes, not ports) ──────────────
_META_ATTRS = frozenset({"name", "ID"})


# ── Validation Result ────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    score: float = 1.0

    def fail(self, msg: str, level: int = 1):
        self.valid = False
        self.score = 0.0
        self.errors.append(f"[L{level}] {msg}")

    def warn(self, msg: str):
        self.warnings.append(f"[W] {msg}")
        self.score = max(0.5, self.score - 0.1)

    def summary(self) -> str:
        if self.valid and not self.warnings:
            return f"OK (score={self.score:.1f}) — L1+L2+L3+L4 passed"
        if self.valid:
            return f"OK with {len(self.warnings)} warning(s) (score={self.score:.1f})"
        return f"INVALID: {self.errors[0]}"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_skill(elem: ET.Element, valid_skills: frozenset[str]) -> str | None:
    """Resolve skill ID from XML element (v4 or v5 format)."""
    if elem.tag in ("Action", "Condition"):
        skill_id = elem.get("ID", "")
        return skill_id if skill_id in valid_skills else None
    if elem.tag in valid_skills:
        return elem.tag
    return None


def _is_condition(elem: ET.Element, condition_skills: frozenset[str]) -> str | None:
    """Return skill ID if element is a valid Condition."""
    if elem.tag == "Condition":
        skill_id = elem.get("ID", "")
        return skill_id if skill_id in condition_skills else None
    if elem.tag in condition_skills:
        return elem.tag
    return None


def _l4_label(elem: ET.Element, skill_id: str, valid_skills: frozenset[str]) -> str:
    if elem.tag in valid_skills:
        return elem.tag
    return f'{elem.tag} ID="{skill_id}"'


def _max_depth(elem: ET.Element, d: int = 0) -> int:
    children = list(elem)
    if not children:
        return d
    return max(_max_depth(c, d + 1) for c in children)


def _skills_dfs(elem: ET.Element, valid_skills: frozenset[str]) -> list[str]:
    """Return skills in DFS order (expected execution order)."""
    result = []
    skill = _resolve_skill(elem, valid_skills)
    if skill:
        result.append(skill)
    for child in elem:
        result.extend(_skills_dfs(child, valid_skills))
    return result


def _condition_in_fallback(
    root: ET.Element, condition_skills: frozenset[str]
) -> set[str]:
    """Return conditions that are descendants of a Fallback/ReactiveFallback."""
    in_fb: set[str] = set()
    for fb_tag in ("Fallback", "ReactiveFallback"):
        for fb in root.iter(fb_tag):
            for e in fb.iter():
                cond = _is_condition(e, condition_skills)
                if cond:
                    in_fb.add(cond)
    return in_fb


# ── Level 1 — Syntactic ─────────────────────────────────────────────────────


def _validate_l1(
    xml_str: str, catalog: SkillsCatalog
) -> tuple[bool, ET.Element | None, str]:
    xml_str = xml_str.strip()
    if xml_str.startswith("<?xml"):
        xml_str = xml_str[xml_str.index("?>") + 2 :].strip()

    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        return False, None, f"XML mal forme : {e}"

    if root.tag != "root":
        return False, None, f"Tag racine '<root>' attendu, obtenu '<{root.tag}>'"

    btcpp_fmt = root.get("BTCPP_format")
    _btcpp_warning = None
    if btcpp_fmt is None:
        _btcpp_warning = "Attribut BTCPP_format='4' absent sur <root> (recommande)"
    elif btcpp_fmt != "4":
        _btcpp_warning = f"BTCPP_format='{btcpp_fmt}' inattendu (attendu: '4')"

    valid_skills = catalog.valid_skills()
    valid_tags = catalog.valid_tags()
    _all_valid = valid_tags | valid_skills

    unknown_tags: set[str] = set()
    unknown_skills: set[str] = set()
    for e in root.iter():
        if e.tag not in _all_valid:
            unknown_tags.add(e.tag)
        if e.tag in ("Action", "Condition"):
            skill_id = e.get("ID", "")
            if skill_id and skill_id not in valid_skills:
                unknown_skills.add(skill_id)

    if unknown_tags:
        return False, None, f"Tags XML inconnus : {sorted(unknown_tags)}"
    if unknown_skills:
        return False, None, f"Skills inconnus (hallucinations) : {sorted(unknown_skills)}"

    has_move_and_stop = any(
        (e.tag == "Action" and e.get("ID") == "MoveAndStop") or e.tag == "MoveAndStop"
        for e in root.iter()
    )
    if not has_move_and_stop:
        return False, None, "MoveAndStop absent — le BT ne se termine jamais"

    return True, root, _btcpp_warning or "OK"


# ── Level 2 — Structural ────────────────────────────────────────────────────


def _validate_l2(root: ET.Element, result: ValidationResult, catalog: SkillsCatalog):
    control_nodes = catalog.control_nodes()
    valid_skills = catalog.valid_skills()
    limits = catalog.limits()
    max_depth = limits.get("max_depth", 12)
    max_skills = limits.get("max_skills", 80)

    if not any(e.tag == "BehaviorTree" for e in root):
        result.fail("<BehaviorTree> manquant sous <root>", level=2)
        return

    for elem in root.iter():
        if elem.tag in control_nodes and not list(elem):
            result.warn(f"<{elem.tag} name='{elem.get('name', '')}> vide (aucun enfant)")

    depth = _max_depth(root)
    if depth > max_depth:
        result.fail(f"Profondeur {depth} > {max_depth} (arbre trop imbrique)", level=2)

    skill_count = sum(
        1 for e in root.iter() if _resolve_skill(e, valid_skills) is not None
    )
    if skill_count > max_skills:
        result.warn(f"{skill_count} noeuds skills > {max_skills} (BT trop long)")

    min_fb = limits.get("min_fallback_children", 2)
    for elem in root.iter():
        if elem.tag in ("Fallback", "ReactiveFallback"):
            n = len(list(elem))
            if n < min_fb:
                result.warn(
                    f"<{elem.tag} name='{elem.get('name', '')}> n'a que {n} branche(s) "
                    f"(minimum {min_fb} requis)"
                )


# ── Level 3 — Semantic ──────────────────────────────────────────────────────


def _validate_l3(root: ET.Element, result: ValidationResult, catalog: SkillsCatalog):
    valid_skills = catalog.valid_skills()
    condition_skills = catalog.condition_skills()
    prerequisites = catalog.prerequisites()

    bt_count = sum(1 for e in root if e.tag == "BehaviorTree")
    is_multi_subtree = bt_count > 1

    skills = _skills_dfs(root, valid_skills)
    if not skills:
        return

    first: dict[str, int] = {}
    for i, s in enumerate(skills):
        if s not in first:
            first[s] = i

    if not is_multi_subtree:
        for skill, prereqs in prerequisites.items():
            if skill not in first:
                continue
            for prereq in prereqs:
                if prereq not in first:
                    result.warn(f"<{prereq}> absent alors qu'il precede normalement <{skill}>")
                elif first[prereq] > first[skill]:
                    result.warn(
                        f"Ordre incorrect : <{skill}> (pos {first[skill]}) "
                        f"avant <{prereq}> (pos {first[prereq]})"
                    )

    if "LoadMission" not in first:
        result.warn("LoadMission absent du BT")

    conditions_present: set[str] = set()
    for e in root.iter():
        cond = _is_condition(e, condition_skills)
        if cond:
            conditions_present.add(cond)
    conditions_in_fb = _condition_in_fallback(root, condition_skills)
    loop_conditions = {
        "MissionTerminated",
        "MissionFullyTreated",
        "MeasurementsQualityValidated",
        "MeasurementsEnforcedValidated",
    }
    for cond in conditions_present & loop_conditions:
        if cond not in conditions_in_fb:
            result.warn(
                f"<{cond}> hors de tout Fallback — "
                f"son signal FAILURE ne sera pas intercepte correctement"
            )


# ── Level 4 — Ports ─────────────────────────────────────────────────────────


def _validate_l4(root: ET.Element, result: ValidationResult, catalog: SkillsCatalog):
    valid_skills = catalog.valid_skills()
    skill_ports = catalog.skill_ports()

    for elem in root.iter():
        if elem.tag in ("Action", "Condition"):
            skill_id = elem.get("ID", "")
        elif elem.tag in valid_skills:
            skill_id = elem.tag
        else:
            continue
        if skill_id not in skill_ports:
            continue

        spec = skill_ports[skill_id]
        required = spec.get("required", [])
        types = spec.get("types", {})
        allowed = spec.get("allowed", {})
        node_attrs = {k: v for k, v in elem.attrib.items() if k not in _META_ATTRS}

        label = _l4_label(elem, skill_id, valid_skills)

        for port in required:
            if port not in node_attrs:
                result.warn(f'[L4] <{label}> : port requis "{port}" manquant')

        known_ports = set(required) | set(types.keys())
        for attr in node_attrs:
            if attr not in known_ports:
                result.warn(f'[L4] <{label}> : attribut inconnu "{attr}"')

        for port, ptype in types.items():
            val = node_attrs.get(port)
            if val is None:
                continue
            if ptype == "bb_var":
                if not (val.startswith("{") and val.endswith("}")):
                    result.warn(
                        f'[L4] <{label}> : port "{port}" devrait etre {{variable}}, recu "{val}"'
                    )
            elif ptype == "int_literal":
                try:
                    int(val)
                except ValueError:
                    result.warn(
                        f'[L4] <{label}> : port "{port}" devrait etre un entier, recu "{val}"'
                    )
            elif ptype == "float_literal":
                try:
                    float(val)
                except ValueError:
                    result.warn(
                        f'[L4] <{label}> : port "{port}" devrait etre un flottant, recu "{val}"'
                    )
            if port in allowed and val not in allowed[port]:
                result.warn(
                    f'[L4] <{label}> : port "{port}"="{val}" hors domaine {sorted(allowed[port])}'
                )

    for elem in root.iter("SubTreePlus"):
        if not elem.get("ID"):
            result.warn("[L4] <SubTreePlus> sans attribut ID")
        if elem.get("__autoremap") is None:
            result.warn(f'[L4] <SubTreePlus ID="{elem.get("ID", "?")}"> sans __autoremap')

    for elem in root.iter("Repeat"):
        if elem.get("num_cycles") is None:
            result.warn(f'[L4] <Repeat name="{elem.get("name", "?")}"> sans num_cycles')


# ── Level 5 — Advanced Safety ────────────────────────────────────────────────


def _validate_l5(root: ET.Element, result: ValidationResult, catalog: SkillsCatalog):
    """Advanced safety checks derived from real_inspection_mission.xml patterns."""
    valid_skills = catalog.valid_skills()

    # SR-023: Blackboard chaining — check that motion actions are preceded
    # by PassMotionParameters in the same Sequence
    _bb_producers = {"PassMotionParameters"}
    _bb_consumers = {"Move", "Deccelerate", "MoveAndStop"}
    for seq in root.iter("Sequence"):
        children_skills = []
        for child in seq:
            sid = _resolve_skill(child, valid_skills)
            if sid:
                children_skills.append(sid)
        seen_producer = False
        for sid in children_skills:
            if sid in _bb_producers:
                seen_producer = True
            elif sid in _bb_consumers and not seen_producer:
                result.warn(
                    f"[L5] <{sid}> utilise motion_params sans PassMotionParameters "
                    f"precedent dans la meme Sequence (blackboard chaining)"
                )

    # SR-025: Mission loop safety — execution subtree must use
    # ReactiveFallback(Repeat(-1)(...), MissionTerminated)
    for bt in root.iter("BehaviorTree"):
        bt_id = bt.get("ID", "")
        if bt_id in ("execute", "execution"):
            has_reactive_fb = any(e.tag == "ReactiveFallback" for e in bt.iter())
            has_repeat_neg1 = any(
                e.tag == "Repeat" and e.get("num_cycles") == "-1" for e in bt.iter()
            )
            has_mission_terminated = any(
                _resolve_skill(e, valid_skills) == "MissionTerminated"
                for e in bt.iter()
            )
            if not (has_reactive_fb and has_repeat_neg1 and has_mission_terminated):
                result.warn(
                    f"[L5] Subtree '{bt_id}' : pattern ReactiveFallback("
                    f"Repeat(-1), MissionTerminated) manquant"
                )


# ── Public API ───────────────────────────────────────────────────────────────


def validate_bt(
    xml_str: str, catalog: SkillsCatalog | None = None
) -> ValidationResult:
    """
    Validate a NAV4RAIL BT XML on 5 levels.

    Args:
        xml_str: XML string to validate
        catalog: SkillsCatalog instance (uses default if None)

    Returns:
        ValidationResult with valid, errors, warnings, score
    """
    cat = _get_catalog(catalog)
    result = ValidationResult()

    # Level 1
    ok, root, msg = _validate_l1(xml_str, cat)
    if not ok:
        result.fail(msg, level=1)
        return result
    if msg != "OK":
        result.warn(msg)

    # Level 2
    _validate_l2(root, result, cat)
    if not result.valid:
        return result

    # Level 3
    _validate_l3(root, result, cat)

    # Level 4
    _validate_l4(root, result, cat)

    # Level 5
    _validate_l5(root, result, cat)

    return result


def validate_ports(xml_str: str, catalog: SkillsCatalog | None = None) -> list[str]:
    """Validate only ports (L4). Returns list of issues."""
    cat = _get_catalog(catalog)
    xml_str = xml_str.strip()
    if xml_str.startswith("<?xml"):
        xml_str = xml_str[xml_str.index("?>") + 2 :].strip()
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return ["XML mal forme"]
    result = ValidationResult()
    _validate_l4(root, result, cat)
    return result.warnings


def enrich_ports(xml_str: str, catalog: SkillsCatalog | None = None) -> str:
    """
    Inject default blackboard ports on nodes that lack them.

    Accepts both formats:
      v4 (proxy)  : <MoveAndStop name="stop"/>  -> adds motion_params="{motion_params}"
      v5 (BTCPP)  : <Action ID="MoveAndStop"/>  -> same

    Does not modify ports already present.
    Returns enriched XML (or original if parsing fails).
    """
    cat = _get_catalog(catalog)
    defaults = cat.default_port_values()
    valid_skills = cat.valid_skills()

    xml_clean = xml_str.strip()
    if xml_clean.startswith("<?xml"):
        xml_clean = xml_clean[xml_clean.index("?>") + 2 :].strip()
    try:
        root = ET.fromstring(xml_clean)
    except ET.ParseError:
        return xml_str

    modified = False
    for elem in root.iter():
        if elem.tag in ("Action", "Condition"):
            skill_id = elem.get("ID", "")
        elif elem.tag in valid_skills:
            skill_id = elem.tag
        else:
            continue

        skill_defaults = defaults.get(skill_id)
        if not skill_defaults:
            continue

        for port, default_val in skill_defaults.items():
            if port not in elem.attrib:
                elem.set(port, default_val)
                modified = True

    if not modified:
        return xml_str

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        xml_str = open(sys.argv[1], encoding="utf-8").read()
    else:
        xml_str = sys.stdin.read()

    result = validate_bt(xml_str)
    print(result.summary())
    for e in result.errors:
        print(f"  {e}")
    for w in result.warnings:
        print(f"  {w}")
    sys.exit(0 if result.valid else 1)
