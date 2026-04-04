from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from xml.etree import ElementTree as ET

from ..contracts import ValidationIssue, ValidationReport, ValidationSummary
from ..data.catalog import control_nodes, default_catalog_path, load_catalog, skill_map


BLACKBOARD_RE = re.compile(r"^\{([A-Za-z0-9_\-]+)\}$")
INSPECTION_TYPES = {"10", "11", "12", "13", "14"}
MOVEMENT_SKILLS = {"Move", "CreatePath", "MoveAndStop", "Deccelerate", "ProjectPointOnNetwork"}


def _digest_xml(xml_text: str) -> str:
    return hashlib.sha256(xml_text.encode("utf-8")).hexdigest()


def _load_xml_text(xml_text: Optional[str], xml_path: Optional[str | Path]) -> Tuple[str, Optional[str]]:
    if xml_text is not None:
        return xml_text, str(Path(xml_path).resolve()) if xml_path else None
    if xml_path is None:
        raise ValueError("Either xml_text or xml_path must be provided.")
    path = Path(xml_path).expanduser().resolve()
    return path.read_text(encoding="utf-8"), str(path)


def _parse_xml(xml_text: str) -> ET.Element:
    return ET.fromstring(xml_text)


def _build_parent_map(root: ET.Element) -> Dict[ET.Element, ET.Element]:
    parent_map: Dict[ET.Element, ET.Element] = {}
    for parent in root.iter():
        for child in list(parent):
            parent_map[child] = parent
    return parent_map


def _xpath_of(node: ET.Element, parent_map: Mapping[ET.Element, ET.Element]) -> str:
    parts: list[str] = []
    current: Optional[ET.Element] = node
    while current is not None:
        parent = parent_map.get(current)
        if parent is None:
            parts.append(f"/{current.tag}")
            break
        siblings = [child for child in list(parent) if child.tag == current.tag]
        index = siblings.index(current) + 1
        parts.append(f"/{current.tag}[{index}]")
        current = parent
    return "".join(reversed(parts))


def _summary(issues: Sequence[ValidationIssue]) -> ValidationSummary:
    by_layer: Dict[str, Dict[str, int]] = {}
    errors = 0
    warnings = 0
    key_map = {"error": "errors", "warning": "warnings", "info": "info"}
    for issue in issues:
        layer_bucket = by_layer.setdefault(issue.layer, {"errors": 0, "warnings": 0, "info": 0})
        bucket_key = key_map[issue.severity]
        layer_bucket[bucket_key] = layer_bucket.get(bucket_key, 0) + 1
        if issue.severity == "error":
            errors += 1
        elif issue.severity == "warning":
            warnings += 1
    return ValidationSummary(
        issues_total=len(issues),
        errors=errors,
        warnings=warnings,
        by_layer=by_layer,
    )


def _issue(
    layer: str,
    code: str,
    message: str,
    *,
    severity: str = "error",
    xpath: Optional[str] = None,
    node: Optional[ET.Element] = None,
) -> ValidationIssue:
    return ValidationIssue(
        layer=layer,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        code=code,
        message=message,
        xpath=xpath,
        node_tag=node.tag if node is not None else None,
        node_id=node.attrib.get("ID") if node is not None else None,
    )


def _coerce_severity(strict: bool) -> str:
    return "error" if strict else "warning"


def _collect_behavior_trees(root: ET.Element) -> Dict[str, ET.Element]:
    trees: Dict[str, ET.Element] = {}
    for bt in root.findall("BehaviorTree"):
        bt_id = bt.attrib.get("ID")
        if bt_id:
            trees[bt_id] = bt
    return trees


def _detect_cycles(tree_defs: Mapping[str, ET.Element]) -> List[List[str]]:
    graph: Dict[str, Set[str]] = {}
    for tree_id, tree_el in tree_defs.items():
        graph[tree_id] = {child.attrib["ID"] for child in tree_el.iter("SubTreePlus") if "ID" in child.attrib}

    visiting: Set[str] = set()
    visited: Set[str] = set()
    path: List[str] = []
    cycles: List[List[str]] = []

    def dfs(node: str) -> None:
        visiting.add(node)
        visited.add(node)
        path.append(node)
        for target in graph.get(node, set()):
            if target not in tree_defs:
                continue
            if target not in visited:
                dfs(target)
            elif target in visiting and target in path:
                start = path.index(target)
                cycles.append(path[start:] + [target])
        path.pop()
        visiting.remove(node)

    for tree_id in tree_defs:
        if tree_id not in visited:
            dfs(tree_id)
    return cycles


def _is_control_node(node: ET.Element, controls: Mapping[str, Dict[str, Any]]) -> bool:
    return node.tag in controls


def _is_skill_node(node: ET.Element) -> bool:
    return node.tag in {"Action", "Condition"}


def _extract_bb_var(value: str) -> Optional[str]:
    match = BLACKBOARD_RE.fullmatch((value or "").strip())
    return match.group(1) if match else None


def _check_attr_type(expected: str, value: str) -> bool:
    optional = "optional" in expected
    normalized = expected.replace(" (optional)", "")
    if optional and value == "":
        return True
    expected = normalized
    if expected == "string":
        return True
    if expected == "bool":
        return value.lower() in {"true", "false"}
    if expected == "integer":
        return re.fullmatch(r"-?\d+", value) is not None
    if expected.startswith("enum:"):
        return value in set(expected.removeprefix("enum:").split("|"))
    if expected.startswith("const:"):
        return value == expected.removeprefix("const:")
    if expected in {"blackboard_port_input", "blackboard_port_output"}:
        return _extract_bb_var(value) is not None
    if expected == "string_or_blackboard":
        return True
    if expected.startswith("enum:") and "optional" in expected:
        return True
    if expected.endswith("(optional)"):
        base = expected.split(" ", 1)[0]
        return not value or _check_attr_type(base, value)
    return True


def _effective_required_attributes(attributes: Mapping[str, str], required: Iterable[str]) -> Set[str]:
    required_set = set(required)
    for attr_name, attr_type in attributes.items():
        if "optional" in attr_type:
            required_set.discard(attr_name)
    return required_set


def _validate_xsd(xml_text: str, xsd_path: Optional[str | Path], issues: List[ValidationIssue]) -> None:
    if not xsd_path:
        return
    try:
        from lxml import etree
    except Exception:
        issues.append(
            _issue(
                "L1",
                "xsd_dependency_missing",
                "XSD validation requested but lxml is not available.",
                severity="warning",
            )
        )
        return

    schema_path = Path(xsd_path).expanduser().resolve()
    if not schema_path.exists():
        issues.append(_issue("L1", "xsd_not_found", f"XSD file not found: {schema_path}"))
        return

    try:
        xml_doc = etree.fromstring(xml_text.encode("utf-8"))
        schema_doc = etree.parse(str(schema_path))
        schema = etree.XMLSchema(schema_doc)
        if not schema.validate(xml_doc):
            for error in schema.error_log:
                issues.append(
                    _issue(
                        "L1",
                        "xsd_validation_failed",
                        str(error.message),
                        xpath=f"line:{error.line}",
                    )
                )
    except Exception as exc:
        issues.append(_issue("L1", "xsd_validation_exception", f"XSD validation failed: {exc}"))


def _behavior_tree_body(bt: ET.Element) -> Optional[ET.Element]:
    children = list(bt)
    return children[0] if children else None


def _sequence_blackboard_check(
    sequence_node: ET.Element,
    parent_map: Mapping[ET.Element, ET.Element],
    skill_specs: Mapping[str, Any],
    available: Set[str],
    issues: List[ValidationIssue],
    *,
    strict: bool,
) -> Set[str]:
    current = set(available)
    xp = _xpath_of(sequence_node, parent_map)
    for child in list(sequence_node):
        if child.tag == "Sequence":
            current |= _sequence_blackboard_check(child, parent_map, skill_specs, current, issues, strict=strict)
            continue
        if child.tag in {"Fallback", "ReactiveFallback", "Repeat", "SubTreePlus"}:
            current |= _collect_produced_vars(child, skill_specs)
            continue
        if not _is_skill_node(child):
            continue
        skill_id = child.attrib.get("ID", "")
        if skill_id not in skill_specs:
            continue
        spec = skill_specs[skill_id]
        missing_inputs = []
        input_attr_names = [attr_name for attr_name, attr_type in spec.attributes.items() if attr_type == "blackboard_port_input"]
        for attr_name in input_attr_names:
            attr_value = child.attrib.get(attr_name)
            if not attr_value:
                continue
            bb_var = _extract_bb_var(attr_value)
            if bb_var and bb_var not in current:
                missing_inputs.append(bb_var)
        if missing_inputs:
            severity = _coerce_severity(strict)
            issues.append(
                _issue(
                    "L2",
                    "blackboard_unproduced",
                    f"Blackboard variables consumed before production in Sequence {xp}: {', '.join(sorted(set(missing_inputs)))}",
                    severity=severity,
                    xpath=_xpath_of(child, parent_map),
                    node=child,
                )
            )
            if skill_id in MOVEMENT_SKILLS:
                issues.append(
                    _issue(
                        "L3",
                        "nav4rail_blackboard_chain",
                        f"{skill_id} consumes blackboard data before an upstream producer in the same execution flow.",
                        severity=severity,
                        xpath=_xpath_of(child, parent_map),
                        node=child,
                    )
                )
        output_attr_names = [attr_name for attr_name, attr_type in spec.attributes.items() if attr_type == "blackboard_port_output"]
        for attr_name in output_attr_names:
            matching_value = child.attrib.get(attr_name)
            if not matching_value:
                continue
            bb_var = _extract_bb_var(matching_value)
            if bb_var:
                current.add(bb_var)
        for produced in spec.blackboard_outputs:
            current.add(produced)
    return current


def _collect_produced_vars(node: ET.Element, skill_specs: Mapping[str, Any]) -> Set[str]:
    produced: Set[str] = set()
    if _is_skill_node(node):
        skill_id = node.attrib.get("ID", "")
        spec = skill_specs.get(skill_id)
        if spec is not None:
            for attr_name, attr_type in spec.attributes.items():
                if attr_type == "blackboard_port_output" and attr_name in node.attrib:
                    bb_var = _extract_bb_var(node.attrib[attr_name])
                    if bb_var:
                        produced.add(bb_var)
            for output_name in spec.blackboard_outputs:
                produced.add(output_name)
    for child in list(node):
        produced |= _collect_produced_vars(child, skill_specs)
    return produced


def _validate_nav4rail_semantics(
    root: ET.Element,
    bt_defs: Mapping[str, ET.Element],
    parent_map: Mapping[ET.Element, ET.Element],
    issues: List[ValidationIssue],
    skill_specs: Mapping[str, Any],
    *,
    strict: bool,
) -> None:
    execute_roots = [
        body
        for body in (_behavior_tree_body(bt) for bt in bt_defs.values())
        if body is not None and body.tag == "ReactiveFallback"
    ]
    has_valid_loop = False
    for node in execute_roots:
        children = list(node)
        has_repeat = any(child.tag == "Repeat" and child.attrib.get("num_cycles") == "-1" for child in children)
        has_terminated = any(
            _is_skill_node(child) and child.attrib.get("ID") == "MissionTerminated" for child in children
        )
        if has_repeat and has_terminated:
            has_valid_loop = True
            break
    if not has_valid_loop:
        issues.append(
            _issue(
                "L3",
                "nav4rail_execution_loop",
                "Expected a ReactiveFallback execution root containing Repeat num_cycles='-1' and MissionTerminated.",
                severity=_coerce_severity(strict),
            )
        )

    for bt in bt_defs.values():
        skill_nodes = [node for node in bt.iter() if _is_skill_node(node)]
        inspection_triggered = any(
            node.attrib.get("ID") == "CheckCurrentStepType" and node.attrib.get("type_to_be_checked") in INSPECTION_TYPES
            for node in skill_nodes
        )
        if not inspection_triggered:
            continue
        ids = {node.attrib.get("ID", "") for node in skill_nodes}
        if "AnalyseMeasurements" not in ids:
            issues.append(
                _issue(
                    "L3",
                    "inspection_requires_analysis",
                    "Inspection subtree is missing AnalyseMeasurements.",
                    severity=_coerce_severity(strict),
                    xpath=_xpath_of(bt, parent_map),
                    node=bt,
                )
            )
        if not ({"GenerateCorrectiveSubSequence", "InsertCorrectiveSubSequence"} <= ids or "MeasurementsEnforcedValidated" in ids or "MeasurementsQualityValidated" in ids):
            issues.append(
                _issue(
                    "L3",
                    "inspection_requires_fallback",
                    "Inspection subtree must contain a quality fallback or corrective sequence.",
                    severity=_coerce_severity(strict),
                    xpath=_xpath_of(bt, parent_map),
                    node=bt,
                )
            )


def validate(
    *,
    xml_text: Optional[str] = None,
    xml_path: Optional[str | Path] = None,
    catalog_path: Optional[str | Path] = None,
    xsd_path: Optional[str | Path] = None,
    strict: bool = True,
    external_blackboard: Optional[Iterable[str]] = None,
) -> ValidationReport:
    resolved_catalog = Path(catalog_path).expanduser().resolve() if catalog_path else default_catalog_path().resolve()
    catalog = load_catalog(resolved_catalog)
    controls = control_nodes(catalog)
    skill_specs = skill_map(catalog)

    raw_xml, resolved_xml_path = _load_xml_text(xml_text, xml_path)
    issues: List[ValidationIssue] = []

    try:
        root = _parse_xml(raw_xml)
    except Exception as exc:
        issues.append(_issue("L1", "xml_parse_error", f"Failed to parse XML: {exc}"))
        return ValidationReport(
            ok=False,
            xml_path=resolved_xml_path,
            xml_digest=_digest_xml(raw_xml),
            catalog_path=str(resolved_catalog),
            xsd_path=str(Path(xsd_path).expanduser().resolve()) if xsd_path else None,
            summary=_summary(issues),
            issues=issues,
            metadata={},
        )

    parent_map = _build_parent_map(root)
    if root.tag != "root":
        issues.append(_issue("L1", "root_tag", "Root tag must be <root>.", xpath=_xpath_of(root, parent_map), node=root))
    if "main_tree_to_execute" not in root.attrib:
        issues.append(_issue("L1", "root_main_tree", "Missing root@main_tree_to_execute.", xpath=_xpath_of(root, parent_map), node=root))

    _validate_xsd(raw_xml, xsd_path, issues)

    bt_defs = _collect_behavior_trees(root)
    main_tree = root.attrib.get("main_tree_to_execute", "")
    if main_tree and main_tree not in bt_defs:
        issues.append(_issue("L1", "main_tree_missing", f"Missing BehaviorTree definition for {main_tree!r}.", xpath=_xpath_of(root, parent_map), node=root))

    seen_ids: Set[str] = set()
    for bt in root.findall("BehaviorTree"):
        bt_id = bt.attrib.get("ID")
        if not bt_id:
            issues.append(_issue("L1", "behavior_tree_missing_id", "BehaviorTree nodes require ID.", xpath=_xpath_of(bt, parent_map), node=bt))
            continue
        if bt_id in seen_ids:
            issues.append(_issue("L1", "behavior_tree_duplicate_id", f"Duplicate BehaviorTree ID {bt_id!r}.", xpath=_xpath_of(bt, parent_map), node=bt))
        seen_ids.add(bt_id)

    subtree_refs = [node.attrib.get("ID", "") for node in root.iter("SubTreePlus") if node.attrib.get("ID")]
    for ref in subtree_refs:
        if ref not in bt_defs:
            issues.append(_issue("L1", "subtree_missing_definition", f"SubTreePlus references undefined BehaviorTree {ref!r}."))
    for cycle in _detect_cycles(bt_defs):
        issues.append(_issue("L1", "subtree_cycle", "SubTree cycle detected: " + " -> ".join(cycle)))

    for node in root.iter():
        xpath = _xpath_of(node, parent_map)
        if node.tag in {"root", "BehaviorTree"}:
            continue
        if _is_control_node(node, controls):
            control_spec = controls[node.tag]
            allowed = set(control_spec.get("attributes", {}).keys())
            required = set(control_spec.get("required_attributes", []))
            unknown_attrs = set(node.attrib) - allowed
            if unknown_attrs:
                issues.append(
                    _issue(
                        "L2",
                        "unknown_control_attr",
                        f"Unknown attributes on <{node.tag}>: {', '.join(sorted(unknown_attrs))}",
                        severity=_coerce_severity(strict),
                        xpath=xpath,
                        node=node,
                    )
                )
            for attr_name in required:
                if attr_name not in node.attrib:
                    issues.append(_issue("L2", "missing_control_attr", f"Missing required attribute {attr_name!r}.", xpath=xpath, node=node))
            for attr_name, attr_type in control_spec.get("attributes", {}).items():
                if attr_name in node.attrib and not _check_attr_type(attr_type, node.attrib[attr_name]):
                    issues.append(_issue("L2", "control_attr_type", f"{node.tag}.{attr_name} expects {attr_type}.", xpath=xpath, node=node))
            if node.tag != "SubTreePlus" and not list(node):
                issues.append(_issue("L2", "empty_control_node", f"Control node <{node.tag}> cannot be empty.", xpath=xpath, node=node))
            continue
        if not _is_skill_node(node):
            issues.append(_issue("L2", "unknown_tag", f"Tag <{node.tag}> is not allowed.", severity=_coerce_severity(strict), xpath=xpath, node=node))
            continue

        skill_id = node.attrib.get("ID")
        if not skill_id:
            issues.append(_issue("L2", "skill_missing_id", f"<{node.tag}> requires an ID attribute.", xpath=xpath, node=node))
            continue
        spec = skill_specs.get(skill_id)
        if spec is None:
            issues.append(_issue("L2", "skill_not_allowed", f"Skill ID {skill_id!r} is not part of the NAV4RAIL catalog.", xpath=xpath, node=node))
            continue
        if node.tag != spec.bt_tag:
            issues.append(_issue("L2", "skill_tag_mismatch", f"Skill {skill_id} must use <{spec.bt_tag}>.", xpath=xpath, node=node))
        required = _effective_required_attributes(spec.attributes, spec.required_attributes)
        unknown_attrs = set(node.attrib) - set(spec.attributes)
        if unknown_attrs:
            issues.append(
                _issue(
                    "L2",
                    "unknown_skill_attr",
                    f"Unknown attributes for {skill_id}: {', '.join(sorted(unknown_attrs))}",
                    severity=_coerce_severity(strict),
                    xpath=xpath,
                    node=node,
                )
            )
        for attr_name in required:
            if attr_name not in node.attrib:
                issues.append(_issue("L2", "missing_skill_attr", f"Skill {skill_id} is missing attribute {attr_name!r}.", xpath=xpath, node=node))
        for attr_name, attr_value in node.attrib.items():
            expected = spec.attributes.get(attr_name)
            if expected and not _check_attr_type(expected, attr_value):
                issues.append(
                    _issue(
                        "L2",
                        "skill_attr_type",
                        f"Skill {skill_id} attribute {attr_name!r} expects {expected}, got {attr_value!r}.",
                        xpath=xpath,
                        node=node,
                    )
                )

    external = set(external_blackboard or [])
    for bt in bt_defs.values():
        body = _behavior_tree_body(bt)
        if body is not None and body.tag == "Sequence":
            _sequence_blackboard_check(body, parent_map, skill_specs, set(external), issues, strict=strict)

    _validate_nav4rail_semantics(root, bt_defs, parent_map, issues, skill_specs, strict=strict)

    report = ValidationReport(
        ok=not any(issue.severity == "error" for issue in issues),
        xml_path=resolved_xml_path,
        xml_digest=_digest_xml(raw_xml),
        catalog_path=str(resolved_catalog),
        xsd_path=str(Path(xsd_path).expanduser().resolve()) if xsd_path else None,
        summary=_summary(issues),
        issues=issues,
        metadata={
            "main_tree_to_execute": main_tree,
            "behavior_tree_ids": sorted(bt_defs.keys()),
            "external_blackboard": sorted(external),
        },
    )
    return replace(report, summary=_summary(report.issues))


def validate_to_json(**kwargs: Any) -> str:
    return json.dumps(validate(**kwargs).to_dict(), indent=2, ensure_ascii=False)
