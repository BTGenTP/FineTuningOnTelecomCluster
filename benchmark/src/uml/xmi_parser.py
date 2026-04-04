from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from xml.etree import ElementTree as ET


UML_NS = {
    "xmi": "http://www.omg.org/spec/XMI/20131001",
    "uml": "http://www.eclipse.org/uml2/5.0.0/UML",
    "robotics.skills": "http://www.eclipse.org/papyrus/robotics/skills/1",
}


@dataclass(frozen=True, slots=True)
class UmlParameter:
    name: str
    direction: str  # in|out|inout|return|unspecified
    type_href: Optional[str]


@dataclass(frozen=True, slots=True)
class UmlOperation:
    op_id: str
    name: str
    parameters: List[UmlParameter]
    description: str


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_skills_uml(path: str | Path) -> List[UmlOperation]:
    p = Path(path).expanduser().resolve()
    tree = ET.parse(str(p))
    root = tree.getroot()

    # Map operation_id -> description via robotics.skills:SkillDefinition base_Operation
    descriptions: Dict[str, str] = {}
    for sd in root.findall(".//robotics.skills:SkillDefinition", UML_NS):
        base_op = sd.attrib.get("base_Operation")
        if not base_op:
            continue
        descriptions[base_op] = (sd.attrib.get("description") or "").strip()

    ops: List[UmlOperation] = []
    xmi_type_attr = f"{{{UML_NS['xmi']}}}type"
    xmi_id_attr = f"{{{UML_NS['xmi']}}}id"
    for iface in root.iter():
        if iface.attrib.get(xmi_type_attr) != "uml:Interface":
            continue
        for op in list(iface):
            if op.tag != "ownedOperation":
                continue
            if op.attrib.get(xmi_type_attr) != "uml:Operation":
                continue
            op_id = op.attrib.get(xmi_id_attr) or op.attrib.get("xmi:id") or ""
            name = op.attrib.get("name") or ""
            if not op_id or not name:
                continue
            params: List[UmlParameter] = []
            for prm in list(op):
                if prm.tag != "ownedParameter":
                    continue
                prm_name = prm.attrib.get("name") or ""
                direction = prm.attrib.get("direction") or "unspecified"
                if not prm_name:
                    continue
                type_node = None
                for ch in list(prm):
                    if ch.tag == "type":
                        type_node = ch
                        break
                type_href = type_node.attrib.get("href") if type_node is not None else None
                params.append(UmlParameter(name=prm_name, direction=direction, type_href=type_href))
            ops.append(
                UmlOperation(
                    op_id=op_id,
                    name=name,
                    parameters=params,
                    description=descriptions.get(op_id, ""),
                )
            )
    return ops


def infer_bt_tags_from_behaviortreeschema(paths: Iterable[Path]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for p in paths:
        try:
            root = ET.parse(str(p)).getroot()
        except Exception:
            continue
        for node in root.iter():
            if node.tag in {"Action", "Condition"} and "ID" in node.attrib:
                sid = node.attrib["ID"]
                tag = node.tag
                prev = mapping.get(sid)
                if prev and prev != tag:
                    raise ValueError(f"Conflicting bt_tag inference for {sid}: {prev} vs {tag} (from {p})")
                mapping[sid] = tag
    return mapping

