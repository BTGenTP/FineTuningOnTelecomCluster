from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
from xml.etree import ElementTree as ET


from finetune_Nav2.validator import validate_bt_xml as v


def validate_bt_xml(
    *,
    xml_path: Path,
    strict_attrs: bool,
    strict_blackboard: bool,
    catalog_path: Optional[Path] = None,
    reference_dir: Optional[Path] = None,
    external_bb_vars: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """
    Validate XML via the vendored validator under finetune_Nav2/validator/.
    """
    ext = external_bb_vars
    if ext is None:
        # Proxy Nav2: goal pose is injected by the /navigate_to_pose action server.
        ext = ["goal"]

    cat = catalog_path or v.DEFAULT_CATALOG_PATH
    ref = reference_dir or v.DEFAULT_REFERENCE_DIR

    report = v._validate_tree(
        xml_path=xml_path.resolve(),
        reference_dir=ref.resolve() if ref and Path(ref).exists() else None,
        catalog_path=Path(cat).resolve(),
        strict_attrs=bool(strict_attrs),
        strict_blackboard=bool(strict_blackboard),
        external_bb_vars=list(ext or []),
    )
    return report


def compute_bt_structure_metrics(xml_path: Path) -> Dict[str, Any]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    tags = [el.tag for el in root.iter()]
    node_count = len(tags)

    subtree_count = sum(1 for t in tags if t == "SubTree")

    base = {"root", "BehaviorTree"}
    control = {
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

    control_node_count = sum(1 for t in tags if t in control)
    control_node_count += sum(1 for t in tags if t == "SubTree")  # treat as structural node

    # Atomic nodes = nodes that are not base/control (best-effort).
    atomic_node_count = sum(1 for t in tags if (t not in base and t not in control and t != "SubTree"))

    # Depth: compute max path length.
    def depth(el: ET.Element, d: int) -> int:
        if len(list(el)) == 0:
            return d
        return max(depth(c, d + 1) for c in list(el))

    bt_depth = depth(root, 1)

    return {
        "bt_depth": int(bt_depth),
        "node_count": int(node_count),
        "subtree_count": int(subtree_count),
        "control_node_count": int(control_node_count),
        "atomic_node_count": int(atomic_node_count),
    }

