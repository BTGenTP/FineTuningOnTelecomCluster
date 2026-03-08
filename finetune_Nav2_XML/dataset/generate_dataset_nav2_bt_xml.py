from __future__ import annotations

import argparse
import json
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
from xml.etree.ElementTree import Element, ElementTree, SubElement

import xml.etree.ElementTree as ET

from finetune_Nav2_XML.catalog.catalog_io import default_catalog_path, load_catalog, required_param_names
from finetune_Nav2_XML.train.prompting import build_mistral_inst_prompt
from finetune_Nav2_XML.validator.validate_bt_xml import _validate_tree, DEFAULT_REFERENCE_DIR


@dataclass(frozen=True)
class Entry:
    mission: str
    xml: str
    meta: Dict[str, Any]
    prompt: Optional[str] = None

    def to_jsonl(self) -> str:
        obj: Dict[str, Any] = {"mission": self.mission, "xml": self.xml, "meta": self.meta}
        if self.prompt:
            obj["prompt"] = self.prompt
        return json.dumps(obj, ensure_ascii=False)


def _choice(xs: List[Any]) -> Any:
    return xs[random.randrange(0, len(xs))]


def _float(val: float) -> float:
    return float(f"{val:.3f}")


def _spin_angle() -> Tuple[float, str]:
    presets = [
        (1.57, "90°"),
        (3.14, "180°"),
        (0.785, "45°"),
        (6.283, "360°"),
    ]
    rad, deg = _choice(presets)
    return _float(rad), deg


def _wait_duration() -> float:
    return _float(_choice([0.5, 1.0, 2.0, 3.0, 5.0]))


def _backup() -> Tuple[float, float]:
    dist = _float(_choice([0.2, 0.3, 0.5, 1.0]))
    speed = _float(_choice([0.05, 0.08, 0.1]))
    return dist, speed


def _drive_on_heading() -> Tuple[float, float, float]:
    dist = _float(_choice([0.5, 1.0, 2.0, 3.0]))
    speed = _float(_choice([0.1, 0.15, 0.2]))
    allowance = _float(_choice([6.0, 10.0, 12.0, 18.0]))
    return dist, speed, allowance


def _service_name(kind: str) -> str:
    if kind == "local":
        return "local_costmap/clear_entirely_local_costmap"
    if kind == "global":
        return "global_costmap/clear_entirely_global_costmap"
    raise ValueError(f"Unknown costmap kind: {kind}")


def _indent_xml(elem: Element, level: int = 0) -> None:
    indent_str = "  "
    i = "\n" + level * indent_str
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + indent_str
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():  # type: ignore[name-defined]
            child.tail = i  # type: ignore[name-defined]
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def _reference_tree_path() -> Path:
    return Path(__file__).resolve().parents[1] / "reference_behavior_trees" / "navigate_then_spin.xml"


def _add_nav_subtree_definition(root: Element) -> None:
    """
    Append <BehaviorTree ID="NavigateToPoseWithReplanningAndRecovery">...</BehaviorTree>
    copied from the local reference XML under finetune_Nav2_XML/reference_behavior_trees/.
    """
    src = _reference_tree_path()
    if not src.exists():
        raise RuntimeError(f"Missing subtree reference XML: {src}")
    src_tree = ET.parse(str(src))
    src_root = src_tree.getroot()
    found = None
    for bt in src_root.findall("BehaviorTree"):
        if bt.get("ID") == "NavigateToPoseWithReplanningAndRecovery":
            found = bt
            break
    if found is None:
        raise RuntimeError(f"Subtree NavigateToPoseWithReplanningAndRecovery not found in {src}")
    copied = ET.fromstring(ET.tostring(found, encoding="utf-8"))
    root.append(copied)


def _mk_leaf(tag: str, attrs: Mapping[str, Any] | None = None) -> Element:
    el = Element(tag)
    for k, v in (attrs or {}).items():
        if isinstance(v, bool):
            el.set(k, "true" if v else "false")
        else:
            el.set(k, str(v))
    return el


def _mk_sequence(name: str, children: List[Element]) -> Element:
    el = Element("Sequence", {"name": name})
    for c in children:
        el.append(c)
    return el


def _mk_round_robin(name: str, children: List[Element]) -> Element:
    el = Element("RoundRobin", {"name": name})
    for c in children:
        el.append(c)
    return el


def _mk_rate_controller(hz: float, child: Element) -> Element:
    el = Element("RateController", {"hz": str(_float(hz))})
    el.append(child)
    return el


def _mk_repeat(num_cycles: int, child: Element) -> Element:
    el = Element("Repeat", {"num_cycles": str(int(num_cycles))})
    el.append(child)
    return el


def _mk_inverter(child: Element) -> Element:
    el = Element("Inverter")
    el.append(child)
    return el


def _mk_keep_running_until_failure(child: Element) -> Element:
    el = Element("KeepRunningUntilFailure")
    el.append(child)
    return el


def _mk_pipeline_sequence(name: str, children: List[Element]) -> Element:
    el = Element("PipelineSequence", {"name": name})
    for c in children:
        el.append(c)
    return el


def _mk_fallback(children: List[Element]) -> Element:
    el = Element("Fallback")
    for c in children:
        el.append(c)
    return el


def _mk_reactive_fallback(name: str, children: List[Element]) -> Element:
    el = Element("ReactiveFallback", {"name": name})
    for c in children:
        el.append(c)
    return el


def _mk_recovery_node(name: str, retries: int, children: List[Element]) -> Element:
    el = Element("RecoveryNode", {"name": name, "number_of_retries": str(int(retries))})
    for c in children:
        el.append(c)
    return el


def _nav_subtree_call() -> Element:
    return Element("SubTree", {"ID": "NavigateToPoseWithReplanningAndRecovery", "__shared_blackboard": "true"})


def _recovery_actions_block() -> Element:
    clearing = _mk_sequence(
        "ClearingActions",
        [
            _mk_leaf(
                "ClearEntireCostmap",
                {"name": "ClearLocalCostmap-Subtree", "service_name": _service_name("local")},
            ),
            _mk_leaf(
                "ClearEntireCostmap",
                {"name": "ClearGlobalCostmap-Subtree", "service_name": _service_name("global")},
            ),
        ],
    )
    spin_rad, _deg = _spin_angle()
    backup_dist, backup_speed = _backup()
    return _mk_round_robin(
        "RecoveryActions",
        [
            clearing,
            _mk_leaf("Spin", {"spin_dist": spin_rad, "is_recovery": True}),
            _mk_leaf("Wait", {"wait_duration": _wait_duration()}),
            _mk_leaf("BackUp", {"backup_dist": backup_dist, "backup_speed": backup_speed}),
        ],
    )


def _pattern_nav_recovery_rate_pipeline(*, variant: str) -> Tuple[str, ElementTree]:
    """
    Build a fully featured MainTree using multiple BT patterns.
    """
    hz = _choice([0.5, 1.0, 2.0])
    retries = _choice([2, 3, 4, 6])

    post_actions: List[Element] = []
    post_parts: List[str] = []
    if _choice([True, False]):
        dur = _wait_duration()
        post_actions.append(_mk_leaf("Wait", {"wait_duration": dur}))
        post_parts.append(f"attends {dur} s")
    if _choice([True, False]):
        rad, deg = _spin_angle()
        post_actions.append(_mk_leaf("Spin", {"spin_dist": rad}))
        post_parts.append(f"tourne de {deg} ({rad} rad)")
    if not post_actions:
        dur = _wait_duration()
        post_actions.append(_mk_leaf("Wait", {"wait_duration": dur}))
        post_parts.append(f"attends {dur} s")

    pipeline = _mk_pipeline_sequence(
        "MainPipeline",
        [
            _mk_rate_controller(float(hz), _nav_subtree_call()),
            _mk_sequence("PostNavActions", post_actions),
        ],
    )

    # Recovery branch (GoalUpdated + RoundRobin actions) — tags are allowed via reference BT scanning.
    recovery = _mk_reactive_fallback("RecoveryFallback", [_mk_leaf("GoalUpdated"), _recovery_actions_block()])

    main = _mk_recovery_node("MainRecovery", int(retries), [pipeline, recovery])

    if variant == "keep_running":
        main = _mk_keep_running_until_failure(main)
    elif variant == "repeat":
        main = _mk_repeat(_choice([2, 3, 5]), main)

    mission = "Navigue vers le goal (Nav2)".capitalize()
    if post_parts:
        mission += ", puis " + ", puis ".join(post_parts)
    mission += "."

    root = Element("root", {"main_tree_to_execute": "MainTree"})
    bt = SubElement(root, "BehaviorTree", {"ID": "MainTree"})
    bt.append(main)
    _add_nav_subtree_definition(root)
    _indent_xml(root)
    return mission, ElementTree(root)


def _pattern_local_actions_with_fallback_roundrobin() -> Tuple[str, ElementTree]:
    dist, speed, allowance = _drive_on_heading()
    spin_rad, deg = _spin_angle()
    dur = _wait_duration()
    backup_dist, backup_speed = _backup()

    primary = _mk_sequence(
        "TryForwardThenSpin",
        [
            _mk_leaf("DriveOnHeading", {"dist_to_travel": dist, "speed": speed, "time_allowance": allowance}),
            _mk_leaf("Spin", {"spin_dist": spin_rad}),
        ],
    )
    recovery = _mk_sequence(
        "FallbackRecovery",
        [
            _mk_leaf("BackUp", {"backup_dist": backup_dist, "backup_speed": backup_speed}),
            _mk_leaf("Wait", {"wait_duration": dur}),
        ],
    )

    rr = _mk_round_robin("LocalActions", [primary, recovery])
    fb = _mk_fallback([rr, _mk_leaf("Wait", {"wait_duration": dur})])
    wrapped = _mk_repeat(_choice([2, 3]), fb)

    mission = f"Avance de {dist} m à {speed} m/s (limite {allowance} s), puis tourne de {deg} ({spin_rad} rad), et boucle avec fallback."
    root = Element("root", {"main_tree_to_execute": "MainTree"})
    bt = SubElement(root, "BehaviorTree", {"ID": "MainTree"})
    bt.append(wrapped)
    _indent_xml(root)
    return mission, ElementTree(root)


def _tree_to_str(tree: ElementTree) -> str:
    import io

    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=False)
    return buf.getvalue().decode("utf-8")


def _validate_required_ports(entries: List[Entry], required_by_skill: Mapping[str, set[str]]) -> None:
    """
    Best-effort sanity check: if the XML contains a catalog atomic skill tag,
    required attrs must be present.
    """
    # Map skill_id -> bt_tag (many-to-one); we validate via required ports per skill_id by checking its bt_tag.
    # For this dataset we only use a small set; validator is the source of truth.
    for e in entries:
        # Parse XML and check required attrs for known leaf tags.
        try:
            root = ET.fromstring(e.xml)
        except Exception as exc:
            raise ValueError(f"Invalid XML in dataset entry: {exc}") from exc
        for el in root.iter():
            tag = el.tag
            # Validate required ports for leaf tags that correspond to skills with those required ports.
            for skill_id, req in required_by_skill.items():
                # Heuristic: skill_id == tag for most atomic nodes; ClearEntireCostmap* share tag.
                # Use XML validator later for the strict allowlist + required attrs.
                if skill_id in ("ClearEntireCostmapLocal", "ClearEntireCostmapGlobal"):
                    if tag != "ClearEntireCostmap":
                        continue
                elif skill_id == "NavigateToGoalWithReplanningAndRecovery":
                    if tag != "SubTree":
                        continue
                else:
                    if tag != skill_id:
                        continue
                for r in sorted(req):
                    if r not in el.attrib:
                        raise ValueError(f"Missing required attr '{r}' on <{tag}> in mission: {e.mission}")


def generate_dataset(*, seed: int, n: int, catalog_path: str) -> List[Entry]:
    random.seed(int(seed))
    catalog = load_catalog(catalog_path)
    required_by_skill = required_param_names(catalog)

    entries: List[Entry] = []
    patterns = [
        ("nav_recovery_keep_running", lambda: _pattern_nav_recovery_rate_pipeline(variant="keep_running")),
        ("nav_recovery_repeat", lambda: _pattern_nav_recovery_rate_pipeline(variant="repeat")),
        ("nav_recovery_plain", lambda: _pattern_nav_recovery_rate_pipeline(variant="plain")),
        ("local_fallback_roundrobin", _pattern_local_actions_with_fallback_roundrobin),
    ]

    # Generate more than n and filter by strict validation to keep dataset clean.
    attempts = 0
    while len(entries) < int(n):
        attempts += 1
        cat, fn = _choice(patterns)
        mission, tree = fn()
        xml = _tree_to_str(tree)

        # Validate statically (strict) before accepting.
        with tempfile.NamedTemporaryFile("w+", suffix=".xml", delete=False, encoding="utf-8") as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(xml)
        try:
            report = _validate_tree(
                xml_path=tmp_path,
                reference_dir=DEFAULT_REFERENCE_DIR if DEFAULT_REFERENCE_DIR.exists() else None,
                catalog_path=Path(catalog_path).expanduser().resolve(),
                strict_attrs=True,
                strict_blackboard=True,
                external_bb_vars=["goal"],
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        if not bool(report.get("ok")):
            # Skip invalid samples (structural integrity is more important than reaching exact n quickly).
            if attempts > int(n) * 50:
                raise RuntimeError(f"Too many invalid samples while building dataset (kept={len(entries)} attempts={attempts}).")
            continue

        prompt, _anchor = build_mistral_inst_prompt(mission=mission, catalog=catalog)
        entries.append(
            Entry(
                mission=mission,
                xml=xml,
                prompt=prompt + xml + " </s>",
                meta={
                    "category": cat,
                    "seed": int(seed),
                    "catalog_path": str(Path(catalog_path).expanduser().resolve()),
                    "dataset_version": "nav2_bt_xml_v1",
                },
            )
        )

    _validate_required_ports(entries, required_by_skill)
    return entries


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a synthetic Nav2 dataset: mission -> BT XML (BehaviorTree.CPP v4).")
    p.add_argument("--out", type=str, required=True, help="Output JSONL path.")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--catalog", type=str, default=str(default_catalog_path()))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    entries = generate_dataset(seed=int(args.seed), n=int(args.n), catalog_path=str(args.catalog))
    with out.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(e.to_jsonl() + "\n")

    print(f"Wrote XML-direct dataset: {out} ({len(entries)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

