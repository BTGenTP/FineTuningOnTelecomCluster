"""
NAV4RAIL Mission Builder
========================

Strict Python API that compiles to a BTCPP v4 XML string. Errors surface at
construction time (UnknownSkillError, PortError, StructuralError) so the
Python interpreter itself becomes the first validation layer for PoT/ReAct
agents.

Two levels of API cohabit:

Low level (building blocks — full flexibility):
    builder.skill("Move", threshold_type="1", motion_params="{motion_params}")
    builder.sequence(name="...", children=[...])
    builder.fallback(name="...", children=[...])
    builder.reactive_fallback(name="...", children=[...])
    builder.repeat(num_cycles=-1, child=...)
    builder.subtree_plus(id="...", name="...", **ports)
    builder.register_behavior_tree(id="...", root=...)

High level (pattern-based — enforce SR-023..SR-027 by construction):
    builder.add_get_mission()            # LoadMission + MissionStructureValid
    builder.add_calculate_path()         # Fallback(Repeat(-1)(update+project+create+agregate), MissionFullyTreated)
    builder.add_base_preparation()
    builder.add_motion_subtree(step_type=12, ...)
    builder.add_execute(step_types=[0, 2, 12])
    builder.add_main_tree()

Output:
    xml = builder.to_xml()     # Final L1 + L2 check, then serialize.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

from src.data.skills_loader import SkillsCatalog, SkillSpec


# ── Errors ───────────────────────────────────────────────────────────────────


class BuilderError(Exception):
    """Base class for all MissionBuilder errors."""


class UnknownSkillError(BuilderError):
    """Raised when a skill ID is not in the catalog (hallucination)."""


class PortError(BuilderError):
    """Raised when a port is missing, unknown, or has an invalid value/type."""


class StructuralError(BuilderError):
    """Raised when a structural constraint (L2) is violated."""


class MissingRequiredSkillError(BuilderError):
    """Raised at to_xml() when a required skill (e.g. MoveAndStop) is absent."""


# ── Node dataclasses (internal AST) ──────────────────────────────────────────


@dataclass
class Node:
    """Base class for any BT AST node."""

    def to_element(self) -> ET.Element:  # pragma: no cover — abstract
        raise NotImplementedError


@dataclass
class SkillNode(Node):
    skill_id: str
    bt_tag: str  # "Action" or "Condition"
    name: str | None = None
    ports: dict[str, str] = field(default_factory=dict)

    def to_element(self) -> ET.Element:
        elem = ET.Element(self.bt_tag)
        if self.name:
            elem.set("name", self.name)
        elem.set("ID", self.skill_id)
        for k, v in self.ports.items():
            elem.set(k, v)
        return elem


@dataclass
class ControlNode(Node):
    tag: str  # "Sequence" | "Fallback" | "ReactiveFallback" | "Parallel"
    name: str | None = None
    children: list[Node] = field(default_factory=list)

    def to_element(self) -> ET.Element:
        elem = ET.Element(self.tag)
        if self.name:
            elem.set("name", self.name)
        for child in self.children:
            elem.append(child.to_element())
        return elem


@dataclass
class RepeatNode(Node):
    num_cycles: int
    name: str | None = None
    child: Node | None = None

    def to_element(self) -> ET.Element:
        elem = ET.Element("Repeat")
        if self.name:
            elem.set("name", self.name)
        elem.set("num_cycles", str(self.num_cycles))
        if self.child is not None:
            elem.append(self.child.to_element())
        return elem


@dataclass
class SubTreePlusNode(Node):
    subtree_id: str
    name: str | None = None
    autoremap: bool = True
    extra_ports: dict[str, str] = field(default_factory=dict)

    def to_element(self) -> ET.Element:
        elem = ET.Element("SubTreePlus")
        if self.name:
            elem.set("name", self.name)
        elem.set("ID", self.subtree_id)
        if self.autoremap:
            elem.set("__autoremap", "true")
        for k, v in self.extra_ports.items():
            elem.set(k, v)
        return elem


@dataclass
class BehaviorTreeNode(Node):
    bt_id: str
    root: Node

    def to_element(self) -> ET.Element:
        elem = ET.Element("BehaviorTree")
        elem.set("ID", self.bt_id)
        elem.append(self.root.to_element())
        return elem


# ── Step-type metadata (derived from catalog) ────────────────────────────────

# Map numeric step_type -> (subtree_id, description)
_STEP_TYPE_DEFAULT_NAMES = {
    0: ("move", "MOVE"),
    1: ("deccelerate", "DECCELERATE"),
    2: ("reach_and_stop", "REACH AND STOP"),
    3: ("pass", "PASS"),
    4: ("reach_stop_and_dont_wait", "REACH STOP AND DONT WAIT"),
    10: ("move_and_inspect", "MOVE AND INSPECT"),
    11: ("deccelerate_and_inspect", "DECCELERATE AND INSPECT"),
    12: ("reach_and_stop_inspecting", "REACH AND STOP INSPECTING"),
    13: ("pass_and_stop_inspecting", "PASS AND STOP INSPECTING"),
    14: ("reach_stop_inspecting_dont_wait", "REACH STOP INSPECTING DONT WAIT"),
}


# ── Public API ────────────────────────────────────────────────────────────────


class MissionBuilder:
    """
    Strict fluent builder that compiles to BTCPP v4 XML.

    All construction methods return Node objects (for composition) or self
    (for chaining, when the method registers a BehaviorTree).
    """

    def __init__(
        self,
        main_tree_id: str = "generated_mission",
        catalog: SkillsCatalog | None = None,
        require_move_and_stop: bool = True,
        max_depth: int | None = None,
    ) -> None:
        self._catalog: SkillsCatalog = catalog or SkillsCatalog()
        self._valid_skills = self._catalog.valid_skills()
        self._skill_ports = self._catalog.skill_ports()
        self._main_tree_id = main_tree_id
        self._btcpp_format = "4"
        self._behavior_trees: list[BehaviorTreeNode] = []
        self._registered_ids: set[str] = set()
        self._require_move_and_stop = require_move_and_stop
        limits = self._catalog.limits()
        self._max_depth = max_depth if max_depth is not None else limits.get("max_depth", 12)
        self._min_fallback_children = limits.get("min_fallback_children", 2)

    # ── Introspection helpers (for the LLM / humans) ────────────────────────

    def list_skills(self) -> list[str]:
        """Return the sorted list of valid skill IDs."""
        return sorted(self._valid_skills)

    def describe_skill(self, skill_id: str) -> dict[str, Any]:
        """Return port spec + description for a skill, or raise UnknownSkillError."""
        spec = self._catalog.get_skill(skill_id)
        if spec is None:
            raise UnknownSkillError(f"Unknown skill: '{skill_id}'")
        return {
            "id": spec.id,
            "family": spec.family,
            "bt_tag": spec.bt_tag,
            "ports": {
                p: {
                    "type": ps.type,
                    "required": ps.required,
                    "default": ps.default,
                    "allowed": list(ps.allowed) if ps.allowed else None,
                }
                for p, ps in spec.ports.items()
            },
            "description": spec.description,
        }

    # ── Low-level: leaf skill ───────────────────────────────────────────────

    def skill(self, skill_id: str, name: str | None = None, **ports: Any) -> SkillNode:
        """
        Build a skill leaf node (Action or Condition — inferred from catalog).

        Validates:
          - skill_id is in the catalog (else UnknownSkillError)
          - required ports are present (else PortError)
          - port values respect their declared type + allowed domain (else PortError)
          - no unknown ports are passed (else PortError)
        """
        spec = self._catalog.get_skill(skill_id)
        if spec is None:
            raise UnknownSkillError(
                f"Unknown skill '{skill_id}'. Available: "
                f"{', '.join(sorted(self._valid_skills))}"
            )

        str_ports = {k: str(v) for k, v in ports.items()}
        self._check_ports(spec, str_ports)

        # Apply defaults for required ports that were not specified.
        port_spec = self._skill_ports.get(skill_id, {})
        required = port_spec.get("required", [])
        for port_name in required:
            if port_name not in str_ports:
                default = (
                    spec.ports[port_name].default if port_name in spec.ports else None
                )
                if default is None:
                    raise PortError(
                        f"Skill '{skill_id}' requires port '{port_name}' and no "
                        f"default is registered in the catalog"
                    )
                str_ports[port_name] = default

        return SkillNode(
            skill_id=skill_id, bt_tag=spec.bt_tag, name=name, ports=str_ports
        )

    def _check_ports(self, spec: SkillSpec, ports: dict[str, str]) -> None:
        port_spec = self._skill_ports.get(spec.id, {})
        types = port_spec.get("types", {})
        allowed = port_spec.get("allowed", {})
        known = set(types.keys())

        for port_name in ports:
            if port_name not in known:
                raise PortError(
                    f"Skill '{spec.id}' has no port named '{port_name}'. "
                    f"Valid ports: {sorted(known) or '[]'}"
                )

        for port_name, val in ports.items():
            ptype = types.get(port_name, "bb_var")
            if ptype == "bb_var":
                if not (val.startswith("{") and val.endswith("}")):
                    raise PortError(
                        f"Skill '{spec.id}' port '{port_name}' expects a blackboard "
                        f"variable like '{{name}}', got '{val}'"
                    )
            elif ptype == "int_literal":
                try:
                    int(val)
                except ValueError as e:
                    raise PortError(
                        f"Skill '{spec.id}' port '{port_name}' expects an integer, "
                        f"got '{val}'"
                    ) from e
            elif ptype == "float_literal":
                try:
                    float(val)
                except ValueError as e:
                    raise PortError(
                        f"Skill '{spec.id}' port '{port_name}' expects a float, "
                        f"got '{val}'"
                    ) from e
            if port_name in allowed and val not in allowed[port_name]:
                raise PortError(
                    f"Skill '{spec.id}' port '{port_name}'='{val}' is out of its "
                    f"allowed domain {sorted(allowed[port_name])}"
                )

    # ── Low-level: control nodes ────────────────────────────────────────────

    def sequence(self, *children: Node, name: str | None = None) -> ControlNode:
        if not children:
            raise StructuralError("Sequence must have at least one child")
        return ControlNode(tag="Sequence", name=name, children=list(children))

    def fallback(self, *children: Node, name: str | None = None) -> ControlNode:
        if len(children) < self._min_fallback_children:
            raise StructuralError(
                f"Fallback requires at least {self._min_fallback_children} children, "
                f"got {len(children)}"
            )
        return ControlNode(tag="Fallback", name=name, children=list(children))

    def reactive_fallback(
        self, *children: Node, name: str | None = None
    ) -> ControlNode:
        if len(children) < self._min_fallback_children:
            raise StructuralError(
                f"ReactiveFallback requires at least {self._min_fallback_children} "
                f"children, got {len(children)}"
            )
        return ControlNode(tag="ReactiveFallback", name=name, children=list(children))

    def parallel(self, *children: Node, name: str | None = None) -> ControlNode:
        if not children:
            raise StructuralError("Parallel must have at least one child")
        return ControlNode(tag="Parallel", name=name, children=list(children))

    def repeat(
        self,
        child: Node,
        num_cycles: int = -1,
        name: str | None = None,
    ) -> RepeatNode:
        if child is None:
            raise StructuralError("Repeat must have exactly one child")
        return RepeatNode(num_cycles=num_cycles, name=name, child=child)

    def subtree_plus(
        self,
        subtree_id: str,
        name: str | None = None,
        autoremap: bool = True,
        **extra_ports: Any,
    ) -> SubTreePlusNode:
        return SubTreePlusNode(
            subtree_id=subtree_id,
            name=name,
            autoremap=autoremap,
            extra_ports={k: str(v) for k, v in extra_ports.items()},
        )

    # ── Register a BehaviorTree ─────────────────────────────────────────────

    def register_behavior_tree(self, bt_id: str, root: Node) -> "MissionBuilder":
        """Register a <BehaviorTree ID="bt_id"> subtree. Returns self for chaining."""
        if bt_id in self._registered_ids:
            raise StructuralError(f"BehaviorTree with ID '{bt_id}' already registered")
        self._behavior_trees.append(BehaviorTreeNode(bt_id=bt_id, root=root))
        self._registered_ids.add(bt_id)
        return self

    # ── High-level patterns (compose low-level, enforce L5) ────────────────

    def add_get_mission(self, bt_id: str = "get_mission") -> "MissionBuilder":
        """<BehaviorTree ID="get_mission"> with LoadMission + MissionStructureValid."""
        root = self.sequence(
            self.skill(
                "LoadMission",
                name="LOAD MISSION",
                mission_file_path="{mission_file_path}",
            ),
            self.skill("MissionStructureValid", name="MISSION STRUCTURE VALID"),
            name="GET MISSION",
        )
        return self.register_behavior_tree(bt_id, root)

    def add_calculate_path(self, bt_id: str = "calculate_path") -> "MissionBuilder":
        """SR-026: Fallback(Repeat(-1)(Sequence(...)), MissionFullyTreated)."""
        inner = self.sequence(
            self.skill(
                "UpdateCurrentGeneratedActivity",
                name="UPDATE ACTIVITY",
                type="{type}",
                origin_sph="{origin_sph}",
                target_sph="{target_sph}",
                forbidden_atoms_out="{forbidden_atoms}",
            ),
            self.skill(
                "ProjectPointOnNetwork",
                name="PROJECT ORIGIN",
                point_in="{origin_sph}",
                point_out="{origin}",
            ),
            self.skill(
                "ProjectPointOnNetwork",
                name="PROJECT TARGET",
                point_in="{target_sph}",
                point_out="{target}",
            ),
            self.skill(
                "CreatePath",
                name="CREATE PATH",
                origin="{origin}",
                target="{target}",
                forbidden_atoms="{forbidden_atoms}",
                path="{path}",
            ),
            self.skill("AgregatePath", name="AGREGATE PATH", path="{path}"),
            name="ACTIVITY",
        )
        loop = self.repeat(inner, num_cycles=-1, name="LOOP")
        root = self.fallback(
            loop,
            self.skill("MissionFullyTreated", name="MISSION FULLY TREATED", type="{type}"),
            name="PATH CALCULATION",
        )
        return self.register_behavior_tree(bt_id, root)

    def add_base_preparation(
        self,
        bt_id: str = "base_preparation",
        get_mission_id: str = "get_mission",
        calculate_path_id: str = "calculate_path",
    ) -> "MissionBuilder":
        """Preparation subtree: get mission + calculate path + pass data."""
        root = self.sequence(
            self.subtree_plus(get_mission_id, name="GET MISSION"),
            self.subtree_plus(calculate_path_id, name="CALCULATE PATH"),
            self.skill("PassAdvancedPath", name="PASS ADVANCED PATH", adv_path="{adv_path}"),
            self.skill("PassMission", name="PASS MISSION", mission="{mission}"),
            self.skill(
                "GenerateMissionSequence",
                name="GENERATE MISSION SEQUENCE",
                mission="{mission}",
                mission_sequence="{mission_sequence}",
            ),
            name="BASE PREPARATION",
        )
        return self.register_behavior_tree(bt_id, root)

    def add_motion_subtree(
        self,
        step_type: int,
        bt_id: str | None = None,
        name: str | None = None,
        signal_message: str | None = None,
    ) -> "MissionBuilder":
        """
        Add a motion subtree for the given step_type (SR-027 pattern).

        Supported:
          Transport: 0 (move), 1 (deccelerate), 2 (reach_and_stop),
                     3 (pass),  4 (reach_stop_and_dont_wait)
          Inspection: 10 (move_and_inspect), 11 (deccelerate_and_inspect),
                      12 (reach_and_stop_inspecting), 13 (pass_and_stop_inspecting),
                      14 (reach_stop_inspecting_dont_wait)
        """
        if step_type not in _STEP_TYPE_DEFAULT_NAMES:
            raise StructuralError(
                f"Unknown step_type: {step_type}. "
                f"Valid: {sorted(_STEP_TYPE_DEFAULT_NAMES)}"
            )
        default_id, default_name = _STEP_TYPE_DEFAULT_NAMES[step_type]
        bt_id = bt_id or default_id
        display_name = name or default_name

        check = self.skill(
            "CheckCurrentStepType",
            name=f"IS CURRENT STEP {display_name}",
            type_to_be_checked=str(step_type),
        )
        pass_params = self.skill(
            "PassMotionParameters",
            name="PASS MOTION PARAMETERS",
            motion_params="{motion_params}",
        )
        update_step = self.skill("UpdateCurrentExecutedStep", name="UPDATE CURRENT STEP")

        if step_type == 0:  # move
            core: list[Node] = [
                self.skill("Move", name="MOVE", threshold_type="1", motion_params="{motion_params}"),
            ]
        elif step_type == 1:  # deccelerate
            core = [
                self.skill("Deccelerate", name="DECCELERATE", motion_params="{motion_params}"),
            ]
        elif step_type == 2:  # reach_and_stop
            msg = signal_message or "need authorization to go further"
            core = [
                self.skill("MoveAndStop", name="MOVE AND STOP", motion_params="{motion_params}"),
                self.skill(
                    "SignalAndWaitForOrder",
                    name="SIGNAL AND WAIT FOR AUTHORIZATION",
                    message=msg,
                ),
            ]
        elif step_type == 3:  # pass
            core = [
                self.skill("Move", name="MOVE", threshold_type="3", motion_params="{motion_params}"),
            ]
        elif step_type == 4:  # reach_stop_and_dont_wait
            core = [
                self.skill("MoveAndStop", name="MOVE AND STOP", motion_params="{motion_params}"),
            ]
        elif step_type == 10:  # move_and_inspect
            core = [
                self.skill("Pause", name="PAUSE", duration="2.0"),
                self.skill("ManageMeasurements", name="START INSPECTION"),
                self.skill("Move", name="MOVE", threshold_type="1", motion_params="{motion_params}"),
            ]
        elif step_type == 11:  # deccelerate_and_inspect
            core = [
                self.skill("Deccelerate", name="DECCELERATE", motion_params="{motion_params}"),
            ]
        elif step_type == 12:  # reach_and_stop_inspecting  (SR-024)
            core = [
                self.skill("MoveAndStop", name="MOVE AND STOP", motion_params="{motion_params}"),
                self.skill("ManageMeasurements", name="STOP INSPECTION"),
                self.skill("AnalyseMeasurements", name="ANALYSE MEASUREMENTS"),
                self.sequence(
                    self.fallback(
                        self.skill(
                            "MeasurementsQualityValidated",
                            name="IS MEASUREMENT QUALITY OK",
                        ),
                        self.skill(
                            "PassDefectsLocalization",
                            name="PASS DEFECTS LOCALIZATION",
                            defects="{defects}",
                        ),
                        name="CORRECTIVE SEQUENCE",
                    ),
                    self.skill(
                        "GenerateCorrectiveSubSequence",
                        name="GENERATE CORRECTIVE SEQUENCE",
                        defects="{defects}",
                    ),
                    self.skill(
                        "InsertCorrectiveSubSequence",
                        name="INSERT CORRECTIVE SEQUENCE",
                    ),
                    name="REACT ON QUALITY",
                ),
            ]
        elif step_type == 13:  # pass_and_stop_inspecting (SR-024)
            core = [
                self.skill("Move", name="MOVE", threshold_type="3", motion_params="{motion_params}"),
                self.skill("ManageMeasurements", name="STOP INSPECTION"),
                self.fallback(
                    self.skill("AnalyseMeasurements", name="ANALYSE MEASUREMENTS"),
                    self.skill(
                        "MeasurementsEnforcedValidated", name="ENFORCED VALIDATION"
                    ),
                    name="ENFORCED ANALYSIS",
                ),
            ]
        elif step_type == 14:  # reach_stop_inspecting_dont_wait (SR-024)
            core = [
                self.skill("MoveAndStop", name="MOVE AND STOP", motion_params="{motion_params}"),
                self.skill("ManageMeasurements", name="STOP INSPECTION"),
                self.skill("AnalyseMeasurements", name="ANALYSE MEASUREMENTS"),
                self.fallback(
                    self.skill(
                        "MeasurementsQualityValidated",
                        name="IS MEASUREMENTS QUALITY OK",
                    ),
                    self.sequence(
                        self.skill(
                            "PassDefectsLocalization",
                            name="PASS DEFECTS LOCALIZATION",
                            defects="{defects}",
                        ),
                        self.skill(
                            "GenerateCorrectiveSubSequence",
                            name="GENERATE CORRECTIVE SEQUENCE",
                            defects="{defects}",
                        ),
                        self.skill(
                            "InsertCorrectiveSubSequence",
                            name="INSERT CORRECTIVE SEQUENCE",
                        ),
                        name="REACT ON MEASUREMENTS QUALITY",
                    ),
                    name="CORRECTIVE SEQUENCE",
                ),
            ]
        else:  # pragma: no cover — unreachable due to the lookup above
            raise StructuralError(f"Unhandled step_type: {step_type}")

        root = self.sequence(check, pass_params, *core, update_step, name=display_name)
        return self.register_behavior_tree(bt_id, root)

    def add_execute(
        self,
        step_types: list[int],
        bt_id: str = "execute",
        name: str = "EXECUTE",
    ) -> "MissionBuilder":
        """
        SR-025: ReactiveFallback(Repeat(-1)(Fallback("motion_selector",
        SubTreePlus per step_type)), MissionTerminated).

        Also registers each motion subtree referenced, unless it is already
        registered (idempotent).
        """
        if not step_types:
            raise StructuralError("add_execute requires at least one step_type")

        for st in step_types:
            if st not in _STEP_TYPE_DEFAULT_NAMES:
                raise StructuralError(
                    f"Unknown step_type {st}. Valid: {sorted(_STEP_TYPE_DEFAULT_NAMES)}"
                )
            subtree_id = _STEP_TYPE_DEFAULT_NAMES[st][0]
            if subtree_id not in self._registered_ids:
                self.add_motion_subtree(st, bt_id=subtree_id)

        motion_selector_children = [
            self.subtree_plus(
                _STEP_TYPE_DEFAULT_NAMES[st][0],
                name=_STEP_TYPE_DEFAULT_NAMES[st][1],
            )
            for st in step_types
        ]
        if len(motion_selector_children) < self._min_fallback_children:
            raise StructuralError(
                f"add_execute requires at least {self._min_fallback_children} distinct "
                f"step_types (got {len(motion_selector_children)})"
            )

        motion_selector = self.fallback(
            *motion_selector_children, name="MOTION SELECTOR"
        )
        loop = self.repeat(motion_selector, num_cycles=-1, name="SEQUENCE STEP LOOP")
        root = self.reactive_fallback(
            loop,
            self.skill("MissionTerminated", name="IS MISSION TERMINATED"),
            name=name,
        )
        return self.register_behavior_tree(bt_id, root)

    def add_main_tree(
        self,
        preparation_id: str = "base_preparation",
        execute_id: str = "execute",
        name: str = "MISSION",
    ) -> "MissionBuilder":
        """Register the main tree: Sequence(preparation + execute via SubTreePlus)."""
        if self._main_tree_id in self._registered_ids:
            raise StructuralError(
                f"Main tree '{self._main_tree_id}' already registered"
            )
        root = self.sequence(
            self.subtree_plus(preparation_id, name="PREPARATION"),
            self.subtree_plus(execute_id, name="EXECUTE"),
            name=name,
        )
        return self.register_behavior_tree(self._main_tree_id, root)

    # ── Serialization ────────────────────────────────────────────────────────

    def to_xml(self, enrich: bool = True) -> str:
        """
        Serialize the registered behavior trees to a BTCPP v4 XML string.

        Runs final checks:
          - main tree is registered (else StructuralError)
          - MoveAndStop is present in at least one skill (else MissingRequiredSkillError)
          - global depth <= max_depth (else StructuralError)

        If enrich=True, runs validate_bt.enrich_ports to fill missing default ports
        (no-op for nodes already fully specified).
        """
        if self._main_tree_id not in self._registered_ids:
            raise StructuralError(
                f"Main tree '{self._main_tree_id}' is not registered — call "
                f"add_main_tree() or register_behavior_tree() before to_xml()"
            )

        root = ET.Element("root")
        root.set("BTCPP_format", self._btcpp_format)
        root.set("main_tree_to_execute", self._main_tree_id)

        # Place the main tree first for readability, then the rest.
        ordered: list[BehaviorTreeNode] = []
        main_first = [bt for bt in self._behavior_trees if bt.bt_id == self._main_tree_id]
        others = [bt for bt in self._behavior_trees if bt.bt_id != self._main_tree_id]
        ordered.extend(main_first)
        ordered.extend(others)

        for bt in ordered:
            root.append(bt.to_element())

        if self._require_move_and_stop:
            has_move_and_stop = any(
                (e.tag == "MoveAndStop")
                or (e.tag == "Action" and e.get("ID") == "MoveAndStop")
                for e in root.iter()
            )
            if not has_move_and_stop:
                raise MissingRequiredSkillError(
                    "MoveAndStop is absent — the BT cannot safely terminate. "
                    "Every mission requires at least one MoveAndStop node."
                )

        def _depth(elem: ET.Element, d: int = 0) -> int:
            children = list(elem)
            if not children:
                return d
            return max(_depth(c, d + 1) for c in children)

        depth = _depth(root)
        if depth > self._max_depth:
            raise StructuralError(
                f"Tree depth {depth} exceeds the limit of {self._max_depth} "
                f"(defined in skills_catalog.yaml.limits.max_depth)"
            )

        ET.indent(root, space="  ")
        xml_str = ET.tostring(root, encoding="unicode")

        if enrich:
            # Import locally to avoid a circular import at module load.
            from src.eval.validate_bt import enrich_ports

            xml_str = enrich_ports(xml_str, self._catalog)

        return xml_str
