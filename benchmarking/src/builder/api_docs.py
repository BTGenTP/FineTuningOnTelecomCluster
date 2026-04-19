"""
API documentation exporter for MissionBuilder.
==============================================

Produces the documentation block that is injected into LLM prompts
(Program-of-Thoughts and ReAct agents). Pulled dynamically from the
SkillsCatalog so the docs can never drift from the code.
"""

from __future__ import annotations

from src.builder.mission_builder import _STEP_TYPE_DEFAULT_NAMES
from src.data.skills_loader import SkillsCatalog


API_PREAMBLE = """\
NAV4RAIL MissionBuilder — Python API reference
==============================================

You write a Python script that uses `MissionBuilder` to construct a
BehaviorTree.CPP v4 XML for a NAV4RAIL railway mission. The script MUST
`print(builder.to_xml())` at the end. Any error at construction time
(unknown skill, missing port, fallback with a single child, ...) will
raise an exception — treat those exceptions as bugs you must fix.

Imports and entry point (already available in the sandbox, do NOT re-import):
    from nav4rail_builder import MissionBuilder

Only functions of `MissionBuilder` may be called. I/O is forbidden.
"""

CONSTRUCTION = """\
Constructor:
    builder = MissionBuilder(main_tree_id="generated_mission")

Low-level API (composable):
    node = builder.skill("SkillName", name="OPTIONAL DESCRIPTION", **ports)
    node = builder.sequence(child1, child2, ..., name="...")
    node = builder.fallback(child1, child2, ..., name="...")        # >=2 children
    node = builder.reactive_fallback(child1, child2, ..., name="...")# >=2 children
    node = builder.parallel(child1, child2, ..., name="...")
    node = builder.repeat(child, num_cycles=-1, name="...")         # one child only
    node = builder.subtree_plus("subtree_id", name="...", **extra_ports)

Register a BehaviorTree subtree (returns self, chainable):
    builder.register_behavior_tree("my_subtree_id", root_node)

High-level helpers (enforce SR-023..SR-027 by construction):
    builder.add_get_mission()                     # LoadMission + MissionStructureValid
    builder.add_calculate_path()                  # path loop + MissionFullyTreated
    builder.add_base_preparation()                # full preparation subtree
    builder.add_motion_subtree(step_type=N, ...)  # one motion subtree
    builder.add_execute(step_types=[0, 2, 12])    # executor + all motion subtrees
    builder.add_main_tree()                       # Sequence(preparation, execute)

Serialize:
    xml = builder.to_xml()
    print(xml)
"""

EXAMPLE = """\
Minimal transport mission (type 0 move + type 2 reach_and_stop):
    from nav4rail_builder import MissionBuilder
    b = MissionBuilder(main_tree_id="my_mission")
    b.add_get_mission()
    b.add_calculate_path()
    b.add_base_preparation()
    b.add_execute(step_types=[0, 2])   # auto-registers motion subtrees
    b.add_main_tree()
    print(b.to_xml())

Full inspection mission with quality control (types 0, 2, 12):
    b = MissionBuilder(main_tree_id="inspection_with_control")
    b.add_get_mission()
    b.add_calculate_path()
    b.add_base_preparation()
    b.add_execute(step_types=[0, 2, 12])   # type 12 triggers SR-024 inspection pattern
    b.add_main_tree()
    print(b.to_xml())
"""

STEP_TYPE_GUIDE = """\
Step types (SR-016, SR-024):
  Transport (always add at least one):
    0  move                         — continuous motion
    1  deccelerate                  — gradual slow-down
    2  reach_and_stop               — stop + SignalAndWaitForOrder
    3  pass                         — passage without stopping
    4  reach_stop_and_dont_wait     — stop without waiting

  Inspection (add ONLY for missions mentioning 'inspecter', 'mesurer', 'controler'):
    10 move_and_inspect             — Pause + ManageMeasurements(start) + Move
    11 deccelerate_and_inspect
    12 reach_and_stop_inspecting    — requires AnalyseMeasurements + corrective Fallback
    13 pass_and_stop_inspecting     — requires Fallback(AnalyseMeasurements, Enforced)
    14 reach_stop_inspecting_dont_wait — like 12, no Signal
"""


def get_skills_reference(catalog: SkillsCatalog | None = None) -> str:
    """Build a compact skills + ports reference, grouped by family."""
    cat = catalog or SkillsCatalog()
    lines = ["Skills reference (name → ports):"]
    for family_name, skill_ids in cat.families().items():
        lines.append(f"  [{family_name}]")
        for sid in skill_ids:
            spec = cat.get_skill(sid)
            if spec is None:
                continue
            tag = spec.bt_tag
            if spec.ports:
                parts = []
                for pname, pspec in spec.ports.items():
                    mark = "*" if pspec.required else ""
                    piece = f"{pname}{mark}:{pspec.type}"
                    if pspec.allowed:
                        piece += f"={list(pspec.allowed)}"
                    parts.append(piece)
                port_str = "(" + ", ".join(parts) + ")"
            else:
                port_str = "()"
            lines.append(f"    {sid} [{tag}]{port_str}")
    lines.append("  Legend: *=required, bb_var='{name}', int_literal/float_literal=literal, "
                 "string_literal=plain text")
    return "\n".join(lines)


def get_full_api_docs(catalog: SkillsCatalog | None = None) -> str:
    """Assemble the complete API docs string for prompt injection."""
    valid_step_types = sorted(_STEP_TYPE_DEFAULT_NAMES.keys())
    return "\n\n".join(
        [
            API_PREAMBLE.rstrip(),
            CONSTRUCTION.rstrip(),
            STEP_TYPE_GUIDE.rstrip(),
            f"Valid step_types for add_motion_subtree / add_execute: {valid_step_types}",
            get_skills_reference(catalog),
            EXAMPLE.rstrip(),
        ]
    )
