"""
NAV4RAIL Mission Builder — a strict Python API for constructing BTCPP v4
Behavior Trees. Designed for Program-of-Thoughts (PoT) and ReAct agents:
the LLM writes a Python script against this API, the interpreter enforces
L1/L2/L3 constraints at construction time.

Public API:
    from src.builder import MissionBuilder, BuilderError

Example:
    builder = MissionBuilder(main_tree_id="transport")
    builder.add_get_mission()
    builder.add_calculate_path()
    builder.add_base_preparation()
    builder.add_motion_subtree(step_type=2)  # reach_and_stop
    builder.add_execute([2])
    builder.add_main_tree()
    xml = builder.to_xml()
"""

from src.builder.mission_builder import (
    BuilderError,
    MissionBuilder,
    MissingRequiredSkillError,
    PortError,
    StructuralError,
    UnknownSkillError,
)

__all__ = [
    "BuilderError",
    "MissingRequiredSkillError",
    "MissionBuilder",
    "PortError",
    "StructuralError",
    "UnknownSkillError",
]
