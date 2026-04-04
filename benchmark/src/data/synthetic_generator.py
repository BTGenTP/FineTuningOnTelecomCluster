from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List
from xml.etree.ElementTree import Element, SubElement, tostring


PROMPT_TEMPLATES = [
    "La mission consiste a preparer la trajectoire, executer les mouvements et appliquer les controles d'inspection si necessaire.",
    "Le robot doit charger une mission ferroviaire, calculer sa trajectoire, puis executer la sequence en respectant les contraintes NAV4RAIL.",
    "La mission necessite de franchir un passage a niveau avec ralentissement, puis d'inspecter la zone critique.",
]


@dataclass(slots=True)
class GenerationContext:
    seed: int
    mission_name: str

    @property
    def vars(self) -> Dict[str, str]:
        return {
            "mission": "{mission}",
            "mission_file": "{mission_file}",
            "mission_sequence": "{mission_sequence}",
            "adv_path": "{adv_path}",
            "type": "{type}",
            "origin_sph": "{origin_sph}",
            "target_sph": "{target_sph}",
            "origin": "{origin}",
            "target": "{target}",
            "forbidden_atoms": "{forbidden_atoms}",
            "path": "{path}",
            "motion_params": "{motion_params}",
            "defects": "{defects}",
        }


def _action(parent: Element, tag: str, **attrs: str) -> Element:
    return SubElement(parent, tag, attrs)


def generate_preparation_pattern(ctx: GenerationContext) -> Element:
    seq = Element("Sequence", {"name": "BASE PREPARATION"})
    _action(seq, "Action", name="LOAD MISSION", ID="LoadMission", mission_file_path=ctx.vars["mission_file"])
    _action(seq, "Condition", name="IS MISSION STRUCTURE VALID", ID="MissionStructureValid")
    _action(
        seq,
        "Action",
        name="UPDATE CURRENT ACTIVITY",
        ID="UpdateCurrentGeneratedActivity",
        type=ctx.vars["type"],
        origin_sph=ctx.vars["origin_sph"],
        target_sph=ctx.vars["target_sph"],
        forbidden_atoms_out=ctx.vars["forbidden_atoms"],
    )
    _action(seq, "Action", name="PROJECT ORIGIN", ID="ProjectPointOnNetwork", point_in=ctx.vars["origin_sph"], point_out=ctx.vars["origin"])
    _action(seq, "Action", name="PROJECT TARGET", ID="ProjectPointOnNetwork", point_in=ctx.vars["target_sph"], point_out=ctx.vars["target"])
    _action(
        seq,
        "Action",
        name="CREATE PATH",
        ID="CreatePath",
        origin=ctx.vars["origin"],
        target=ctx.vars["target"],
        forbidden_atoms=ctx.vars["forbidden_atoms"],
        path=ctx.vars["path"],
    )
    _action(seq, "Action", name="AGREGATE PATH", ID="AgregatePath", path=ctx.vars["path"])
    _action(seq, "Action", name="PASS ADVANCED PATH", ID="PassAdvancedPath", adv_path=ctx.vars["adv_path"])
    _action(seq, "Action", name="PASS MISSION", ID="PassMission", mission=ctx.vars["mission"])
    _action(
        seq,
        "Action",
        name="GENERATE MOTION SEQUENCE",
        ID="GenerateMissionSequence",
        mission=ctx.vars["mission"],
        mission_sequence=ctx.vars["mission_sequence"],
    )
    return seq


def generate_inspection_pattern(ctx: GenerationContext, *, step_type: str, move_node_id: str) -> Element:
    seq = Element("Sequence", {"name": "INSPECTION MOTION"})
    _action(seq, "Condition", name="CHECK STEP TYPE", ID="CheckCurrentStepType", type_to_be_checked=step_type)
    _action(seq, "Action", name="PASS MOTION PARAMETERS", ID="PassMotionParameters", motion_params=ctx.vars["motion_params"])
    _action(seq, "Action", name="START INSPECTION", ID="ManageMeasurements", mode="start")
    _action(seq, "Action", name="MOVE", ID=move_node_id, motion_params=ctx.vars["motion_params"], threshold_type="1" if move_node_id == "Move" else ctx.vars["motion_params"])
    if move_node_id == "MoveAndStop":
        seq[-1].attrib.pop("threshold_type", None)
    _action(seq, "Action", name="STOP INSPECTION", ID="ManageMeasurements", mode="stop")
    _action(seq, "Action", name="ANALYSE MEASUREMENTS", ID="AnalyseMeasurements")
    fallback = _action(seq, "Fallback", name="INSPECTION RECOVERY")
    _action(fallback, "Condition", name="QUALITY VALIDATED", ID="MeasurementsQualityValidated")
    corrective = _action(fallback, "Sequence", name="CORRECTIVE BRANCH")
    _action(corrective, "Action", name="PASS DEFECTS", ID="PassDefectsLocalization", defects=ctx.vars["defects"])
    _action(corrective, "Action", name="GENERATE CORRECTIVE SUBSEQUENCE", ID="GenerateCorrectiveSubSequence", defects=ctx.vars["defects"])
    _action(corrective, "Action", name="INSERT CORRECTIVE SUBSEQUENCE", ID="InsertCorrectiveSubSequence")
    _action(seq, "Action", name="UPDATE CURRENT STEP", ID="UpdateCurrentExecutedStep")
    return seq


def generate_execution_loop(ctx: GenerationContext) -> Element:
    root = Element("ReactiveFallback", {"name": "EXECUTE"})
    repeat = _action(root, "Repeat", name="SEQUENCE STEP LOOP", num_cycles="-1")
    selector = _action(repeat, "Fallback", name="MOTION SELECTOR")

    move = _action(selector, "SubTreePlus", name="MOVE", ID="move", __autoremap="true")
    move_inspect = _action(selector, "SubTreePlus", name="MOVE AND INSPECT", ID="move_and_inspect", __autoremap="true")
    decelerate = _action(selector, "SubTreePlus", name="DECCELERATE", ID="deccelerate", __autoremap="true")
    reach_stop = _action(selector, "SubTreePlus", name="REACH AND STOP INSPECTING", ID="reach_and_stop_inspecting", __autoremap="true")
    assert move is not None and move_inspect is not None and decelerate is not None and reach_stop is not None
    _action(root, "Condition", name="IS MISSION TERMINATED", ID="MissionTerminated")
    return root


def _behavior_tree(root: Element, tree_id: str, body: Element) -> None:
    bt = SubElement(root, "BehaviorTree", {"ID": tree_id})
    bt.append(body)


def build_nav4rail_tree(mission_name: str = "synthetic_inspection_mission") -> str:
    ctx = GenerationContext(seed=42, mission_name=mission_name)
    root = Element("root", {"main_tree_to_execute": mission_name})
    main_seq = Element("Sequence", {"name": "SYNTHETIC NAV4RAIL"})
    _action(main_seq, "SubTreePlus", name="PREPARATION", ID="preparation", __autoremap="true")
    _action(main_seq, "SubTreePlus", name="EXECUTE", ID="execute", __autoremap="true")
    _behavior_tree(root, mission_name, main_seq)
    _behavior_tree(root, "preparation", generate_preparation_pattern(ctx))
    _behavior_tree(root, "execute", generate_execution_loop(ctx))

    move = Element("Sequence", {"name": "MOVE"})
    _action(move, "Condition", name="IS CURRENT STEP MOVE", ID="CheckCurrentStepType", type_to_be_checked="0")
    _action(move, "Action", name="PASS MOTION PARAMETERS", ID="PassMotionParameters", motion_params=ctx.vars["motion_params"])
    _action(move, "Action", name="MOVE", ID="Move", threshold_type="1", motion_params=ctx.vars["motion_params"])
    _action(move, "Action", name="UPDATE CURRENT STEP", ID="UpdateCurrentExecutedStep")
    _behavior_tree(root, "move", move)

    _behavior_tree(root, "move_and_inspect", generate_inspection_pattern(ctx, step_type="10", move_node_id="Move"))

    dec = Element("Sequence", {"name": "DECCELERATE"})
    _action(dec, "Condition", name="IS CURRENT STEP DECCELERATE", ID="CheckCurrentStepType", type_to_be_checked="1")
    _action(dec, "Action", name="PASS MOTION PARAMETERS", ID="PassMotionParameters", motion_params=ctx.vars["motion_params"])
    _action(dec, "Action", name="DECCELERATE", ID="Deccelerate", motion_params=ctx.vars["motion_params"])
    _action(dec, "Action", name="UPDATE CURRENT STEP", ID="UpdateCurrentExecutedStep")
    _behavior_tree(root, "deccelerate", dec)

    _behavior_tree(root, "reach_and_stop_inspecting", generate_inspection_pattern(ctx, step_type="12", move_node_id="MoveAndStop"))
    return tostring(root, encoding="unicode")


def generate_record(seed: int) -> Dict[str, Any]:
    random.seed(seed)
    mission = random.choice(PROMPT_TEMPLATES)
    xml = build_nav4rail_tree(mission_name=f"synthetic_inspection_{seed}")
    return {
        "mission": mission,
        "xml": xml,
        "metadata": {
            "seed": seed,
            "pattern_family": "design_patterns",
            "uses_blackboard": True,
            "contains_inspection_branch": True,
        },
    }


def generate_dataset(total: int) -> List[Dict[str, Any]]:
    return [generate_record(seed) for seed in range(total)]


def iter_dataset(total: int) -> Iterator[Dict[str, Any]]:
    for seed in range(total):
        yield generate_record(seed)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path
