from __future__ import annotations

from typing import Any, Mapping

from finetune_Nav2.catalog.catalog_io import allowed_skills


def _gbnf_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def build_nav2_bt_xml_gbnf(catalog: Mapping[str, Any]) -> str:
    """
    Minimal GBNF grammar for Nav2 BT XML (BehaviorTree.CPP style).

    Notes:
    - Focuses on tag allowlist (skills) and overall structure.
    - Does NOT fully enforce per-skill attributes; strict validation is still required.
    """
    skills = sorted(allowed_skills(catalog).keys())
    if not skills:
        raise ValueError("Catalog has no atomic_skills.")

    # Map skill_id -> bt_tag; multiple skills may share bt_tag (e.g., ClearEntireCostmapLocal/Global).
    bt_tags = sorted({str(it.get("bt_tag")) for it in allowed_skills(catalog).values() if it.get("bt_tag")})
    # Also allow SubTree (navigation) and common control nodes used in the reference subtree.
    ctrl = [
        "Sequence",
        "Fallback",
        "ReactiveSequence",
        "ReactiveFallback",
        "RoundRobin",
        "PipelineSequence",
        "RateController",
        "KeepRunningUntilFailure",
        "Repeat",
        "Inverter",
        "RecoveryNode",
    ]
    tags = sorted(set(bt_tags + ["SubTree"] + ctrl))

    tag_alts = " | ".join([f'"{_gbnf_escape(t)}"' for t in tags])

    return f"""
root ::= "<root main_tree_to_execute=\\"MainTree\\">\\n  <BehaviorTree ID=\\"MainTree\\">\\n" node+ "  </BehaviorTree>\\n" subtree_defs? "</root>"

node ::= seq | leaf | ctrl_node
seq ::= "    <Sequence name=\\"" name "\\">\\n" node+ "    </Sequence>\\n"

ctrl_node ::= "    <" ctrl_tag ctrl_attrs? ">\\n" node+ "    </" ctrl_tag ">\\n"
ctrl_tag ::= "KeepRunningUntilFailure" | "Fallback" | "ReactiveFallback" | "ReactiveSequence" | "RoundRobin" | "PipelineSequence" | "RateController" | "Repeat" | "Inverter" | "RecoveryNode"
ctrl_attrs ::= (" " attr_kv)+

leaf ::= "      <" leaf_tag leaf_attrs? "/>\\n"
leaf_tag ::= {tag_alts}
leaf_attrs ::= (" " attr_kv)+

attr_kv ::= attr_name "=\\"" attr_val "\\""
attr_name ::= [A-Za-z_][A-Za-z0-9_\\-]*
attr_val ::= [^"\\n\\r\\t]+

subtree_defs ::= "\\n" (subtree_def)+
subtree_def ::= "  <BehaviorTree ID=\\"" name "\\">\\n" any_inner+ "  </BehaviorTree>\\n"
any_inner ::= [\\s\\S]
any_inner+ ::= any_inner any_inner*

name ::= [A-Za-z_][A-Za-z0-9_\\-]*
""".strip() + "\n"

