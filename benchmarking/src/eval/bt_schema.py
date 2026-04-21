"""
bt_schema.py — Pydantic schema for Outlines JSON-mode constrained decoding.
============================================================================
Mirror of `skills_catalog.yaml` as a recursive Pydantic model. Used by
Outlines' JSONLogitsProcessor to force the LLM to emit valid BT JSON,
which is then converted to BTCPP v4 XML by `json_to_xml()`.

Why JSON + conversion rather than direct XML?
  - Outlines' JSONLogitsProcessor is mature and fast (FSM compiled once).
  - JSON Schema can express "one-of skill-ID from enum" cleanly.
  - The XML writer is trivial (~50 LOC) and deterministic.

If you prefer constraining XML directly, use GBNF mode instead (see
`src/eval/bt_grammar.gbnf` + `transformers-cfg`).

The enums below are auto-discovered from the catalog at import time — no
hard-coded lists to keep in sync.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ── Enum population from the catalog ─────────────────────────────────────────

def _load_ids() -> tuple[list[str], list[str]]:
    """Return (action_ids, condition_ids) from skills_catalog.yaml."""
    import yaml

    bench_root = Path(__file__).parent.parent.parent
    catalog_path = bench_root / "data" / "skills_catalog.yaml"
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    actions, conditions = [], []
    for fdata in catalog.get("families", {}).values():
        for skill_id, skill in fdata.get("skills", {}).items():
            tag = skill.get("bt_tag", "Action")
            if tag == "Action":
                actions.append(skill_id)
            elif tag == "Condition":
                conditions.append(skill_id)
    return sorted(set(actions)), sorted(set(conditions))


_ACTIONS, _CONDITIONS = _load_ids()

# Pydantic does not like dynamic Enum names for JSON Schema. Build them at
# import and expose them as regular Enum classes.
ActionID = Enum("ActionID", {a: a for a in _ACTIONS})  # type: ignore[misc]
ConditionID = Enum("ConditionID", {c: c for c in _CONDITIONS})  # type: ignore[misc]


# ── Leaf models ──────────────────────────────────────────────────────────────

class Port(BaseModel):
    """Single BTCPP port — key=value on a node."""
    name: str = Field(..., pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    value: str  # typically "{var}" (bb-ref) or a literal


class Action(BaseModel):
    kind: Literal["Action"] = "Action"
    id: ActionID
    name: Optional[str] = None
    ports: List[Port] = Field(default_factory=list)


class Condition(BaseModel):
    kind: Literal["Condition"] = "Condition"
    id: ConditionID
    name: Optional[str] = None
    ports: List[Port] = Field(default_factory=list)


class SubTree(BaseModel):
    kind: Literal["SubTreePlus"] = "SubTreePlus"
    id: str
    name: Optional[str] = None
    ports: List[Port] = Field(default_factory=list)


# ── Control nodes (recursive) ────────────────────────────────────────────────
# Forward refs: `Node` is defined below after the control models.

class Sequence(BaseModel):
    kind: Literal["Sequence"] = "Sequence"
    name: Optional[str] = None
    children: "List[Node]"


class Fallback(BaseModel):
    kind: Literal["Fallback"] = "Fallback"
    name: Optional[str] = None
    children: "List[Node]"


class ReactiveSequence(BaseModel):
    kind: Literal["ReactiveSequence"] = "ReactiveSequence"
    name: Optional[str] = None
    children: "List[Node]"


class ReactiveFallback(BaseModel):
    kind: Literal["ReactiveFallback"] = "ReactiveFallback"
    name: Optional[str] = None
    children: "List[Node]"


class Parallel(BaseModel):
    kind: Literal["Parallel"] = "Parallel"
    name: Optional[str] = None
    success_count: str = "1"   # stored as string because BT ports are strings
    failure_count: str = "1"
    children: "List[Node]"


class Repeat(BaseModel):
    kind: Literal["Repeat"] = "Repeat"
    name: Optional[str] = None
    num_cycles: str = "-1"     # -1 = infinite (common NAV4RAIL idiom)
    child: "Node"


Node = Annotated[
    Union[
        Action, Condition, SubTree,
        Sequence, Fallback, ReactiveSequence, ReactiveFallback, Parallel, Repeat,
    ],
    Field(discriminator="kind"),
]

# Rebuild forward references
for _cls in (Sequence, Fallback, ReactiveSequence, ReactiveFallback, Parallel, Repeat):
    _cls.model_rebuild()


# ── Top-level BT ─────────────────────────────────────────────────────────────

class BehaviorTreeBlock(BaseModel):
    id: str = Field(..., pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    root: Node


class BehaviorTree(BaseModel):
    """Full BTCPP v4 document: one or more BehaviorTree blocks + a main name."""
    btcpp_format: Literal["4"] = "4"
    main_tree_to_execute: str = Field(..., pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    trees: List[BehaviorTreeBlock] = Field(..., min_length=1)


# ── JSON → XML writer ────────────────────────────────────────────────────────

def _render_ports(ports: List[Port]) -> str:
    return "".join(f' {p.name}="{p.value}"' for p in ports)


def _render_name(name: Optional[str]) -> str:
    return f' name="{name}"' if name else ""


def _render_node(node, indent: int = 2) -> str:
    pad = " " * indent
    kind = node.kind

    if kind in ("Action", "Condition"):
        return f'{pad}<{kind}{_render_name(node.name)} ID="{node.id.value}"{_render_ports(node.ports)}/>'

    if kind == "SubTreePlus":
        return f'{pad}<SubTreePlus{_render_name(node.name)} ID="{node.id}" __autoremap="true"{_render_ports(node.ports)}/>'

    if kind in ("Sequence", "Fallback", "ReactiveSequence", "ReactiveFallback"):
        inner = "\n".join(_render_node(c, indent + 2) for c in node.children)
        return f'{pad}<{kind}{_render_name(node.name)}>\n{inner}\n{pad}</{kind}>'

    if kind == "Parallel":
        inner = "\n".join(_render_node(c, indent + 2) for c in node.children)
        return (
            f'{pad}<Parallel{_render_name(node.name)} '
            f'success_count="{node.success_count}" failure_count="{node.failure_count}">\n'
            f'{inner}\n{pad}</Parallel>'
        )

    if kind == "Repeat":
        inner = _render_node(node.child, indent + 2)
        return (
            f'{pad}<Repeat{_render_name(node.name)} num_cycles="{node.num_cycles}">\n'
            f'{inner}\n{pad}</Repeat>'
        )

    raise ValueError(f"Unknown node kind: {kind!r}")


def to_xml(bt: BehaviorTree) -> str:
    """Render a BehaviorTree Pydantic model as BTCPP v4 XML."""
    trees_xml = []
    for block in bt.trees:
        body = _render_node(block.root, indent=4)
        trees_xml.append(
            f'  <BehaviorTree ID="{block.id}">\n{body}\n  </BehaviorTree>'
        )
    trees_joined = "\n".join(trees_xml)
    return (
        f'<root BTCPP_format="{bt.btcpp_format}" '
        f'main_tree_to_execute="{bt.main_tree_to_execute}">\n'
        f'{trees_joined}\n'
        f'</root>'
    )


def json_to_xml(data: dict | str) -> str:
    """Convenience: parse raw JSON (dict or str) into a BehaviorTree and render XML."""
    import json as _json

    if isinstance(data, str):
        data = _json.loads(data)
    bt = BehaviorTree.model_validate(data)
    return to_xml(bt)


# ── Quick self-test when run directly ────────────────────────────────────────

if __name__ == "__main__":
    # Emit JSON schema size so you can sanity-check Outlines compile cost.
    schema = BehaviorTree.model_json_schema()
    print(f"Schema top-level keys: {list(schema.keys())}")
    print(f"ActionID variants:   {len(_ACTIONS)}")
    print(f"ConditionID variants: {len(_CONDITIONS)}")
    print(f"Schema JSON size:    {len(str(schema))} chars")
