"""
build_grammar.py — Generate bt_grammar.gbnf from skills_catalog.yaml.
======================================================================
The GBNF grammar enforces BehaviorTree.CPP v4 XML syntax and constrains
Action / Condition IDs to skills that actually exist in the catalog.

The grammar is recomputed from `data/skills_catalog.yaml` so that adding a
skill to the catalog propagates automatically: no hard-coded lists anywhere.

GBNF syntax used here (llama.cpp-compatible, consumed by transformers-cfg):
  rule ::= alt1 | alt2                # alternation
  "literal"                            # terminal string
  [char-class]                         # regex-like character class
  rule*  rule+  rule?                  # repetition / optional
  (group)                              # grouping

Usage:
  python -m src.eval.build_grammar
  python -m src.eval.build_grammar --catalog data/skills_catalog.yaml --out src/eval/bt_grammar.gbnf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


GRAMMAR_HEADER = r"""# ============================================================================
# NAV4RAIL BehaviorTree.CPP v4 grammar (auto-generated)
# ============================================================================
# Regenerate with:  python -m src.eval.build_grammar
# Source of truth:  data/skills_catalog.yaml
# Consumer:         transformers-cfg (llama.cpp-style GBNF)
# ----------------------------------------------------------------------------

root ::= document

document ::= xml-header ws btree-block (ws btree-block)* ws close-root

xml-header ::= "<root BTCPP_format=\"4\" main_tree_to_execute=\"" btree-name "\">"
close-root ::= "</root>"

btree-block ::= "<BehaviorTree ID=\"" btree-name "\">" ws node ws "</BehaviorTree>"
btree-name ::= ident

# ----------------------------------------------------------------------------
# Behavior Tree nodes
# ----------------------------------------------------------------------------

node ::= control-node | action-node | condition-node | subtree-node

# ---- Control nodes ---------------------------------------------------------
control-node ::= sequence | fallback | reactive-fallback | reactive-sequence | parallel | repeat

sequence          ::= "<Sequence"         name-attr? ">" ws (node ws)+ "</Sequence>"
fallback          ::= "<Fallback"         name-attr? ">" ws (node ws)+ "</Fallback>"
reactive-sequence ::= "<ReactiveSequence" name-attr? ">" ws (node ws)+ "</ReactiveSequence>"
reactive-fallback ::= "<ReactiveFallback" name-attr? ">" ws (node ws)+ "</ReactiveFallback>"
parallel          ::= "<Parallel" name-attr? " success_count=\"" int-or-bb "\" failure_count=\"" int-or-bb "\">" ws (node ws)+ "</Parallel>"
repeat            ::= "<Repeat"   name-attr? " num_cycles=\""   int-or-bb "\">"                                            ws node     ws "</Repeat>"

# ---- Leaf nodes ------------------------------------------------------------
action-node    ::= "<Action"    name-attr? " ID=\"" action-id    "\"" port-attr* "/>"
condition-node ::= "<Condition" name-attr? " ID=\"" condition-id "\"" port-attr* "/>"

# ---- SubTree (BTCPP v4 uses SubTreePlus with __autoremap) ------------------
subtree-node ::= "<SubTreePlus" name-attr? " ID=\"" ident "\" __autoremap=\"true\"" port-attr* "/>"

# ----------------------------------------------------------------------------
# Attributes
# ----------------------------------------------------------------------------
name-attr ::= " name=\"" name-value "\""
# Hyphen first: transformers_cfg rejects `\-` inside character classes.
name-value ::= [-A-Za-z0-9_ ]+

port-attr ::= " " port-key "=\"" port-value "\""
port-key   ::= [a-zA-Z_] [a-zA-Z0-9_]*
port-value ::= bb-ref | literal-value

# Blackboard reference — the overwhelming norm in NAV4RAIL ({var_name}).
bb-ref ::= "{" [a-zA-Z_] [a-zA-Z0-9_]* "}"

# Literal values: int, float, bool, simple identifier, or quoted string-ish.
literal-value ::= int-lit | float-lit | bool-lit | ident
int-lit   ::= "-"? [0-9]+
float-lit ::= "-"? [0-9]+ "." [0-9]+
bool-lit  ::= "true" | "false"

int-or-bb ::= int-lit | bb-ref

ident ::= [a-zA-Z_] [a-zA-Z0-9_]*

# Whitespace — BT XML is typically pretty-printed but a constrained decoder
# must tolerate both flat and indented output. Allow any amount of newlines,
# spaces, or tabs between nodes.
ws ::= [ \t\n\r]*
"""


def _collect_skills(catalog: dict) -> tuple[list[str], list[str]]:
    """Walk the catalog and split skills by bt_tag."""
    actions: list[str] = []
    conditions: list[str] = []
    for _family, fdata in catalog.get("families", {}).items():
        for skill_id, skill in fdata.get("skills", {}).items():
            tag = skill.get("bt_tag", "Action")
            if tag == "Action":
                actions.append(skill_id)
            elif tag == "Condition":
                conditions.append(skill_id)
    return sorted(set(actions)), sorted(set(conditions))


def _alt(items: list[str]) -> str:
    """Render a GBNF alternation of literal strings."""
    return " | ".join(f'"{s}"' for s in items)


def build(catalog_path: Path, out_path: Path) -> None:
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    actions, conditions = _collect_skills(catalog)
    if not actions:
        print("ERROR: no Action skills found in catalog", file=sys.stderr)
        sys.exit(1)

    footer = (
        "# ----------------------------------------------------------------------------\n"
        "# Skill IDs (auto-filled from skills_catalog.yaml)\n"
        "# ----------------------------------------------------------------------------\n"
        f"action-id ::= {_alt(actions)}\n\n"
        f"condition-id ::= {_alt(conditions) if conditions else '[a-zA-Z_]+'}\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(GRAMMAR_HEADER + footer, encoding="utf-8")
    print(
        f"Wrote {out_path} "
        f"({len(actions)} Action, {len(conditions)} Condition, "
        f"{out_path.stat().st_size} bytes)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NAV4RAIL BTCPP v4 GBNF grammar.")
    parser.add_argument(
        "--catalog",
        default="data/skills_catalog.yaml",
        help="Path to skills_catalog.yaml (default: data/skills_catalog.yaml).",
    )
    parser.add_argument(
        "--out",
        default="src/eval/bt_grammar.gbnf",
        help="Output path for the generated GBNF (default: src/eval/bt_grammar.gbnf).",
    )
    args = parser.parse_args()

    catalog_path = Path(args.catalog).resolve()
    out_path = Path(args.out).resolve()

    if not catalog_path.is_file():
        print(f"ERROR: catalog not found: {catalog_path}", file=sys.stderr)
        sys.exit(2)

    build(catalog_path, out_path)


if __name__ == "__main__":
    main()
