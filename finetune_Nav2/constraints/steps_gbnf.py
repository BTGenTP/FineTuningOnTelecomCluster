from __future__ import annotations

from typing import Any, Mapping

from finetune_Nav2.catalog.catalog_io import all_param_names, allowed_skills


def _gbnf_escape(s: str) -> str:
    # Minimal escaping for double quotes and backslashes inside GBNF string terminals.
    return s.replace("\\", "\\\\").replace('"', '\\"')


def build_steps_json_gbnf(catalog: Mapping[str, Any]) -> str:
    """
    Build a GBNF grammar for a STRICT steps JSON list:
      [
        {"skill":"Wait","params":{"wait_duration":2.0},"comment":"..."},
        ...
      ]

    This grammar focuses on:
    - Valid JSON shape (array of objects)
    - skill enum from the catalog allowlist
    - param keys limited to the UNION of known port names across skills

    NOTE: mapping skill -> allowed param keys is context-sensitive and not handled here.
    We enforce required ports and per-skill ports in post-hoc validation.
    """
    skills = sorted(allowed_skills(catalog).keys())
    if not skills:
        raise ValueError("Catalog has no atomic_skills; cannot build grammar.")

    ports_union = sorted({p for s in all_param_names(catalog).values() for p in s})
    if not ports_union:
        # Still allow empty params objects.
        ports_union = []

    skill_alts = " | ".join([f'"{_gbnf_escape(s)}"' for s in skills])
    if ports_union:
        key_alts = " | ".join([f'"{_gbnf_escape(p)}"' for p in ports_union])
    else:
        key_alts = '"_"'  # unreachable fallback; params-obj will be empty

    return f"""
root ::= ws steps_list ws

steps_list ::= "[" ws (step (ws "," ws step)*)? ws "]"

step ::= "{{" ws skill_kv ws "," ws params_kv (ws "," ws comment_kv)? ws "}}"

skill_kv ::= "\\"skill\\"" ws ":" ws skill_value
skill_value ::= {skill_alts}

params_kv ::= "\\"params\\"" ws ":" ws params_obj

params_obj ::= "{{" ws (param_kv (ws "," ws param_kv)*)? ws "}}"
param_kv ::= param_key ws ":" ws json_value
param_key ::= {key_alts}

comment_kv ::= "\\"comment\\"" ws ":" ws json_string

json_value ::= json_string | json_number | json_bool | json_null

json_string ::= "\\"" json_chars* "\\""
json_chars ::= json_char | escape_seq
json_char ::= [^"\\\\]
escape_seq ::= "\\\\" (["\\\\/bfnrt] | "u" hex hex hex hex)

json_number ::= "-"? int_part frac_part? exp_part?
int_part ::= "0" | [1-9] [0-9]*
frac_part ::= "." [0-9]+
exp_part ::= ("e" | "E") ("+" | "-")? [0-9]+

json_bool ::= "true" | "false"
json_null ::= "null"

hex ::= [0-9a-fA-F]

ws ::= ([ \\t\\n\\r])*
""".strip() + "\n"

