from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .catalog import nav4rail_system_rules, skill_map


@dataclass(frozen=True, slots=True)
class PromptExample:
    mission: str
    xml: str
    metadata: dict[str, Any]


def render_catalog_compact(catalog: Mapping[str, Any]) -> str:
    lines = ["NAV4RAIL catalog (27 skills):"]
    for skill_id, spec in sorted(skill_map(catalog).items()):
        attrs = ", ".join(f"{name}:{kind}" for name, kind in spec.attributes.items())
        lines.append(f"- {skill_id} [{spec.bt_tag}] attrs={attrs}")
    return "\n".join(lines)


def render_system_prompt(catalog: Mapping[str, Any], *, include_schema: bool) -> str:
    lines = [
        "You generate only BehaviorTree.CPP v4 XML.",
        "Output must start with <root> and contain complete BehaviorTree definitions.",
        "Use only NAV4RAIL skills from the catalog.",
        "Never invent IDs, tags, or blackboard ports.",
    ]
    lines.extend(f"- {rule}" for rule in nav4rail_system_rules(catalog))
    if include_schema:
        lines.append("Schema hint:")
        lines.append(render_catalog_compact(catalog))
    return "\n".join(lines)


def render_few_shot_block(examples: Sequence[PromptExample]) -> str:
    if not examples:
        return ""
    chunks: list[str] = []
    for index, example in enumerate(examples, start=1):
        chunks.append(f"Example {index} mission:\n{example.mission}\nExample {index} XML:\n{example.xml.strip()}")
    return "\n\n".join(chunks)


def build_chat_messages(
    *,
    mission: str,
    catalog: Mapping[str, Any],
    mode: str,
    few_shot_examples: Sequence[PromptExample] = (),
    include_schema: bool = False,
) -> list[dict[str, str]]:
    system = render_system_prompt(catalog, include_schema=include_schema or mode == "schema_guided")
    user_parts = [f"Mission:\n{mission.strip()}"]
    shot_block = render_few_shot_block(few_shot_examples)
    if shot_block:
        user_parts.append(shot_block)
    user_parts.append("Return only the XML.")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def format_sft_record(record: Mapping[str, Any], *, system_prompt: str) -> dict[str, Any]:
    mission = str(record["mission"]).strip()
    xml = str(record["xml"]).strip()
    prompt = f"{system_prompt}\n\nMission:\n{mission}\n\nXML:\n"
    completion = xml
    return {
        "prompt": prompt,
        "completion": completion,
        "text": prompt + completion,
        "metadata": dict(record.get("metadata", {})),
    }


def export_jsonl(records: Iterable[Mapping[str, Any]]) -> str:
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
