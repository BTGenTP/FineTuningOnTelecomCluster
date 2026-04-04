from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ..contracts import PromptConfig
from ..data.formatting import PromptExample, build_chat_messages


def render_prompt_bundle(
    *,
    mission: str,
    catalog: Mapping[str, Any],
    prompt_config: PromptConfig,
    few_shot_examples: Sequence[PromptExample] = (),
    xsd_path: Optional[str] = None,
) -> dict[str, Any]:
    selected_examples = list(few_shot_examples[: prompt_config.few_shot_k]) if prompt_config.few_shot_k else []
    messages = build_chat_messages(
        mission=mission,
        catalog=catalog,
        mode=prompt_config.mode,
        few_shot_examples=selected_examples,
        include_schema=prompt_config.include_schema,
        xsd_path=xsd_path,
        include_xsd=prompt_config.include_xsd,
        xsd_max_chars=prompt_config.xsd_max_chars,
    )
    return {
        "mode": prompt_config.mode,
        "messages": messages,
        "metadata": {
            "few_shot_k": len(selected_examples),
            "include_schema": prompt_config.include_schema,
        },
    }
