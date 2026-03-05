from __future__ import annotations

from typing import Any, Callable, Mapping

from finetune_Nav2.constraints.steps_jsonschema import build_steps_jsonschema


def build_prefix_allowed_tokens_fn(tokenizer, catalog: Mapping[str, Any]) -> Callable[[int, Any], list[int]]:
    """
    HuggingFace constrained decoding via lm-format-enforcer.

    This constrains generation to match a JSON Schema derived from the Nav2 catalog:
    - output must be valid JSON list of steps
    - skill must be in allowlist
    - params keys limited to union of known ports

    Requires: lm-format-enforcer
    """
    try:
        from lmformatenforcer.integrations.transformers import (  # type: ignore
            build_transformers_prefix_allowed_tokens_fn,
        )
        # lm-format-enforcer exposes JsonSchemaParser in newer versions.
        from lmformatenforcer import JsonSchemaParser  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "lm-format-enforcer not installed. Install it to use constrained decoding.\n"
            "Example: pip install lm-format-enforcer"
        ) from exc

    schema = build_steps_jsonschema(catalog)
    parser = JsonSchemaParser(schema)
    return build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

