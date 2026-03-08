from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Mapping, Set, Tuple
from xml.etree import ElementTree as ET

from finetune_Nav2_XML.catalog.catalog_io import allowed_skills


def _scan_reference_tags(reference_dir: Path) -> Set[str]:
    tags: Set[str] = set()
    if not reference_dir.exists():
        return tags
    for p in sorted(reference_dir.rglob("*.xml")):
        try:
            root = ET.parse(str(p)).getroot()
        except Exception:
            continue
        for el in root.iter():
            tags.add(el.tag)
    return tags


def _build_allowed_tags(catalog: Mapping[str, Any], reference_dir: Path | None) -> Set[str]:
    bt_tags = {str(it.get("bt_tag")) for it in allowed_skills(catalog).values() if it.get("bt_tag")}
    ctrl_tags = {
        # Catalog control nodes (superset is fine; validator remains source of truth).
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
        "DistanceController",
        "SpeedController",
    }
    base = {"root", "BehaviorTree", "SubTree"}
    ref = _scan_reference_tags(reference_dir) if reference_dir else set()
    return set(bt_tags) | ctrl_tags | base | ref


def _build_xml_regex(*, allowed_tags: Set[str]) -> str:
    """
    Build a best-effort regex for XML direct generation.

    Goals:
    - enforce that every XML tag name belongs to allowed_tags
    - enforce document starts with <root ...> and includes a MainTree definition
    - enforce it ends with </root>

    Limits:
    - cannot guarantee balanced nesting (validator does that)
    - attributes are not fully validated here
    """
    # Long alternation can be expensive; keep it deterministic.
    tags_alt = "|".join(sorted(re.escape(t) for t in allowed_tags if t and t.strip()))
    if not tags_alt:
        raise ValueError("No allowed tags for regex constraint.")

    # A single allowed XML element or comment. We accept any attributes except '<' and '"'.
    tag = rf"(?:{tags_alt})"
    open_or_self = rf"<{tag}(?:\s+[^<>\"=]+\s*=\s*\"[^\"<>]*\")*\s*/?>"
    close = rf"</{tag}\s*>"
    comment = r"<!--[\s\S]*?-->"

    # Any sequence of allowed tags/comments/whitespace.
    body = rf"(?:\s*(?:{comment}|{open_or_self}|{close})\s*)+"

    # Strong anchors for root + MainTree presence (best-effort).
    # - root attribute main_tree_to_execute="MainTree" must appear in the opening tag.
    # - a BehaviorTree with ID="MainTree" must appear somewhere.
    start = r"^\s*<root\b[^<>]*\bmain_tree_to_execute\s*=\s*\"MainTree\"[^<>]*>"
    must_have_main = r"(?=[\s\S]*<BehaviorTree\b[^<>]*\bID\s*=\s*\"MainTree\"[^<>]*>)"
    end = r"[\s\S]*</root>\s*$"

    return start + must_have_main + body + end


def build_prefix_allowed_tokens_fn(
    tokenizer,
    catalog: Mapping[str, Any],
    *,
    reference_dir: str | Path | None = None,
) -> Tuple[Callable[[int, Any], list[int]], str]:
    """
    HuggingFace constrained decoding via lm-format-enforcer (RegexParser).

    Returns (prefix_allowed_tokens_fn, debug_regex).
    """
    try:
        from lmformatenforcer import RegexParser  # type: ignore
        from lmformatenforcer.integrations.transformers import (  # type: ignore
            build_transformers_prefix_allowed_tokens_fn,
        )
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "lm-format-enforcer not installed. Install it to use constrained decoding.\n"
            "Example: pip install lm-format-enforcer"
        ) from exc

    ref = Path(reference_dir).expanduser().resolve() if reference_dir is not None else None
    allowed = _build_allowed_tags(catalog, ref)
    regex = _build_xml_regex(allowed_tags=allowed)

    # lm-format-enforcer expects a pattern; it will enforce token-by-token compliance.
    parser = RegexParser(regex, flags=re.MULTILINE)
    fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    return fn, regex

