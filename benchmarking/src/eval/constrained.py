"""
constrained.py — Grammar-constrained decoding backends.
========================================================
Two backends, one interface: both return a `ConstraintHandle` carrying a
`LogitsProcessor` that plugs into `model.generate(..., logits_processor=[lp])`.

  - mode="none"     -> handle.logits_processor is None         (baseline)
  - mode="gbnf"     -> transformers-cfg GrammarConstrainedLogitsProcessor
  - mode="outlines" -> outlines.processors.JSONLogitsProcessor (or Regex)

Post-processing:
  GBNF emits XML directly — no post-processing.
  Outlines emits JSON — `handle.post_process(raw)` converts to XML via bt_schema.

Import is lazy: if a backend package is missing, the failure happens when that
mode is requested, not at module import. This lets zero-shot baselines run on
machines where transformers-cfg / outlines are not installed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


# ── Public handle ────────────────────────────────────────────────────────────

@dataclass
class ConstraintHandle:
    """Everything `_generate_xml` needs to apply a decoding-time constraint."""
    mode: str
    logits_processor: Optional[Any] = None
    backend_version: Optional[str] = None
    post_process: Optional[Callable[[str], str]] = None   # raw_gen_str -> final_xml_str
    # Extra kwargs to inject into model.generate (e.g. suppress EOS until root closes).
    extra_generate_kwargs: dict = field(default_factory=dict)

    def is_active(self) -> bool:
        return self.mode != "none"


def build_constraint(
    mode: str,
    tokenizer,
    cfg: dict,
    catalog=None,
) -> ConstraintHandle:
    """Factory. See module docstring for modes."""
    mode = (mode or "none").lower().strip()
    if mode == "none":
        return ConstraintHandle(mode="none")

    constraint_cfg = (cfg or {}).get("eval", {}).get("constraint", {}) or {}

    if mode == "gbnf":
        return _build_gbnf(tokenizer, constraint_cfg)
    if mode == "outlines":
        return _build_outlines(tokenizer, constraint_cfg)

    raise ValueError(
        f"Unknown constraint mode: {mode!r}. Use one of: none | gbnf | outlines."
    )


# ── GBNF backend (transformers-cfg) ──────────────────────────────────────────

def _resolve_path(p: str | Path) -> Path:
    """Make a path absolute, relative to benchmarking/ root if needed."""
    p = Path(p)
    if p.is_absolute():
        return p
    bench_root = Path(__file__).parent.parent.parent
    return (bench_root / p).resolve()


def _build_gbnf(tokenizer, constraint_cfg: dict) -> ConstraintHandle:
    try:
        import transformers_cfg
        from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
        from transformers_cfg.generation.logits_process import (
            GrammarConstrainedLogitsProcessor,
        )
    except ImportError as e:
        raise RuntimeError(
            "GBNF mode requires transformers-cfg. Install with:\n"
            "  pip install 'transformers-cfg>=0.2.5'\n"
            f"Original error: {e}"
        ) from e

    grammar_path = _resolve_path(
        constraint_cfg.get("gbnf_path", "src/eval/bt_grammar.gbnf")
    )
    if not grammar_path.is_file():
        raise FileNotFoundError(
            f"GBNF grammar file not found: {grammar_path}\n"
            "Generate it with:\n"
            "  python -m src.eval.build_grammar"
        )

    grammar_str = grammar_path.read_text(encoding="utf-8")
    # transformers-cfg ≥ 0.2.5 exposes IncrementalGrammarConstraint(grammar_str, start_rule, tokenizer).
    # "root" must match the first non-terminal in bt_grammar.gbnf.
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    processor = GrammarConstrainedLogitsProcessor(grammar)
    logger.info(
        "GBNF constraint loaded (%d bytes, start=root) from %s",
        grammar_path.stat().st_size,
        grammar_path,
    )
    return ConstraintHandle(
        mode="gbnf",
        logits_processor=processor,
        backend_version=getattr(transformers_cfg, "__version__", "unknown"),
    )


# ── Outlines backend ─────────────────────────────────────────────────────────

def _build_outlines(tokenizer, constraint_cfg: dict) -> ConstraintHandle:
    try:
        import outlines
    except ImportError as e:
        raise RuntimeError(
            "Outlines mode requires `outlines`. Install with:\n"
            "  pip install 'outlines>=0.0.46'\n"
            f"Original error: {e}"
        ) from e

    # Import the processor module. outlines renamed this path in 0.1.x.
    processor_cls_json = None
    processor_cls_regex = None
    try:
        from outlines.processors import JSONLogitsProcessor as _J
        from outlines.processors import RegexLogitsProcessor as _R
        processor_cls_json, processor_cls_regex = _J, _R
    except ImportError:
        try:
            from outlines.fsm.logits_processors import JSONLogitsProcessor as _J  # type: ignore
            from outlines.fsm.logits_processors import RegexLogitsProcessor as _R  # type: ignore
            processor_cls_json, processor_cls_regex = _J, _R
        except ImportError as e:
            raise RuntimeError(
                "Could not find Outlines LogitsProcessor — incompatible outlines version.\n"
                "Pin to a known-good range: pip install 'outlines>=0.0.46,<0.2'\n"
                f"Original error: {e}"
            ) from e

    regex_str = constraint_cfg.get("outlines_regex")
    schema_spec = constraint_cfg.get("outlines_schema", "src.eval.bt_schema:BehaviorTree")

    # Outlines expects its own tokenizer wrapper on most versions. Try to wrap;
    # fall back to the raw HF tokenizer if the wrapper moved.
    try:
        from outlines.models.transformers import TransformerTokenizer  # type: ignore
        wrapped_tokenizer = TransformerTokenizer(tokenizer)
    except ImportError:
        wrapped_tokenizer = tokenizer

    if regex_str:
        logger.info("Outlines constraint: regex (%d chars)", len(regex_str))
        processor = processor_cls_regex(regex_str, tokenizer=wrapped_tokenizer)
        post_process: Optional[Callable[[str], str]] = None  # regex output is already XML-shaped

    else:
        # JSON-schema mode — import the Pydantic class and build its schema.
        mod_path, _, cls_name = schema_spec.partition(":")
        if not cls_name:
            raise ValueError(
                f"outlines_schema must be 'module.path:ClassName', got: {schema_spec!r}"
            )
        import importlib
        mod = importlib.import_module(mod_path)
        schema_cls = getattr(mod, cls_name)
        schema_json = (
            schema_cls.model_json_schema()
            if hasattr(schema_cls, "model_json_schema")
            else schema_cls.schema()
        )
        logger.info(
            "Outlines constraint: JSON schema (%s.%s, %d keys)",
            mod_path, cls_name, len(schema_json),
        )
        processor = processor_cls_json(
            json.dumps(schema_json) if not isinstance(schema_json, str) else schema_json,
            tokenizer=wrapped_tokenizer,
        )

        # Post-process: JSON -> XML via the same Pydantic model.
        from src.eval.bt_schema import json_to_xml

        def post_process(raw: str) -> str:
            """Convert the model's JSON output to BTCPP v4 XML."""
            try:
                return json_to_xml(raw)
            except Exception as e:  # noqa: BLE001
                logger.warning("json_to_xml failed: %s — returning raw", e)
                return raw

    return ConstraintHandle(
        mode="outlines",
        logits_processor=processor,
        backend_version=getattr(outlines, "__version__", "unknown"),
        post_process=post_process,
    )


# ── Helper for benchmark.py: inject handle into generate kwargs ──────────────

def apply_to_generate_kwargs(
    gen_kwargs: dict,
    handle: ConstraintHandle,
) -> dict:
    """Merge a ConstraintHandle's processor + extras into a model.generate kwargs dict.

    Mutates and returns the dict (for convenience at call sites).
    """
    if handle is None or not handle.is_active():
        return gen_kwargs

    if handle.logits_processor is not None:
        existing: List = list(gen_kwargs.get("logits_processor") or [])
        existing.append(handle.logits_processor)
        gen_kwargs["logits_processor"] = existing

    for k, v in (handle.extra_generate_kwargs or {}).items():
        gen_kwargs.setdefault(k, v)

    return gen_kwargs
