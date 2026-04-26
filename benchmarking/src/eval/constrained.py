"""
constrained.py — Grammar-constrained decoding backends.
========================================================
Two backends, one interface: both return a `ConstraintHandle` carrying a
`LogitsProcessor` that plugs into `model.generate(..., logits_processor=[lp])`.

  - mode="none"     -> handle.logits_processor is None         (baseline)
  - mode="gbnf"     -> transformers-cfg GrammarConstrainedLogitsProcessor
  - mode="outlines" -> outlines.processors.JSONLogitsProcessor (or Regex)

Stateful backends:
  transformers-cfg's `GrammarConstrainedLogitsProcessor` tracks the sequence of
  token ids seen so far and raises
      "Input ID's length is inconsistent with the current state of the
       GrammarConstrainedLogitsProcessor."
  if the same instance is reused across independent `model.generate(...)` calls.
  Solution: `ConstraintHandle.fresh_processor()` rebuilds the processor. The
  benchmark loop calls it before each mission.

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
import time
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
    # Rebuilds a fresh LogitsProcessor — required for stateful backends (GBNF).
    _rebuild_fn: Optional[Callable[[], Any]] = None

    def is_active(self) -> bool:
        return self.mode != "none"

    def fresh_processor(self) -> Optional[Any]:
        """Rebuild a stateless processor if the backend is stateful.

        GBNF's `GrammarConstrainedLogitsProcessor` caches the input-ids prefix;
        calling `model.generate` twice with the same instance raises
        "Input ID's length is inconsistent…". We rebuild per mission.
        Outlines processors are stateless wrt input ids across calls — safe to reuse.
        """
        if self._rebuild_fn is not None:
            self.logits_processor = self._rebuild_fn()
        return self.logits_processor


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


def _patch_gpt2_tokenizer_byte_encoder() -> None:
    """transformers ≥ 4.46 dropped the `byte_encoder` instance attr on GPT2Tokenizer;
    transformers-cfg's GPT2TokenizerMiddleMapping still reads it and crashes with
    `AttributeError: GPT2Tokenizer has no attribute byte_encoder`.

    Important: `from transformers import GPT2Tokenizer` returns a _LazyModule
    proxy — patching it may not propagate to the real class. We import directly
    from the concrete module to guarantee the patch lands on the right object.

    We patch at two levels:
      1. class attribute (covers normal attribute lookup)
      2. __init__ wrap that sets instance attrs via object.__setattr__
         (covers PreTrainedTokenizerBase.__getattr__ overrides)
    """
    try:
        from transformers.models.gpt2.tokenization_gpt2 import (  # type: ignore
            GPT2Tokenizer,
            bytes_to_unicode,
        )
    except ImportError as e:
        logger.warning(
            "GBNF: cannot import transformers.models.gpt2.tokenization_gpt2 (%s) "
            "— byte_encoder patch skipped. GBNF will likely fail for BPE models.",
            e,
        )
        return

    if getattr(GPT2Tokenizer, "_nav4rail_byte_encoder_patched", False):
        logger.debug("GBNF: GPT2Tokenizer byte_encoder already patched")
        return

    encoder = bytes_to_unicode()
    decoder = {v: k for k, v in encoder.items()}

    # Layer 1 — class-level attrs (cheap; works if attribute resolution reaches the class).
    try:
        GPT2Tokenizer.byte_encoder = encoder  # type: ignore[attr-defined]
        GPT2Tokenizer.byte_decoder = decoder  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        logger.warning("GBNF: class-level GPT2Tokenizer patch failed (%s)", e)

    # Layer 2 — wrap __init__ so every new instance carries the attrs in its
    # own __dict__. This beats any __getattr__ override that might shadow class attrs.
    try:
        orig_init = GPT2Tokenizer.__init__

        def _patched_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            try:
                object.__setattr__(self, "byte_encoder", encoder)
                object.__setattr__(self, "byte_decoder", decoder)
            except Exception:  # noqa: BLE001
                pass

        GPT2Tokenizer.__init__ = _patched_init
    except Exception as e:  # noqa: BLE001
        logger.warning("GBNF: GPT2Tokenizer __init__ wrap failed (%s)", e)

    GPT2Tokenizer._nav4rail_byte_encoder_patched = True  # type: ignore[attr-defined]
    logger.info(
        "GBNF: patched GPT2Tokenizer (class attr + __init__ wrap) "
        "for transformers-cfg compat — module id=%s",
        id(GPT2Tokenizer),
    )


def _reload_slow_tokenizer(tokenizer):
    """Reload a *truly slow* (Python, SentencePiece) tokenizer for the same model.

    `AutoTokenizer.from_pretrained(name, use_fast=False)` sometimes silently
    returns a PreTrainedTokenizerFast instance (seen for Mistral v0.2 on recent
    transformers), which transformers-cfg 0.2.7 still refuses with
    `NotImplementedError: Tokenizer not supported: TokenizersBackend`.

    So we try, in order:
      1. AutoTokenizer(use_fast=False) — check `is_fast is False`
      2. LlamaTokenizer (sentencepiece) — works for Mistral v0.2 and any
         llama-family model that ships tokenizer.model
      3. Give up, return original.

    Returns the original tokenizer on any failure (e.g. Llama3 ships only
    tokenizer.json — no slow variant exists).
    """
    name = getattr(tokenizer, "name_or_path", None)
    if not name:
        return tokenizer

    # Attempt 1: AutoTokenizer in slow mode.
    try:
        from transformers import AutoTokenizer  # lazy import

        cand = AutoTokenizer.from_pretrained(name, use_fast=False)
        if getattr(cand, "is_fast", True) is False:
            logger.info(
                "GBNF: reloaded slow tokenizer via AutoTokenizer(use_fast=False) "
                "for %s (%s).", name, type(cand).__name__,
            )
            return cand
        logger.info(
            "GBNF: AutoTokenizer(use_fast=False) still returned a fast tokenizer "
            "for %s — trying LlamaTokenizer fallback.", name,
        )
    except Exception as e:  # noqa: BLE001
        logger.info("GBNF: AutoTokenizer(use_fast=False) raised for %s: %s", name, e)

    # Attempt 2: LlamaTokenizer (sentencepiece). Good for Mistral v0.2.
    try:
        from transformers import LlamaTokenizer  # type: ignore

        cand = LlamaTokenizer.from_pretrained(name)
        logger.info(
            "GBNF: reloaded slow tokenizer via LlamaTokenizer for %s.", name,
        )
        return cand
    except Exception as e:  # noqa: BLE001
        logger.info("GBNF: LlamaTokenizer fallback failed for %s: %s", name, e)

    logger.info(
        "GBNF: no slow tokenizer available for %s; GBNF may be unsupported on this model.",
        name,
    )
    return tokenizer


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

    # Compatibility shim for transformers ≥ 4.46 × transformers-cfg 0.2.7.
    _patch_gpt2_tokenizer_byte_encoder()

    def _try_build(tok):
        grammar = IncrementalGrammarConstraint(grammar_str, "root", tok)
        return GrammarConstrainedLogitsProcessor(grammar)

    # Try the tokenizer we were given first (fast or slow). If transformers-cfg
    # refuses it — seen with Mistral's TokenizersBackend and some byte-BPE edge
    # cases — reload the slow variant and retry. Keeps the happy path fast.
    try:
        initial_processor = _try_build(tokenizer)
        chosen_tok = tokenizer
    except (NotImplementedError, AttributeError, TypeError) as e:
        logger.info(
            "GBNF: primary tokenizer rejected by transformers-cfg (%s). "
            "Retrying with slow tokenizer…", e,
        )
        slow_tok = _reload_slow_tokenizer(tokenizer)
        if slow_tok is tokenizer:
            raise  # nothing to fall back to — surface the original error
        initial_processor = _try_build(slow_tok)
        chosen_tok = slow_tok
        logger.info(
            "GBNF: using slow tokenizer (%s) for grammar constraint.",
            getattr(chosen_tok, "name_or_path", "<unknown>"),
        )

    def _build_fresh_processor():
        # Rebuilt per mission — state is reset each time (see ConstraintHandle.fresh_processor).
        return _try_build(chosen_tok)

    logger.info(
        "GBNF constraint loaded (%d bytes, start=root) from %s",
        grammar_path.stat().st_size,
        grammar_path,
    )
    return ConstraintHandle(
        mode="gbnf",
        logits_processor=initial_processor,
        backend_version=getattr(transformers_cfg, "__version__", "unknown"),
        _rebuild_fn=_build_fresh_processor,
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

    post_process: Optional[Callable[[str], str]] = None

    if regex_str:
        logger.info(
            "Outlines constraint: compiling regex FSM (pattern_len=%d) — may take a while…",
            len(regex_str),
        )
        t0 = time.time()
        processor = processor_cls_regex(regex_str, tokenizer=wrapped_tokenizer)
        logger.info("Outlines regex FSM compiled in %.1fs", time.time() - t0)
        # Regex output is XML-shaped already.

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
        schema_str = (
            json.dumps(schema_json) if not isinstance(schema_json, str) else schema_json
        )
        try:
            vocab_size = len(tokenizer.get_vocab())
        except Exception:  # noqa: BLE001
            vocab_size = getattr(tokenizer, "vocab_size", -1)
        logger.info(
            "Outlines constraint: compiling JSON-schema FSM (%s.%s, schema_bytes=%d, "
            "vocab=%d) — recursive schemas × large vocabs can take many minutes. "
            "If this hangs, switch to outlines_regex in your config.",
            mod_path, cls_name, len(schema_str), vocab_size,
        )
        t0 = time.time()
        processor = processor_cls_json(
            schema_str,
            tokenizer=wrapped_tokenizer,
        )
        logger.info("Outlines JSON FSM compiled in %.1fs", time.time() - t0)

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
    Callers should invoke `handle.fresh_processor()` BEFORE this for stateful
    backends (GBNF) — that rebuilds `handle.logits_processor` in place, then
    this call appends it.
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
