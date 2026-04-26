"""
Rich / text feedback decomposition for NAV4RAIL alignment training.
====================================================================

Wraps `validate_bt` to produce a structured feedback object containing:

  - dense_score    : weighted scalar in [-1, 1] (the existing reward)
  - components     : dict of L1-L5 sub-scores (parse, structure, semantic,
                     coherence, hallucination)
  - errors_text    : human-readable validator errors (multi-line string)
  - warnings_text  : human-readable validator warnings (multi-line string)
  - skills_used    : sorted list of skill IDs found in the BT
  - hallucinated   : sorted list of skills NOT in the catalogue

Used **at training time only** by SDPO/multi-pair DPO/KTO-weighted to provide
the model with NL feedback on its own generations. Never injected into
inference prompts (per Q clarification 2026-04-27).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.data.skills_loader import SkillsCatalog
from src.eval.validate_bt import enrich_ports, validate_bt


_ROOT_RE = re.compile(r"<root[\s\S]*?</root\s*>", re.IGNORECASE)
_SKILL_ID_RE = re.compile(r'<(?:Action|Condition)[^>]*\bID="([^"]+)"')


@dataclass
class RichFeedback:
    dense_score: float          # final weighted reward in [-1, 1]
    raw_score: float            # validate_bt score in [0, 1]
    valid: bool
    components: dict[str, float]  # parse, structure, semantic, coherence, hallucination
    errors_text: str
    warnings_text: str
    skills_used: list[str] = field(default_factory=list)
    hallucinated: list[str] = field(default_factory=list)

    def to_prompt_text(self, max_lines: int = 6) -> str:
        """Render feedback as compact NL for use in a refinement prompt
        (training-side only — distillation target, never inference input)."""
        parts = [f"Score: {self.raw_score:.2f} (valid={self.valid})"]
        if self.errors_text:
            parts.append(
                "Erreurs validateur:\n"
                + "\n".join(self.errors_text.splitlines()[:max_lines])
            )
        if self.warnings_text:
            parts.append(
                "Avertissements:\n"
                + "\n".join(self.warnings_text.splitlines()[:max_lines])
            )
        if self.hallucinated:
            parts.append("Skills hors catalogue: " + ", ".join(self.hallucinated[:8]))
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_xml(text: str) -> str | None:
    m = _ROOT_RE.search(text)
    return m.group(0) if m else None


def compute_rich_feedback(
    completion: str,
    mission: str,
    catalog: SkillsCatalog,
    weights: dict[str, float] | None = None,
) -> RichFeedback:
    """Score one completion and return the full feedback object.

    Default weights come from PLAN.md §1.3:
      parse 0.3, structure 0.2, semantic 0.2, coherence 0.2, hallucination 0.1
    """
    weights = weights or {
        "parse": 0.3,
        "structure": 0.2,
        "semantic": 0.2,
        "coherence": 0.2,
        "hallucination": 0.1,
    }

    xml = _extract_xml(completion)
    if xml is None:
        return RichFeedback(
            dense_score=-1.0,
            raw_score=0.0,
            valid=False,
            components={"parse": -1.0, "structure": -0.5, "semantic": 0.0,
                        "coherence": 0.0, "hallucination": 0.0},
            errors_text="No <root> XML block found in completion",
            warnings_text="",
            skills_used=[],
            hallucinated=[],
        )

    try:
        xml_enriched = enrich_ports(xml, catalog)
    except Exception:  # noqa: BLE001
        xml_enriched = xml

    try:
        result = validate_bt(xml_enriched, catalog)
    except Exception as exc:  # noqa: BLE001
        return RichFeedback(
            dense_score=-0.8,
            raw_score=0.0,
            valid=False,
            components={"parse": -0.8, "structure": -0.5, "semantic": 0.0,
                        "coherence": 0.0, "hallucination": 0.0},
            errors_text=f"Validator crash: {exc}",
            warnings_text="",
            skills_used=[],
            hallucinated=[],
        )

    skills_used = sorted(set(_SKILL_ID_RE.findall(xml_enriched)))
    valid_skills = catalog.valid_skills()
    hallucinated = sorted(s for s in skills_used if s not in valid_skills)

    # ── Component scores ────────────────────────────────────────────────────
    raw_score = float(result.score)
    n_warn = len(result.warnings)
    n_err = len(result.errors)

    c_parse = 1.0  # XML parsed OK if we got here (validate_bt didn't crash)
    c_structure = raw_score if result.valid else -0.5
    c_semantic = max(-1.0, 1.0 - 0.1 * n_warn)
    # Coherence: reuse existing heuristic from reward_fn for transport vs inspection
    c_coherence = _mission_coherence(xml_enriched, mission)
    c_hallucination = max(-1.0, 1.0 - 0.5 * len(hallucinated))

    components = {
        "parse": c_parse,
        "structure": c_structure,
        "semantic": c_semantic,
        "coherence": c_coherence,
        "hallucination": c_hallucination,
    }

    dense = sum(weights[k] * components[k] for k in components)
    dense = max(-1.0, min(1.0, dense))

    return RichFeedback(
        dense_score=dense,
        raw_score=raw_score,
        valid=result.valid,
        components=components,
        errors_text="\n".join(result.errors),
        warnings_text="\n".join(result.warnings),
        skills_used=skills_used,
        hallucinated=hallucinated,
    )


def _mission_coherence(xml: str, mission: str) -> float:
    """Lightweight inline check (mirrors reward_fn._check_mission_coherence)."""
    ml = mission.lower()
    has_manage = "ManageMeasurements" in xml
    has_analyse = "AnalyseMeasurements" in xml
    is_inspection = any(w in ml for w in ("inspection", "inspecter", "mesure", "controle"))
    has_control = any(w in ml for w in ("verifier", "controler", "analyser", "qualite", "corriger"))

    if not is_inspection:
        return 1.0 if not has_manage else 0.4
    if has_control:
        return 1.0 if (has_manage and has_analyse) else (0.6 if has_manage else 0.2)
    return 1.0 if (has_manage and not has_analyse) else (0.7 if has_manage else 0.3)
