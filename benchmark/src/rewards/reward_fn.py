from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from ..contracts import ValidationIssue, ValidationReport
from .validator import validate


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """total_errors = somme brute par couche ; penalty_errors_* = entrée de la pénalité (masquage palier si actif)."""

    score: float
    pass_l1: bool
    pass_l2: bool
    pass_l3: bool
    errors_l1: int
    errors_l2: int
    errors_l3: int
    total_errors: int
    penalty_errors_l1: int
    penalty_errors_l2: int
    penalty_errors_l3: int
    penalty_applied: float
    base_score: float
    hierarchical_layers: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "pass_l1": self.pass_l1,
            "pass_l2": self.pass_l2,
            "pass_l3": self.pass_l3,
            "errors_l1": self.errors_l1,
            "errors_l2": self.errors_l2,
            "errors_l3": self.errors_l3,
            "total_errors": self.total_errors,
            "penalty_errors_l1": self.penalty_errors_l1,
            "penalty_errors_l2": self.penalty_errors_l2,
            "penalty_errors_l3": self.penalty_errors_l3,
            "penalty_applied": self.penalty_applied,
            "base_score": self.base_score,
            "hierarchical_layers": self.hierarchical_layers,
        }


def _layer_error_counts(issues: Iterable[ValidationIssue]) -> dict[str, int]:
    counts = {"L1": 0, "L2": 0, "L3": 0}
    for it in issues:
        if it.severity != "error":
            continue
        if it.layer in counts:
            counts[it.layer] += 1
    return counts


def _penalty_counts_hierarchical(raw: Mapping[str, int]) -> dict[str, int]:
    """Ne pénalise que la première couche en échec (L1 puis L2 puis L3)."""
    if raw["L1"] > 0:
        return {"L1": raw["L1"], "L2": 0, "L3": 0}
    if raw["L2"] > 0:
        return {"L1": 0, "L2": raw["L2"], "L3": 0}
    return {"L1": 0, "L2": 0, "L3": raw["L3"]}


def reward_from_report(
    report: ValidationReport,
    *,
    per_error_penalty: float = 0.1,
    per_code_penalty: Optional[Mapping[str, float]] = None,
    error_weight_by_layer: Optional[Mapping[str, float]] = None,
    penalty_cap: Optional[float] = 2.0,
    score_floor: Optional[float] = -1.0,
    hierarchical_layers: bool = True,
) -> RewardBreakdown:
    """Calcule score = base_score - pénalité.

    Si ``hierarchical_layers`` est vrai (défaut) :
    - Bonus : +0,2 L1, +0,3 L2 seulement si L1 ok, +0,5 L3 seulement si L1 et L2 ok.
    - Pénalité : uniquement les erreurs de la première couche en échec (L1 sinon L2 sinon L3),
      pour éviter de sur-pénaliser des issues aval souvent redondantes quand un palier casse.

    Le validateur émet toujours toutes les couches sur XML parsable ; ce n'est pas une garantie
    logique L1 ⇒ pas d'erreurs L2, d'où ce masquage optionnel côté récompense.
    """
    counts = _layer_error_counts(report.issues)
    pass_l1 = counts["L1"] == 0
    pass_l2 = counts["L2"] == 0
    pass_l3 = counts["L3"] == 0

    if hierarchical_layers:
        base_score = 0.0
        base_score += 0.2 if pass_l1 else 0.0
        base_score += 0.3 if pass_l1 and pass_l2 else 0.0
        base_score += 0.5 if pass_l1 and pass_l2 and pass_l3 else 0.0
        first_layer = "L1" if counts["L1"] > 0 else ("L2" if counts["L2"] > 0 else "L3")
    else:
        base_score = 0.0
        base_score += 0.2 if pass_l1 else 0.0
        base_score += 0.3 if pass_l2 else 0.0
        base_score += 0.5 if pass_l3 else 0.0
        first_layer = None

    total_errors = counts["L1"] + counts["L2"] + counts["L3"]
    layer_w = {layer: float((error_weight_by_layer or {}).get(layer, 1.0)) for layer in ("L1", "L2", "L3")}
    error_issues = [it for it in report.issues if it.severity == "error" and it.layer in {"L1", "L2", "L3"}]
    if hierarchical_layers and first_layer is not None:
        selected = [it for it in error_issues if it.layer == first_layer]
    else:
        selected = error_issues
    penalty_src = {"L1": 0, "L2": 0, "L3": 0}
    penalty_raw = 0.0
    for it in selected:
        penalty_src[it.layer] += 1
        per = float((per_code_penalty or {}).get(it.code, per_error_penalty))
        penalty_raw += layer_w[it.layer] * per
    if penalty_cap is not None:
        penalty_applied = min(penalty_raw, float(penalty_cap))
    else:
        penalty_applied = penalty_raw

    score = base_score - penalty_applied
    if score_floor is not None:
        score = max(float(score_floor), score)

    return RewardBreakdown(
        score=round(score, 6),
        pass_l1=pass_l1,
        pass_l2=pass_l2,
        pass_l3=pass_l3,
        errors_l1=counts["L1"],
        errors_l2=counts["L2"],
        errors_l3=counts["L3"],
        total_errors=total_errors,
        penalty_errors_l1=penalty_src["L1"],
        penalty_errors_l2=penalty_src["L2"],
        penalty_errors_l3=penalty_src["L3"],
        penalty_applied=round(penalty_applied, 6),
        base_score=round(base_score, 6),
        hierarchical_layers=hierarchical_layers,
    )


def reward_from_xml(
    *,
    xml_text: str,
    catalog_path: Optional[str] = None,
    xsd_path: Optional[str] = None,
    strict: bool = True,
    constraints_dir: Optional[str] = None,
    external_blackboard: Optional[Iterable[str]] = None,
    per_error_penalty: float = 0.1,
    per_code_penalty: Optional[Mapping[str, float]] = None,
    error_weight_by_layer: Optional[Mapping[str, float]] = None,
    penalty_cap: Optional[float] = 2.0,
    score_floor: Optional[float] = -1.0,
    hierarchical_layers: bool = True,
) -> tuple[float, RewardBreakdown, dict[str, Any]]:
    report = validate(
        xml_text=xml_text,
        catalog_path=catalog_path,
        xsd_path=xsd_path,
        strict=strict,
        constraints_dir=constraints_dir,
        external_blackboard=external_blackboard,
    )
    breakdown = reward_from_report(
        report,
        per_error_penalty=per_error_penalty,
        per_code_penalty=per_code_penalty,
        error_weight_by_layer=error_weight_by_layer,
        penalty_cap=penalty_cap,
        score_floor=score_floor,
        hierarchical_layers=hierarchical_layers,
    )
    return breakdown.score, breakdown, report.to_dict()
