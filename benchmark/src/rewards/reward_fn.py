from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from ..contracts import ValidationIssue, ValidationReport
from .validator import validate


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    score: float
    pass_l1: bool
    pass_l2: bool
    pass_l3: bool
    errors_l1: int
    errors_l2: int
    errors_l3: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "pass_l1": self.pass_l1,
            "pass_l2": self.pass_l2,
            "pass_l3": self.pass_l3,
            "errors_l1": self.errors_l1,
            "errors_l2": self.errors_l2,
            "errors_l3": self.errors_l3,
        }


def _layer_error_counts(issues: Iterable[ValidationIssue]) -> dict[str, int]:
    counts = {"L1": 0, "L2": 0, "L3": 0}
    for it in issues:
        if it.severity != "error":
            continue
        if it.layer in counts:
            counts[it.layer] += 1
    return counts


def reward_from_report(report: ValidationReport) -> RewardBreakdown:
    counts = _layer_error_counts(report.issues)
    pass_l1 = counts["L1"] == 0
    pass_l2 = counts["L2"] == 0
    pass_l3 = counts["L3"] == 0
    score = 0.0
    score += 0.2 if pass_l1 else 0.0
    score += 0.3 if pass_l2 else 0.0
    score += 0.5 if pass_l3 else 0.0
    return RewardBreakdown(
        score=round(score, 6),
        pass_l1=pass_l1,
        pass_l2=pass_l2,
        pass_l3=pass_l3,
        errors_l1=counts["L1"],
        errors_l2=counts["L2"],
        errors_l3=counts["L3"],
    )


def reward_from_xml(
    *,
    xml_text: str,
    catalog_path: Optional[str] = None,
    xsd_path: Optional[str] = None,
    strict: bool = True,
    constraints_dir: Optional[str] = None,
    external_blackboard: Optional[Iterable[str]] = None,
) -> tuple[float, RewardBreakdown, dict[str, Any]]:
    report = validate(
        xml_text=xml_text,
        catalog_path=catalog_path,
        xsd_path=xsd_path,
        strict=strict,
        constraints_dir=constraints_dir,
        external_blackboard=external_blackboard,
    )
    breakdown = reward_from_report(report)
    return breakdown.score, breakdown, report.to_dict()

