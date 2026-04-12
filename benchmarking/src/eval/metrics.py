"""
Metrics computation for NAV4RAIL benchmarking.
===============================================
Computes structural, content, and performance metrics for generated BTs.

Usage:
    from src.eval.metrics import compute_all_metrics
    metrics = compute_all_metrics(xml_str, reference_xml, latency_s, catalog)
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

from src.data.skills_loader import SkillsCatalog
from src.eval.validate_bt import ValidationResult, validate_bt


# ── Structural Metrics ───────────────────────────────────────────────────────


def _max_depth(elem: ET.Element, d: int = 0) -> int:
    children = list(elem)
    if not children:
        return d
    return max(_max_depth(c, d + 1) for c in children)


def compute_structural_metrics(xml_str: str) -> dict[str, Any]:
    """
    Compute tree structure metrics: depth, node counts, branching factor.
    Returns defaults on parse failure.
    """
    defaults = {
        "max_depth": 0,
        "total_nodes": 0,
        "n_behavior_trees": 0,
        "n_subtreeplus": 0,
        "n_sequence": 0,
        "n_fallback": 0,
        "n_reactive_fallback": 0,
        "n_repeat": 0,
        "n_skills": 0,
        "mean_branching_factor": 0.0,
    }
    xml_str = xml_str.strip()
    if xml_str.startswith("<?xml"):
        xml_str = xml_str[xml_str.index("?>") + 2 :].strip()
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return defaults

    total = sum(1 for _ in root.iter())
    n_bt = sum(1 for e in root.iter("BehaviorTree"))
    n_stp = sum(1 for e in root.iter("SubTreePlus"))
    n_seq = sum(1 for e in root.iter("Sequence"))
    n_fb = sum(1 for e in root.iter("Fallback"))
    n_rfb = sum(1 for e in root.iter("ReactiveFallback"))
    n_rep = sum(1 for e in root.iter("Repeat"))

    control_tags = {"Sequence", "Fallback", "ReactiveFallback", "Parallel", "Repeat"}
    control_children = []
    for e in root.iter():
        if e.tag in control_tags:
            control_children.append(len(list(e)))
    mean_bf = sum(control_children) / len(control_children) if control_children else 0.0

    # Count skill (leaf) nodes — Action, Condition, or skill-as-tag
    n_skills = sum(
        1
        for e in root.iter()
        if e.tag in ("Action", "Condition") or (e.tag[0].isupper() and e.tag not in control_tags and e.tag not in {"root", "BehaviorTree", "SubTreePlus"})
    )

    return {
        "max_depth": _max_depth(root),
        "total_nodes": total,
        "n_behavior_trees": n_bt,
        "n_subtreeplus": n_stp,
        "n_sequence": n_seq,
        "n_fallback": n_fb,
        "n_reactive_fallback": n_rfb,
        "n_repeat": n_rep,
        "n_skills": n_skills,
        "mean_branching_factor": round(mean_bf, 2),
    }


# ── Content Metrics ──────────────────────────────────────────────────────────


def compute_hallucination_metrics(
    xml_str: str, catalog: SkillsCatalog
) -> dict[str, Any]:
    """
    Count hallucinated skills (not in catalog).
    Returns {"hallucinated_skills": [...], "hallucination_count": N, "total_skills": N}.
    """
    valid_skills = catalog.valid_skills()
    xml_str = xml_str.strip()
    if xml_str.startswith("<?xml"):
        xml_str = xml_str[xml_str.index("?>") + 2 :].strip()
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return {"hallucinated_skills": [], "hallucination_count": 0, "total_skills": 0}

    hallucinated: list[str] = []
    total = 0
    control_tags = {"root", "BehaviorTree", "SubTreePlus", "Sequence", "Fallback",
                    "ReactiveFallback", "Parallel", "Repeat"}
    for e in root.iter():
        if e.tag in ("Action", "Condition"):
            skill_id = e.get("ID", "")
            total += 1
            if skill_id and skill_id not in valid_skills:
                hallucinated.append(skill_id)
        elif e.tag not in control_tags and e.tag[0].isupper():
            total += 1
            if e.tag not in valid_skills:
                hallucinated.append(e.tag)

    return {
        "hallucinated_skills": sorted(set(hallucinated)),
        "hallucination_count": len(hallucinated),
        "total_skills": total,
    }


def compute_tree_edit_distance(xml_str: str, reference_xml: str | None) -> float | None:
    """
    Compute Zhang-Shasha Tree Edit Distance between generated and reference BT.
    Returns None if reference is unavailable or parsing fails.
    Requires: pip install zss
    """
    if reference_xml is None:
        return None

    try:
        import zss
    except ImportError:
        return None

    def _parse_tree(xml: str):
        xml = xml.strip()
        if xml.startswith("<?xml"):
            xml = xml[xml.index("?>") + 2 :].strip()
        return ET.fromstring(xml)

    def _to_zss(elem: ET.Element) -> zss.Node:
        label = elem.tag
        if elem.tag in ("Action", "Condition"):
            label = elem.get("ID", elem.tag)
        node = zss.Node(label)
        for child in elem:
            node.addkid(_to_zss(child))
        return node

    try:
        gen_root = _parse_tree(xml_str)
        ref_root = _parse_tree(reference_xml)
        gen_tree = _to_zss(gen_root)
        ref_tree = _to_zss(ref_root)
        return float(zss.simple_distance(gen_tree, ref_tree))
    except Exception:
        return None


def compute_port_completeness(xml_str: str, catalog: SkillsCatalog) -> float:
    """
    Fraction of required ports that are present across all skill nodes.
    Returns 1.0 if all required ports are present, 0.0 on parse failure.
    """
    valid_skills = catalog.valid_skills()
    skill_ports = catalog.skill_ports()

    xml_str = xml_str.strip()
    if xml_str.startswith("<?xml"):
        xml_str = xml_str[xml_str.index("?>") + 2 :].strip()
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return 0.0

    total_required = 0
    total_present = 0
    for elem in root.iter():
        if elem.tag in ("Action", "Condition"):
            skill_id = elem.get("ID", "")
        elif elem.tag in valid_skills:
            skill_id = elem.tag
        else:
            continue
        if skill_id not in skill_ports:
            continue
        required = skill_ports[skill_id].get("required", [])
        attrs = set(elem.attrib.keys()) - {"name", "ID"}
        for port in required:
            total_required += 1
            if port in attrs:
                total_present += 1

    if total_required == 0:
        return 1.0
    return total_present / total_required


# ── Performance Metrics ──────────────────────────────────────────────────────


def compute_vram_usage_gb() -> float:
    """Return peak VRAM usage in GB. Returns 0.0 if CUDA not available."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**3)
    except ImportError:
        pass
    return 0.0


# ── Aggregate ────────────────────────────────────────────────────────────────


@dataclass
class BenchmarkMetrics:
    """All metrics for a single generated BT."""

    # Validation
    valid: bool = False
    score: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Structure
    max_depth: int = 0
    total_nodes: int = 0
    n_behavior_trees: int = 0
    n_subtreeplus: int = 0
    n_skills: int = 0
    n_fallback: int = 0
    mean_branching_factor: float = 0.0
    # Content
    hallucinated_skills: list[str] = field(default_factory=list)
    hallucination_count: int = 0
    tree_edit_distance: float | None = None
    port_completeness: float = 0.0
    # Performance
    latency_s: float = 0.0
    n_tokens: int = 0


def compute_all_metrics(
    xml_str: str,
    reference_xml: str | None = None,
    latency_s: float = 0.0,
    n_tokens: int = 0,
    catalog: SkillsCatalog | None = None,
) -> BenchmarkMetrics:
    """Compute all metrics for a single BT generation."""
    from src.data.skills_loader import SkillsCatalog as _Cat

    if catalog is None:
        catalog = _Cat()

    # Validation
    vr = validate_bt(xml_str, catalog)

    # Structure
    struct = compute_structural_metrics(xml_str)

    # Hallucination
    halluc = compute_hallucination_metrics(xml_str, catalog)

    # TED
    ted = compute_tree_edit_distance(xml_str, reference_xml)

    # Port completeness
    port_comp = compute_port_completeness(xml_str, catalog)

    return BenchmarkMetrics(
        valid=vr.valid,
        score=vr.score,
        errors=vr.errors,
        warnings=vr.warnings,
        max_depth=struct["max_depth"],
        total_nodes=struct["total_nodes"],
        n_behavior_trees=struct["n_behavior_trees"],
        n_subtreeplus=struct["n_subtreeplus"],
        n_skills=struct["n_skills"],
        n_fallback=struct["n_fallback"],
        mean_branching_factor=struct["mean_branching_factor"],
        hallucinated_skills=halluc["hallucinated_skills"],
        hallucination_count=halluc["hallucination_count"],
        tree_edit_distance=ted,
        port_completeness=port_comp,
        latency_s=latency_s,
        n_tokens=n_tokens,
    )


def aggregate_metrics(all_metrics: list[BenchmarkMetrics]) -> dict[str, Any]:
    """Aggregate metrics across multiple BT generations into summary stats."""
    n = len(all_metrics)
    if n == 0:
        return {}
    valid_count = sum(1 for m in all_metrics if m.valid)
    perfect_count = sum(1 for m in all_metrics if m.score == 1.0)
    halluc_count = sum(1 for m in all_metrics if m.hallucination_count > 0)
    ted_values = [m.tree_edit_distance for m in all_metrics if m.tree_edit_distance is not None]

    return {
        "n_samples": n,
        "xml_validity_rate": valid_count / n,
        "mean_score": sum(m.score for m in all_metrics) / n,
        "perfect_score_rate": perfect_count / n,
        "hallucination_rate": halluc_count / n,
        "mean_hallucination_count": sum(m.hallucination_count for m in all_metrics) / n,
        "mean_tree_edit_distance": sum(ted_values) / len(ted_values) if ted_values else None,
        "mean_latency_s": sum(m.latency_s for m in all_metrics) / n,
        "mean_depth": sum(m.max_depth for m in all_metrics) / n,
        "mean_n_skills": sum(m.n_skills for m in all_metrics) / n,
        "mean_port_completeness": sum(m.port_completeness for m in all_metrics) / n,
        "vram_peak_gb": compute_vram_usage_gb(),
    }
