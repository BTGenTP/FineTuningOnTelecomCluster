"""
Reward function for GRPO/PPO reinforcement learning.
=====================================================
Uses validate_bt.py as the core reward signal.
Compatible with trl.GRPOTrainer reward function signature.

Usage:
    from src.reward.reward_fn import make_reward_fn
    reward_fn = make_reward_fn(catalog, config)
    rewards = reward_fn(completions=["<root>...</root>"], prompts=["mission..."])
"""

from __future__ import annotations

import re
from typing import Any, Callable

from src.data.skills_loader import SkillsCatalog
from src.eval.validate_bt import validate_bt


def _extract_xml(text: str) -> str | None:
    """Extract XML from a generated completion. Returns None if no XML found."""
    # Try to find <root>...</root> block
    match = re.search(r"<root[\s>].*?</root>", text, re.DOTALL)
    if match:
        return match.group(0)
    # Try to find any XML-like content
    if "<root" in text:
        idx = text.index("<root")
        return text[idx:]
    return None


def _classify_mission(mission: str) -> str:
    """Classify mission type from natural language description."""
    mission_lower = mission.lower()

    has_inspection = any(
        w in mission_lower
        for w in ["inspection", "inspecter", "mesure", "releve", "controle"]
    )
    has_control = any(
        w in mission_lower
        for w in ["verifier", "controler", "analyser", "qualite", "correction", "corriger"]
    )
    has_simulation = "simulation" in mission_lower or "simuler" in mission_lower

    if has_simulation:
        return "simulation"
    elif has_inspection and has_control:
        return "inspection_avec_controle"
    elif has_inspection:
        return "inspection_sans_controle"
    else:
        return "transport"


def _check_mission_coherence(xml_str: str, mission_type: str) -> float:
    """Check semantic coherence between mission type and BT content."""
    has_manage = "ManageMeasurements" in xml_str
    has_analyse = "AnalyseMeasurements" in xml_str
    has_simulation = "SimulationStarted" in xml_str

    if mission_type == "transport":
        # Transport should NOT have inspection skills
        return 1.0 if not has_manage else 0.5
    elif mission_type == "inspection_avec_controle":
        # Should have ManageMeasurements AND AnalyseMeasurements
        if has_manage and has_analyse:
            return 1.0
        elif has_manage:
            return 0.7
        return 0.3
    elif mission_type == "inspection_sans_controle":
        # Should have ManageMeasurements but NOT AnalyseMeasurements
        if has_manage and not has_analyse:
            return 1.0
        elif has_manage:
            return 0.8
        return 0.3
    elif mission_type == "simulation":
        return 1.0 if has_simulation else 0.5
    return 0.5


def compute_reward(
    prompt: str,
    completion: str,
    catalog: SkillsCatalog,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Compute reward for a single (prompt, completion) pair.

    Returns float in [-1.0, 1.0].
    """
    if weights is None:
        weights = {
            "parse_weight": 0.3,
            "structure_weight": 0.2,
            "semantic_weight": 0.2,
            "coherence_weight": 0.2,
            "hallucination_weight": 0.1,
        }

    xml_str = _extract_xml(completion)
    if xml_str is None:
        return -1.0

    # Validate
    result = validate_bt(xml_str, catalog)

    # Parse component
    r_parse = 1.0 if result.score > 0 else -1.0

    # Structure component
    r_structure = result.score if result.score > 0 else -0.5

    # Semantic component (warnings penalty)
    r_semantic = max(0.0, 1.0 - 0.1 * len(result.warnings))

    # Mission coherence
    mission_type = _classify_mission(prompt)
    r_coherence = _check_mission_coherence(xml_str, mission_type)

    # Hallucination check
    valid_skills = catalog.valid_skills()
    import xml.etree.ElementTree as ET

    r_hallucination = 1.0
    try:
        clean = xml_str.strip()
        if clean.startswith("<?xml"):
            clean = clean[clean.index("?>") + 2 :].strip()
        root = ET.fromstring(clean)
        for e in root.iter():
            if e.tag in ("Action", "Condition"):
                sid = e.get("ID", "")
                if sid and sid not in valid_skills:
                    r_hallucination -= 0.5
    except ET.ParseError:
        r_hallucination = 0.0

    r_hallucination = max(0.0, r_hallucination)

    # Weighted sum
    reward = (
        weights["parse_weight"] * r_parse
        + weights["structure_weight"] * r_structure
        + weights["semantic_weight"] * r_semantic
        + weights["coherence_weight"] * r_coherence
        + weights["hallucination_weight"] * r_hallucination
    )

    return max(-1.0, min(1.0, reward))


def make_reward_fn(
    catalog: SkillsCatalog | None = None,
    config: dict | None = None,
) -> Callable[[list[str], list[str]], list[float]]:
    """
    Create a reward function compatible with trl.GRPOTrainer.

    Args:
        catalog: SkillsCatalog instance
        config: Config dict with reward.components weights

    Returns:
        Callable(completions, prompts) -> list[float]
    """
    if catalog is None:
        catalog = SkillsCatalog()

    weights = None
    if config:
        weights = config.get("reward", {}).get("components")

    def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            try:
                r = compute_reward(prompt, completion, catalog, weights)
            except Exception:
                r = -1.0
            rewards.append(r)
        return rewards

    return reward_fn
