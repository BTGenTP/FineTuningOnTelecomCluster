"""
Skills Catalog Loader
=====================
Loads skills_catalog.yaml and safety_rules.yaml as structured data.
Provides the same constants that validate_bt.py previously hardcoded.

Usage:
    from src.data.skills_loader import SkillsCatalog
    catalog = SkillsCatalog("data/skills_catalog.yaml")
    print(catalog.valid_skills())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PortSpec:
    """Specification for a single skill port."""

    direction: str  # "input" or "output"
    type: str  # "bb_var", "int_literal", "float_literal", "string_literal"
    required: bool = True
    default: str | None = None
    allowed: list[str] | None = None


@dataclass
class SkillSpec:
    """Specification for a single NAV4RAIL skill."""

    id: str
    family: str
    bt_tag: str  # "Action" or "Condition"
    ports: dict[str, PortSpec]
    prerequisites: list[str]
    description: str = ""


@dataclass
class SafetyRule:
    """A single safety rule."""

    id: str
    name: str
    description: str
    level: str
    enforcement: list[str]
    blocking: bool


class SkillsCatalog:
    """
    Single source of truth for NAV4RAIL skills, ports, and constraints.
    Replaces all hardcoded constants in the codebase.
    """

    def __init__(self, catalog_path: str | Path | None = None):
        if catalog_path is None:
            catalog_path = Path(__file__).parent.parent.parent / "data" / "skills_catalog.yaml"
        self._path = Path(catalog_path)
        self._raw: dict = {}
        self._skills: dict[str, SkillSpec] = {}
        self._load()

    def _load(self) -> None:
        with open(self._path, encoding="utf-8") as f:
            self._raw = yaml.safe_load(f)
        self._parse_skills()

    def _parse_skills(self) -> None:
        for family_name, family_data in self._raw.get("families", {}).items():
            for skill_id, skill_data in family_data.get("skills", {}).items():
                ports = {}
                for port_name, port_data in skill_data.get("ports", {}).items():
                    if isinstance(port_data, dict):
                        ports[port_name] = PortSpec(
                            direction=port_data.get("direction", "input"),
                            type=port_data.get("type", "bb_var"),
                            required=port_data.get("required", False),
                            default=port_data.get("default"),
                            allowed=port_data.get("allowed"),
                        )
                self._skills[skill_id] = SkillSpec(
                    id=skill_id,
                    family=family_name,
                    bt_tag=skill_data.get("bt_tag", "Action"),
                    ports=ports,
                    prerequisites=skill_data.get("prerequisites", []),
                    description=skill_data.get("description", ""),
                )

    # ── Accessors (replacing hardcoded constants in validate_bt.py) ──────────

    def valid_skills(self) -> frozenset[str]:
        """All valid skill IDs (count read from catalog, not hardcoded)."""
        return frozenset(self._skills.keys())

    def condition_skills(self) -> frozenset[str]:
        """Skills with bt_tag='Condition'. Replaces CONDITION_SKILLS."""
        return frozenset(
            sid for sid, spec in self._skills.items() if spec.bt_tag == "Condition"
        )

    def action_skills(self) -> frozenset[str]:
        """Skills with bt_tag='Action'."""
        return frozenset(
            sid for sid, spec in self._skills.items() if spec.bt_tag == "Action"
        )

    def skill_ports(self) -> dict[str, dict]:
        """
        Port specifications per skill in validate_bt.py format.
        Replaces SKILL_PORTS.

        Returns:
            {skill_id: {"required": [...], "types": {...}, "allowed": {...}}}
        """
        result = {}
        for sid, spec in self._skills.items():
            entry: dict[str, Any] = {
                "required": [
                    pname for pname, pspec in spec.ports.items() if pspec.required
                ]
            }
            types = {pname: pspec.type for pname, pspec in spec.ports.items()}
            if types:
                entry["types"] = types
            allowed = {
                pname: set(pspec.allowed)
                for pname, pspec in spec.ports.items()
                if pspec.allowed
            }
            if allowed:
                entry["allowed"] = allowed
            result[sid] = entry
        return result

    def default_port_values(self) -> dict[str, dict[str, str]]:
        """
        Default values for ports. Replaces _DEFAULT_PORT_VALUES.

        Returns:
            {skill_id: {port_name: default_value}}
        """
        result = {}
        for sid, spec in self._skills.items():
            defaults = {}
            for pname, pspec in spec.ports.items():
                if pspec.default is not None:
                    defaults[pname] = pspec.default
            if defaults:
                result[sid] = defaults
        return result

    def prerequisites(self) -> dict[str, list[str]]:
        """Partial ordering constraints. Replaces PREREQUISITES."""
        return dict(self._raw.get("prerequisites", {}))

    def valid_tags(self) -> frozenset[str]:
        """All valid XML tags. Replaces VALID_TAGS."""
        return frozenset(self._raw.get("valid_tags", []))

    def control_nodes(self) -> frozenset[str]:
        """Control node tag names. Replaces CONTROL_NODES."""
        return frozenset(self._raw.get("control_nodes", []))

    def step_types(self) -> dict[str, list[dict]]:
        """Motion step type definitions (transport + inspection)."""
        return dict(self._raw.get("step_types", {}))

    def limits(self) -> dict[str, int]:
        """Structural limits (max_depth, max_skills, min_fallback_children)."""
        return dict(self._raw.get("limits", {}))

    def get_skill(self, skill_id: str) -> SkillSpec | None:
        """Get spec for a single skill."""
        return self._skills.get(skill_id)

    def families(self) -> dict[str, list[str]]:
        """Map of family name to list of skill IDs."""
        result: dict[str, list[str]] = {}
        for sid, spec in self._skills.items():
            result.setdefault(spec.family, []).append(sid)
        return result

    def all_skills(self) -> dict[str, SkillSpec]:
        """All skill specs keyed by ID."""
        return dict(self._skills)

    # ── Prompt generation ────────────────────────────────────────────────────

    def summarize(self) -> str:
        """
        Human-readable summary for injection into LLM prompts.
        Lists skills grouped by family with port info.
        """
        lines = [f"NAV4RAIL Skills Catalog (v{self._raw.get('metadata', {}).get('version', '?')})", ""]
        for family_name, skill_ids in self.families().items():
            lines.append(f"## {family_name.upper()} ({len(skill_ids)} skills)")
            for sid in skill_ids:
                spec = self._skills[sid]
                tag_type = "Condition" if spec.bt_tag == "Condition" else "Action"
                port_strs = []
                for pname, pspec in spec.ports.items():
                    req = "*" if pspec.required else ""
                    port_strs.append(f"{pname}{req}:{pspec.type}")
                ports_line = f" [{', '.join(port_strs)}]" if port_strs else ""
                lines.append(f"  - {sid} ({tag_type}){ports_line}")
            lines.append("")
        return "\n".join(lines)


class SafetyRulesLoader:
    """Loads and provides access to safety_rules.yaml."""

    def __init__(self, rules_path: str | Path | None = None):
        if rules_path is None:
            rules_path = Path(__file__).parent.parent.parent / "data" / "safety_rules.yaml"
        with open(rules_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self._rules = []
        for r in raw.get("rules", []):
            self._rules.append(
                SafetyRule(
                    id=r["id"],
                    name=r["name"],
                    description=r["description"],
                    level=r["level"],
                    enforcement=r.get("enforcement", []),
                    blocking=r.get("blocking", True),
                )
            )

    def all_rules(self) -> list[SafetyRule]:
        return list(self._rules)

    def rules_for_level(self, level: str) -> list[SafetyRule]:
        return [r for r in self._rules if r.level == level]

    def blocking_rules(self) -> list[SafetyRule]:
        return [r for r in self._rules if r.blocking]

    def rules_for_prompt(self) -> list[SafetyRule]:
        """Rules that should be injected into the LLM prompt."""
        return [r for r in self._rules if "prompt" in r.enforcement]

    def summarize_for_prompt(self) -> str:
        """Format rules for injection into LLM system prompt."""
        lines = ["REGLES DE SECURITE FERROVIAIRE NAV4RAIL:", ""]
        for rule in self.rules_for_prompt():
            lines.append(f"- [{rule.id}] {rule.name}: {rule.description.strip()}")
        return "\n".join(lines)
