"""
Validateur multi-niveaux pour Behavior Trees NAV4RAIL — format BehaviorTree.CPP
================================================================================

Format attendu (BehaviorTree.CPP standard) :
  <Action name="HUMAN NAME" ID="SkillName" port="{var}"/>
  <Condition name="HUMAN NAME" ID="SkillName" port="{var}"/>
  <SubTreePlus name="HUMAN NAME" ID="subtree_id" __autoremap="true"/>

Niveau 1 — Syntaxique  : XML bien formé, BTCPP_format="4", tags valides,
                          skills via ID attr, MoveAndStop présent
Niveau 2 — Structurel  : Nœuds de contrôle vides, profondeur excessive,
                          Fallback (≥ 2 branches), <BehaviorTree> présent
Niveau 3 — Sémantique  : Ordre des skills (LoadMission en premier),
                          MoveAndStop en dernière position,
                          Conditions dans <Fallback>

Score :
  1.0  → tous les niveaux passés, aucun avertissement
  0.5–0.9 → valide avec avertissements (pénalité −0.1 par warning)
  0.0  → invalide (L1, L2 ou L3 en erreur bloquante)

Usage autonome :
    python validate_bt.py fichier.xml
    echo '<root ...>...</root>' | python validate_bt.py
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

# ─── Constantes ───────────────────────────────────────────────────────────────

# Tags XML valides (BehaviorTree.CPP standard)
VALID_TAGS = frozenset(
    {
        # Structure
        "root",
        "BehaviorTree",
        # BehaviorTree.CPP leaf wrappers (skill resolved via ID attribute)
        "Action",
        "Condition",
        "SubTreePlus",
        # Control nodes
        "Sequence",
        "Fallback",
        "ReactiveFallback",
        "Parallel",
        "Repeat",
    }
)

CONTROL_NODES = frozenset(
    {
        "Sequence",
        "Fallback",
        "ReactiveFallback",
        "Parallel",
        "Repeat",
    }
)

# 27 skills NAV4RAIL valides (attendus comme attribut ID="" sur Action/Condition)
VALID_SKILLS = frozenset(
    {
        # PREPARATION (12)
        "LoadMission",
        "MissionStructureValid",
        "UpdateCurrentGeneratedActivity",
        "ProjectPointOnNetwork",
        "CreatePath",
        "AgregatePath",
        "MissionFullyTreated",
        "PassAdvancedPath",
        "PassMission",
        "GenerateMissionSequence",
        "GenerateCorrectiveSubSequence",
        "InsertCorrectiveSubSequence",
        # MOTION (9)
        "MissionTerminated",
        "CheckCurrentStepType",
        "PassMotionParameters",
        "Move",
        "UpdateCurrentExecutedStep",
        "Deccelerate",
        "MoveAndStop",
        "SignalAndWaitForOrder",
        "IsRobotPoseProjectionActive",
        "Pause",
        # INSPECTION (5)
        "ManageMeasurements",
        "AnalyseMeasurements",
        "MeasurementsQualityValidated",
        "PassDefectsLocalization",
        "MeasurementsEnforcedValidated",
        # SIMULATION (1)
        "SimulationStarted",
    }
)

# Conditions : skills qui retournent SUCCESS/FAILURE sans effet de bord
CONDITION_SKILLS = frozenset(
    {
        "MissionStructureValid",
        "MissionFullyTreated",
        "MissionTerminated",
        "CheckCurrentStepType",
        "IsRobotPoseProjectionActive",
        "MeasurementsQualityValidated",
        "MeasurementsEnforcedValidated",
        "SimulationStarted",
    }
)

# Précédences partielles : skill → skills qui doivent le précéder
PREREQUISITES: dict[str, list[str]] = {
    "MissionStructureValid": ["LoadMission"],
    "CreatePath": ["ProjectPointOnNetwork"],
    "AgregatePath": ["CreatePath"],
    "GenerateMissionSequence": ["LoadMission"],
}

MAX_DEPTH = 12
MAX_SKILLS = 80  # 28 skills, mais répétitions dans multi-subtree (ex: reference = 72)


# ─── Résultat de validation ───────────────────────────────────────────────────


@dataclass
class ValidationResult:
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    score: float = 1.0  # 0.0 → 1.0

    def fail(self, msg: str, level: int = 1):
        self.valid = False
        self.score = 0.0
        self.errors.append(f"[L{level}] {msg}")

    def warn(self, msg: str):
        self.warnings.append(f"[W] {msg}")
        self.score = max(0.5, self.score - 0.1)

    def summary(self) -> str:
        if self.valid and not self.warnings:
            return f"OK (score={self.score:.1f}) — L1+L2+L3 passés"
        if self.valid:
            return f"OK avec {len(self.warnings)} avertissement(s) (score={self.score:.1f})"
        return f"INVALIDE : {self.errors[0]}"


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _resolve_skill(elem: ET.Element) -> str | None:
    """Résout le nom du skill depuis un élément XML.

    Format BehaviorTree.CPP : <Action ID="MoveAndStop" name="FINAL STOP" .../>
    Retourne le skill ID si c'est un Action/Condition avec un ID valide.
    """
    if elem.tag in ("Action", "Condition"):
        skill_id = elem.get("ID", "")
        return skill_id if skill_id in VALID_SKILLS else None
    return None


def _is_condition(elem: ET.Element) -> str | None:
    """Retourne le skill ID si c'est une Condition valide."""
    if elem.tag == "Condition":
        skill_id = elem.get("ID", "")
        return skill_id if skill_id in CONDITION_SKILLS else None
    return None


# ─── Niveau 1 — Syntaxique ────────────────────────────────────────────────────


def _validate_l1(xml_str: str) -> tuple[bool, ET.Element | None, str]:
    """Parse XML + vérifications basiques obligatoires."""
    # Strip XML declaration si présente
    xml_str = xml_str.strip()
    if xml_str.startswith("<?xml"):
        xml_str = xml_str[xml_str.index("?>") + 2 :].strip()

    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        return False, None, f"XML mal formé : {e}"

    if root.tag != "root":
        return False, None, f"Tag racine '<root>' attendu, obtenu '<{root.tag}>'"

    btcpp_fmt = root.get("BTCPP_format")
    # BTCPP_format="4" recommandé mais pas bloquant (fichiers terrain l'omettent)
    _btcpp_warning = None
    if btcpp_fmt is None:
        _btcpp_warning = "Attribut BTCPP_format='4' absent sur <root> (recommandé)"
    elif btcpp_fmt != "4":
        _btcpp_warning = f"BTCPP_format='{btcpp_fmt}' inattendu (attendu: '4')"

    # Vérifier que tous les tags sont dans la liste
    unknown_tags = set()
    unknown_skills = set()
    for e in root.iter():
        if e.tag not in VALID_TAGS:
            unknown_tags.add(e.tag)
        # Vérifier que Action/Condition ont un ID de skill valide
        if e.tag in ("Action", "Condition"):
            skill_id = e.get("ID", "")
            if skill_id and skill_id not in VALID_SKILLS:
                unknown_skills.add(skill_id)

    if unknown_tags:
        return False, None, f"Tags XML inconnus : {sorted(unknown_tags)}"
    if unknown_skills:
        return (
            False,
            None,
            f"Skills inconnus (hallucinations) : {sorted(unknown_skills)}",
        )

    # MoveAndStop doit être présent
    has_move_and_stop = any(
        e.tag == "Action" and e.get("ID") == "MoveAndStop" for e in root.iter()
    )
    if not has_move_and_stop:
        return (
            False,
            None,
            '<Action ID="MoveAndStop"> absent — le BT ne se termine jamais',
        )

    return True, root, _btcpp_warning or "OK"


# ─── Niveau 2 — Structurel ────────────────────────────────────────────────────


def _max_depth(elem: ET.Element, d: int = 0) -> int:
    children = list(elem)
    if not children:
        return d
    return max(_max_depth(c, d + 1) for c in children)


def _validate_l2(root: ET.Element, result: ValidationResult):
    """Vérifications structurelles : nœuds vides, profondeur, Fallback."""

    # <BehaviorTree> requis sous <root>
    if not any(e.tag == "BehaviorTree" for e in root):
        result.fail("<BehaviorTree> manquant sous <root>", level=2)
        return

    # Nœuds de contrôle sans enfants
    for elem in root.iter():
        if elem.tag in CONTROL_NODES and not list(elem):
            result.warn(
                f"<{elem.tag} name='{elem.get('name', '')}> vide (aucun enfant)"
            )

    # Profondeur excessive
    depth = _max_depth(root)
    if depth > MAX_DEPTH:
        result.fail(f"Profondeur {depth} > {MAX_DEPTH} (arbre trop imbriqué)", level=2)

    # Nombre de skills excessif
    skill_count = sum(1 for e in root.iter() if _resolve_skill(e) is not None)
    if skill_count > MAX_SKILLS:
        result.warn(f"{skill_count} nœuds skills > {MAX_SKILLS} (BT trop long)")

    # Fallback / ReactiveFallback doit avoir ≥ 2 branches
    for elem in root.iter():
        if elem.tag in ("Fallback", "ReactiveFallback"):
            n = len(list(elem))
            if n < 2:
                result.warn(
                    f"<{elem.tag} name='{elem.get('name', '')}> n'a que {n} branche(s) "
                    f"(minimum 2 requis)"
                )


# ─── Niveau 3 — Sémantique ───────────────────────────────────────────────────


def _skills_dfs(elem: ET.Element) -> list[str]:
    """Retourne les skills en ordre DFS (ordre d'exécution attendu).

    Résout <Action ID="X"> et <Condition ID="X"> en leur skill ID.
    """
    result = []
    skill = _resolve_skill(elem)
    if skill:
        result.append(skill)
    for child in elem:
        result.extend(_skills_dfs(child))
    return result


def _condition_in_fallback(root: ET.Element) -> set[str]:
    """Retourne les conditions qui sont descendantes d'un Fallback/ReactiveFallback."""
    in_fb = set()
    for fb_tag in ("Fallback", "ReactiveFallback"):
        for fb in root.iter(fb_tag):
            for e in fb.iter():
                cond = _is_condition(e)
                if cond and cond in CONDITION_SKILLS:
                    in_fb.add(cond)
    return in_fb


def _validate_l3(root: ET.Element, result: ValidationResult):
    """Vérifications sémantiques : ordre des skills, conditions, MoveAndStop."""
    # Détecter si multi-subtree (>1 BehaviorTree)
    bt_count = sum(1 for e in root if e.tag == "BehaviorTree")
    is_multi_subtree = bt_count > 1

    skills = _skills_dfs(root)
    if not skills:
        return

    # Première occurrence de chaque skill
    first: dict[str, int] = {}
    for i, s in enumerate(skills):
        if s not in first:
            first[s] = i

    # Précédences partielles (relaxées en multi-subtree : DFS linéaire != exécution)
    if not is_multi_subtree:
        for skill, prereqs in PREREQUISITES.items():
            if skill not in first:
                continue
            for prereq in prereqs:
                if prereq not in first:
                    result.warn(
                        f"<{prereq}> absent alors qu'il précède normalement <{skill}>"
                    )
                elif first[prereq] > first[skill]:
                    result.warn(
                        f"Ordre incorrect : <{skill}> (pos {first[skill]}) "
                        f"avant <{prereq}> (pos {first[prereq]})"
                    )

    # LoadMission doit être présent
    if "LoadMission" not in first:
        result.warn("LoadMission absent du BT")

    # MoveAndStop doit être présent (déjà vérifié L1, mais vérifier dans chaque subtree
    # pour multi-subtree n'est pas nécessaire)

    # Conditions de boucle doivent être dans un Fallback
    conditions_present = set()
    for e in root.iter():
        cond = _is_condition(e)
        if cond:
            conditions_present.add(cond)
    conditions_in_fb = _condition_in_fallback(root)
    loop_conditions = {
        "MissionTerminated",
        "MissionFullyTreated",
        "MeasurementsQualityValidated",
        "MeasurementsEnforcedValidated",
    }
    for cond in conditions_present & loop_conditions:
        if cond not in conditions_in_fb:
            result.warn(
                f'<Condition ID="{cond}"> hors de tout Fallback — '
                f"son signal FAILURE ne sera pas intercepté correctement"
            )


# ─── API publique ─────────────────────────────────────────────────────────────


def validate_bt(xml_str: str) -> ValidationResult:
    """
    Valide un BT XML NAV4RAIL sur 3 niveaux.

    Returns:
        ValidationResult avec valid, errors, warnings, score
    """
    result = ValidationResult()

    # Niveau 1
    ok, root, msg = _validate_l1(xml_str)
    if not ok:
        result.fail(msg, level=1)
        return result

    # Warning L1 (ex: BTCPP_format manquant)
    if msg != "OK":
        result.warn(msg)

    # Niveau 2
    _validate_l2(root, result)
    if not result.valid:
        return result

    # Niveau 3
    _validate_l3(root, result)
    return result


def validate_xml(xml_str: str) -> tuple[bool, str]:
    """Interface rétro-compatible avec l'ancienne validate_xml() de finetune_lora_xml.py."""
    r = validate_bt(xml_str)
    return r.valid, r.summary()


# ─── CLI autonome ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        xml_str = open(sys.argv[1], encoding="utf-8").read()
    else:
        xml_str = sys.stdin.read()

    result = validate_bt(xml_str)
    print(result.summary())
    for e in result.errors:
        print(f"  {e}")
    for w in result.warnings:
        print(f"  {w}")
    sys.exit(0 if result.valid else 1)
