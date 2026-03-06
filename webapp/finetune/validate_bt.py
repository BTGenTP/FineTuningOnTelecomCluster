"""
Validateur multi-niveaux pour Behavior Trees NAV4RAIL (v4 — 27 skills réels)
=============================================================================

Niveau 1 — Syntaxique  : XML bien formé, BTCPP_format, tag allowlist, <MoveAndStop> présent
Niveau 2 — Structurel  : Nœuds de contrôle vides, profondeur excessive,
                          structure Fallback (≥ 2 branches), <BehaviorTree> présent
Niveau 3 — Sémantique  : Ordre des skills (LoadMission en premier),
                          <MoveAndStop> en dernière position,
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

VALID_TAGS = frozenset({
    # Structure
    "root", "BehaviorTree", "Sequence", "Fallback", "Parallel",
    # PREPARATION (12)
    "LoadMission", "MissionStructureValid", "UpdateCurrentGeneratedActivity",
    "ProjectPointOnNetwork", "CreatePath", "AgregatePath", "MissionFullyTreated",
    "PassAdvancedPath", "PassMission", "GenerateMissionSequence",
    "GenerateCorrectiveSubSequence", "InsertCorrectiveSubSequence",
    # MOTION (9)
    "MissionTerminated", "CheckCurrentStepType", "PassMotionParameters",
    "Move", "UpdateCurrentExecutedStep", "Deccelerate", "MoveAndStop",
    "SignalAndWaitForOrder", "IsRobotPoseProjectionActive",
    # INSPECTION (5)
    "ManageMeasurements", "AnalyseMeasurements", "MeasurementsQualityValidated",
    "PassDefectsLocalization", "MeasurementsEnforcedValidated",
    # SIMULATION (1)
    "SimulationStarted",
})

CONTROL_NODES = frozenset({"Sequence", "Fallback", "Parallel"})

SKILL_NODES = VALID_TAGS - {"root", "BehaviorTree"} - CONTROL_NODES

# Conditions : skills qui retournent SUCCESS/FAILURE sans effet de bord
CONDITION_NODES = frozenset({
    "MissionStructureValid", "MissionFullyTreated",
    "MissionTerminated", "CheckCurrentStepType",
    "IsRobotPoseProjectionActive",
    "MeasurementsQualityValidated", "MeasurementsEnforcedValidated",
    "SimulationStarted",
})

# Précédences partielles : skill → skills qui doivent le précéder
PREREQUISITES: dict[str, list[str]] = {
    "MissionStructureValid":  ["LoadMission"],
    "CreatePath":             ["ProjectPointOnNetwork"],
    "AgregatePath":           ["CreatePath"],
    "GenerateMissionSequence": ["LoadMission"],
}

MAX_DEPTH  = 12   # profondeur maximale (BTs v4 plus profonds)
MAX_SKILLS = 35   # nombre maximal de nœuds skills (27 skills, répétitions possibles)


# ─── Résultat de validation ───────────────────────────────────────────────────

@dataclass
class ValidationResult:
    valid:    bool       = True
    errors:   list[str]  = field(default_factory=list)
    warnings: list[str]  = field(default_factory=list)
    score:    float      = 1.0   # 0.0 → 1.0

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


# ─── Niveau 1 — Syntaxique ────────────────────────────────────────────────────

def _validate_l1(xml_str: str) -> tuple[bool, ET.Element | None, str]:
    """Parse XML + vérifications basiques obligatoires."""
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        return False, None, f"XML mal formé : {e}"

    if root.tag != "root":
        return False, None, f"Tag racine '<root>' attendu, obtenu '<{root.tag}>'"

    if root.get("BTCPP_format") != "4":
        return False, None, "Attribut BTCPP_format='4' manquant sur <root>"

    unknown = sorted({e.tag for e in root.iter() if e.tag not in VALID_TAGS})
    if unknown:
        return False, None, f"Tags inconnus / hallucinations : {unknown}"

    if not any(e.tag == "MoveAndStop" for e in root.iter()):
        return False, None, "<MoveAndStop> absent — le BT ne se termine jamais"

    return True, root, "OK"


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
    skill_count = sum(1 for e in root.iter() if e.tag in SKILL_NODES)
    if skill_count > MAX_SKILLS:
        result.warn(f"{skill_count} nœuds skills > {MAX_SKILLS} (BT trop long)")

    # Fallback doit avoir ≥ 2 branches
    for elem in root.iter():
        if elem.tag == "Fallback":
            n = len(list(elem))
            if n < 2:
                result.warn(
                    f"<Fallback name='{elem.get('name', '')}> n'a que {n} branche(s) "
                    f"(minimum 2 requis pour un if/else correct)"
                )


# ─── Niveau 3 — Sémantique ───────────────────────────────────────────────────

def _skills_dfs(elem: ET.Element) -> list[str]:
    """Retourne les tags skills en ordre DFS (ordre d'exécution attendu)."""
    result = []
    if elem.tag in SKILL_NODES:
        result.append(elem.tag)
    for child in elem:
        result.extend(_skills_dfs(child))
    return result


def _condition_in_fallback(root: ET.Element) -> set[str]:
    """Retourne les conditions qui sont bien descendantes d'un <Fallback>."""
    in_fb = set()
    for fb in root.iter("Fallback"):
        for e in fb.iter():
            if e.tag in CONDITION_NODES:
                in_fb.add(e.tag)
    return in_fb


def _validate_l3(root: ET.Element, result: ValidationResult):
    """Vérifications sémantiques : ordre des skills, conditions, MoveAndStop."""
    skills = _skills_dfs(root)
    if not skills:
        return

    # Première occurrence de chaque skill
    first: dict[str, int] = {}
    for i, s in enumerate(skills):
        if s not in first:
            first[s] = i

    # Précédences partielles
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

    # <LoadMission> doit être le premier skill (sauf SimulationStarted)
    first_skill = "LoadMission" if "SimulationStarted" not in first else "SimulationStarted"
    if first_skill in first and first[first_skill] != 0:
        result.warn(
            f"<{first_skill}> n'est pas en première position "
            f"(position {first[first_skill]})"
        )

    # Le dernier <MoveAndStop> doit être le dernier (ou avant-dernier) skill
    if "MoveAndStop" in first:
        last_stop = max(i for i, s in enumerate(skills) if s == "MoveAndStop")
        # Tolérance : MoveAndStop peut être suivi de SignalAndWaitForOrder
        if last_stop < len(skills) - 2:
            result.warn(
                f"<MoveAndStop> n'est pas en fin de séquence "
                f"(position {last_stop}/{len(skills) - 1})"
            )

    # Conditions (MissionTerminated, MissionFullyTreated, etc.) dans un Fallback
    conditions_present = {e.tag for e in root.iter() if e.tag in CONDITION_NODES}
    conditions_in_fb = _condition_in_fallback(root)
    # Exclure les conditions de début de séquence (MissionStructureValid, SimulationStarted,
    # IsRobotPoseProjectionActive) qui peuvent être hors Fallback
    loop_conditions = {"MissionTerminated", "MissionFullyTreated",
                       "MeasurementsQualityValidated", "MeasurementsEnforcedValidated"}
    for cond in conditions_present & loop_conditions:
        if cond not in conditions_in_fb:
            result.warn(
                f"<{cond}> présent hors de tout <Fallback> — "
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
