"""
Validateur multi-niveaux pour Behavior Trees NAV4RAIL
======================================================

Niveau 1 — Syntaxique  : XML bien formé, BTCPP_format, tag allowlist, <Stop> présent
Niveau 2 — Structurel  : Nœuds de contrôle vides, profondeur excessive,
                          structure Fallback (≥ 2 branches), <BehaviorTree> présent
Niveau 3 — Sémantique  : Ordre des skills (GetMission → CalculatePath → Move),
                          <Stop> en dernière position, <CheckObstacle> dans <Fallback>

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
    "root", "BehaviorTree", "Sequence", "Fallback", "Parallel",
    "GetMission", "CalculatePath", "Move", "Decelerate",
    "ManageMeasurement", "CheckObstacle", "Alert", "Stop",
})

CONTROL_NODES = frozenset({"Sequence", "Fallback", "Parallel"})

SKILL_NODES = frozenset({
    "GetMission", "CalculatePath", "Move", "Decelerate",
    "ManageMeasurement", "CheckObstacle", "Alert", "Stop",
})

# Précédences partielles : skill → skills qui doivent le précéder
PREREQUISITES: dict[str, list[str]] = {
    "CalculatePath": ["GetMission"],
    "Move":          ["CalculatePath"],
}

MAX_DEPTH  = 10   # profondeur maximale de l'arbre
MAX_SKILLS = 25   # nombre maximal de nœuds skills


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

    if not any(e.tag == "Stop" for e in root.iter()):
        return False, None, "<Stop> absent — le BT ne se termine jamais"

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


def _check_obstacle_in_fallback(root: ET.Element) -> bool:
    """Vérifie qu'au moins un <CheckObstacle> est descendant d'un <Fallback>."""
    for fb in root.iter("Fallback"):
        if any(e.tag == "CheckObstacle" for e in fb.iter()):
            return True
    return False


def _validate_l3(root: ET.Element, result: ValidationResult):
    """Vérifications sémantiques : ordre des skills, CheckObstacle, Stop."""
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

    # <GetMission> doit être en premier
    if "GetMission" in first and first["GetMission"] != 0:
        result.warn(
            f"<GetMission> n'est pas en première position "
            f"(position {first['GetMission']})"
        )

    # Le dernier <Stop> doit être le dernier skill
    if "Stop" in first:
        last_stop = max(i for i, s in enumerate(skills) if s == "Stop")
        if last_stop < len(skills) - 1:
            result.warn(
                f"<Stop> n'est pas le dernier skill exécuté "
                f"(position {last_stop}/{len(skills) - 1})"
            )

    # <CheckObstacle> doit être dans un contexte <Fallback>
    has_check = any(e.tag == "CheckObstacle" for e in root.iter())
    if has_check and not _check_obstacle_in_fallback(root):
        result.warn(
            "<CheckObstacle> présent hors de tout <Fallback> — "
            "son signal FAILURE ne sera pas intercepté correctement"
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
