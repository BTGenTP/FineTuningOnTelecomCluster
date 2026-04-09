"""
Validateur multi-niveaux pour Behavior Trees NAV4RAIL
================================================================================

Formats acceptés :
  v5 (BehaviorTree.CPP) : <Action name="HUMAN NAME" ID="SkillName" port="{var}"/>
  v4 (proxy)            : <SkillName name="human name" port="{var}"/>

Niveau 1 — Syntaxique  : XML bien formé, BTCPP_format="4", tags valides,
                          skills (tag ou ID attr), MoveAndStop présent
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

# ─── Catalogue des ports par skill ─────────────────────────────────────────────
# Dérivé du SKILLS_DOC de generate_dataset_llm.py + few-shot example

SKILL_PORTS: dict[str, dict] = {
    # PREPARATION
    "LoadMission": {
        "required": ["mission_file_path"],
        "types": {"mission_file_path": "bb_var"},
    },
    "MissionStructureValid": {"required": []},
    "UpdateCurrentGeneratedActivity": {
        "required": ["type", "origin_sph", "target_sph", "forbidden_atoms_out"],
        "types": {
            "type": "bb_var",
            "origin_sph": "bb_var",
            "target_sph": "bb_var",
            "forbidden_atoms_out": "bb_var",
        },
    },
    "ProjectPointOnNetwork": {
        "required": ["point_in", "point_out"],
        "types": {"point_in": "bb_var", "point_out": "bb_var"},
    },
    "CreatePath": {
        "required": ["origin", "target", "forbidden_atoms", "path"],
        "types": {
            "origin": "bb_var",
            "target": "bb_var",
            "forbidden_atoms": "bb_var",
            "path": "bb_var",
        },
    },
    "AgregatePath": {"required": ["path"], "types": {"path": "bb_var"}},
    "MissionFullyTreated": {"required": ["type"], "types": {"type": "bb_var"}},
    "PassAdvancedPath": {"required": ["adv_path"], "types": {"adv_path": "bb_var"}},
    "PassMission": {"required": ["mission"], "types": {"mission": "bb_var"}},
    "GenerateMissionSequence": {
        "required": ["mission", "mission_sequence"],
        "types": {"mission": "bb_var", "mission_sequence": "bb_var"},
    },
    "GenerateCorrectiveSubSequence": {
        "required": ["defects"],
        "types": {"defects": "bb_var"},
    },
    "InsertCorrectiveSubSequence": {"required": []},
    # MOTION
    "MissionTerminated": {"required": []},
    "CheckCurrentStepType": {
        "required": ["type_to_be_checked"],
        "types": {"type_to_be_checked": "int_literal"},
        "allowed": {
            "type_to_be_checked": {
                "0",
                "1",
                "2",
                "3",
                "4",
                "10",
                "11",
                "12",
                "13",
                "14",
            }
        },
    },
    "PassMotionParameters": {
        "required": ["motion_params"],
        "types": {"motion_params": "bb_var"},
    },
    "Move": {
        "required": ["threshold_type", "motion_params"],
        "types": {"threshold_type": "int_literal", "motion_params": "bb_var"},
        "allowed": {"threshold_type": {"1", "3"}},
    },
    "UpdateCurrentExecutedStep": {"required": []},
    "Deccelerate": {
        "required": ["motion_params"],
        "types": {"motion_params": "bb_var"},
    },
    "MoveAndStop": {
        "required": ["motion_params"],
        "types": {"motion_params": "bb_var"},
    },
    "SignalAndWaitForOrder": {
        "required": ["message"],
        "types": {"message": "string_literal"},
    },
    "IsRobotPoseProjectionActive": {
        "required": ["adv_path", "pub_proj"],
        "types": {"adv_path": "bb_var", "pub_proj": "bb_var"},
    },
    "Pause": {"required": ["duration"], "types": {"duration": "float_literal"}},
    # INSPECTION
    "ManageMeasurements": {"required": []},
    "AnalyseMeasurements": {"required": []},
    "MeasurementsQualityValidated": {"required": []},
    "PassDefectsLocalization": {
        "required": ["defects"],
        "types": {"defects": "bb_var"},
    },
    "MeasurementsEnforcedValidated": {"required": []},
    # SIMULATION
    "SimulationStarted": {"required": []},
}

# Attributs non-fonctionnels (présents sur tous les nœuds, pas des ports)
_META_ATTRS = frozenset({"name", "ID"})


def _l4_label(elem: ET.Element, skill_id: str) -> str:
    """Label lisible pour les messages L4 (v4: <Skill>, v5: <Action ID='Skill'>)."""
    if elem.tag in VALID_SKILLS:
        return elem.tag
    return f'{elem.tag} ID="{skill_id}"'

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

    Format v5 (BehaviorTree.CPP) : <Action ID="MoveAndStop" name="FINAL STOP" .../>
    Format v4 (proxy)            : <MoveAndStop name="..."/>
    """
    if elem.tag in ("Action", "Condition"):
        skill_id = elem.get("ID", "")
        return skill_id if skill_id in VALID_SKILLS else None
    if elem.tag in VALID_SKILLS:
        return elem.tag
    return None


def _is_condition(elem: ET.Element) -> str | None:
    """Retourne le skill ID si c'est une Condition valide (v4 ou v5)."""
    if elem.tag == "Condition":
        skill_id = elem.get("ID", "")
        return skill_id if skill_id in CONDITION_SKILLS else None
    if elem.tag in CONDITION_SKILLS:
        return elem.tag
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

    # Vérifier que tous les tags sont dans la liste (accepter v4 skill-as-tag ET v5 Action/Condition)
    _all_valid_tags = VALID_TAGS | VALID_SKILLS
    unknown_tags = set()
    unknown_skills = set()
    for e in root.iter():
        if e.tag not in _all_valid_tags:
            unknown_tags.add(e.tag)
        # Vérifier que Action/Condition (v5) ont un ID de skill valide
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

    # MoveAndStop doit être présent (v5: <Action ID="MoveAndStop">, v4: <MoveAndStop>)
    has_move_and_stop = any(
        (e.tag == "Action" and e.get("ID") == "MoveAndStop")
        or e.tag == "MoveAndStop"
        for e in root.iter()
    )
    if not has_move_and_stop:
        return (
            False,
            None,
            'MoveAndStop absent — le BT ne se termine jamais',
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
                f'<{cond}> hors de tout Fallback — '
                f"son signal FAILURE ne sera pas intercepté correctement"
            )


# ─── Niveau 4 — Ports ─────────────────────────────────────────────────────────


def _validate_l4(root: ET.Element, result: ValidationResult):
    """Vérifications des ports : attributs requis, types, valeurs licites."""

    for elem in root.iter():
        # v5: <Action ID="Skill">, v4: <Skill ...>
        if elem.tag in ("Action", "Condition"):
            skill_id = elem.get("ID", "")
        elif elem.tag in VALID_SKILLS:
            skill_id = elem.tag
        else:
            continue
        if skill_id not in SKILL_PORTS:
            continue  # L1 attrape déjà les skills inconnus

        spec = SKILL_PORTS[skill_id]
        required = spec.get("required", [])
        types = spec.get("types", {})
        allowed = spec.get("allowed", {})
        node_attrs = {k: v for k, v in elem.attrib.items() if k not in _META_ATTRS}

        # Ports requis manquants
        for port in required:
            if port not in node_attrs:
                result.warn(
                    f'[L4] <{_l4_label(elem, skill_id)}> : port requis "{port}" manquant'
                )

        # Attributs inconnus
        known_ports = set(required)
        for attr in node_attrs:
            if attr not in known_ports and attr not in types:
                result.warn(
                    f'[L4] <{_l4_label(elem, skill_id)}> : attribut inconnu "{attr}"'
                )

        # Type + valeur
        for port, ptype in types.items():
            val = node_attrs.get(port)
            if val is None:
                continue
            _label = _l4_label(elem, skill_id)
            if ptype == "bb_var":
                if not (val.startswith("{") and val.endswith("}")):
                    result.warn(
                        f'[L4] <{_label}> : '
                        f'port "{port}" devrait être {{variable}}, reçu "{val}"'
                    )
            elif ptype == "int_literal":
                try:
                    int(val)
                except ValueError:
                    result.warn(
                        f'[L4] <{_label}> : '
                        f'port "{port}" devrait être un entier, reçu "{val}"'
                    )
            elif ptype == "float_literal":
                try:
                    float(val)
                except ValueError:
                    result.warn(
                        f'[L4] <{_label}> : '
                        f'port "{port}" devrait être un flottant, reçu "{val}"'
                    )
            # Valeurs autorisées
            if port in allowed and val not in allowed[port]:
                result.warn(
                    f'[L4] <{_label}> : '
                    f'port "{port}"="{val}" hors domaine '
                    f"{sorted(allowed[port])}"
                )

    # SubTreePlus : ID + __autoremap requis
    for elem in root.iter("SubTreePlus"):
        if not elem.get("ID"):
            result.warn("[L4] <SubTreePlus> sans attribut ID")
        if elem.get("__autoremap") is None:
            result.warn(
                f'[L4] <SubTreePlus ID="{elem.get("ID", "?")}"> sans __autoremap'
            )

    # Repeat : num_cycles requis
    for elem in root.iter("Repeat"):
        if elem.get("num_cycles") is None:
            result.warn(f'[L4] <Repeat name="{elem.get("name", "?")}"> sans num_cycles')


# ─── API publique ─────────────────────────────────────────────────────────────


def validate_bt(xml_str: str) -> ValidationResult:
    """
    Valide un BT XML NAV4RAIL sur 4 niveaux.

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

    # Niveau 4
    _validate_l4(root, result)

    return result


def validate_ports(xml_str: str) -> list[str]:
    """Valide uniquement les ports (L4). Retourne la liste des issues."""
    xml_str = xml_str.strip()
    if xml_str.startswith("<?xml"):
        xml_str = xml_str[xml_str.index("?>") + 2 :].strip()
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return ["XML mal formé"]
    result = ValidationResult()
    _validate_l4(root, result)
    return result.warnings


def validate_xml(xml_str: str) -> tuple[bool, str]:
    """Interface rétro-compatible avec l'ancienne validate_xml() de finetune_lora_xml.py."""
    r = validate_bt(xml_str)
    return r.valid, r.summary()


# ─── Post-traitement : injection des ports blackboard ──────────────────────────

# Valeurs par défaut des ports (blackboard variables ou littéraux)
_DEFAULT_PORT_VALUES: dict[str, dict[str, str]] = {
    "LoadMission": {"mission_file_path": "{mission_file_path}"},
    "UpdateCurrentGeneratedActivity": {
        "type": "{type}",
        "origin_sph": "{origin_sph}",
        "target_sph": "{target_sph}",
        "forbidden_atoms_out": "{forbidden_atoms}",
    },
    "ProjectPointOnNetwork": {"point_in": "{point_in}", "point_out": "{point_out}"},
    "CreatePath": {
        "origin": "{origin}",
        "target": "{target}",
        "forbidden_atoms": "{forbidden_atoms}",
        "path": "{path}",
    },
    "AgregatePath": {"path": "{path}"},
    "MissionFullyTreated": {"type": "{type}"},
    "PassAdvancedPath": {"adv_path": "{adv_path}"},
    "PassMission": {"mission": "{mission}"},
    "GenerateMissionSequence": {
        "mission": "{mission}",
        "mission_sequence": "{mission_sequence}",
    },
    "GenerateCorrectiveSubSequence": {"defects": "{defects}"},
    "CheckCurrentStepType": {"type_to_be_checked": "0"},
    "PassMotionParameters": {"motion_params": "{motion_params}"},
    "Move": {"threshold_type": "1", "motion_params": "{motion_params}"},
    "Deccelerate": {"motion_params": "{motion_params}"},
    "MoveAndStop": {"motion_params": "{motion_params}"},
    "SignalAndWaitForOrder": {"message": "waiting_for_order"},
    "IsRobotPoseProjectionActive": {
        "adv_path": "{adv_path}",
        "pub_proj": "{pub_proj}",
    },
    "Pause": {"duration": "2.0"},
    "PassDefectsLocalization": {"defects": "{defects}"},
}


def enrich_ports(xml_str: str) -> str:
    """Injecte les ports blackboard par défaut sur les nœuds qui n'en ont pas.

    Accepte les deux formats :
      v4 (proxy)  : <MoveAndStop name="stop"/>  → ajoute motion_params="{motion_params}"
      v5 (BTCPP)  : <Action ID="MoveAndStop"/>  → idem

    Ne modifie pas les ports déjà présents.
    Retourne le XML enrichi (ou l'original si le parsing échoue).
    """
    xml_clean = xml_str.strip()
    if xml_clean.startswith("<?xml"):
        xml_clean = xml_clean[xml_clean.index("?>") + 2 :].strip()
    try:
        root = ET.fromstring(xml_clean)
    except ET.ParseError:
        return xml_str  # XML invalide → retourner tel quel

    modified = False
    for elem in root.iter():
        # Résoudre le skill_id selon le format
        if elem.tag in ("Action", "Condition"):
            skill_id = elem.get("ID", "")
        elif elem.tag in VALID_SKILLS:
            skill_id = elem.tag
        else:
            continue

        defaults = _DEFAULT_PORT_VALUES.get(skill_id)
        if not defaults:
            continue

        for port, default_val in defaults.items():
            if port not in elem.attrib:
                elem.set(port, default_val)
                modified = True

    if not modified:
        return xml_str

    # Sérialiser avec indentation propre
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


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
