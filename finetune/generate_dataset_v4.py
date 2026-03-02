"""
Générateur de dataset NAV4RAIL v4 : (prompt mission, BehaviorTree XML)
Utilise les **27 skills réels** organisés en 4 familles.

Format XML : BehaviorTree.CPP v4 (BTCPP_format="4"), format aplati (pas de SubTreePlus).
Nœuds de contrôle : Sequence, Fallback
Skills réels :
  PREPARATION : LoadMission, MissionStructureValid, UpdateCurrentGeneratedActivity,
                ProjectPointOnNetwork, CreatePath, AgregatePath, MissionFullyTreated,
                PassAdvancedPath, PassMission, GenerateMissionSequence,
                GenerateCorrectiveSubSequence, InsertCorrectiveSubSequence
  MOTION      : MissionTerminated, CheckCurrentStepType, PassMotionParameters,
                Move, UpdateCurrentExecutedStep, Deccelerate, MoveAndStop,
                SignalAndWaitForOrder, IsRobotPoseProjectionActive
  INSPECTION  : ManageMeasurements, AnalyseMeasurements, MeasurementsQualityValidated,
                PassDefectsLocalization, MeasurementsEnforcedValidated
  SIMULATION  : SimulationStarted

Catégories (550 exemples) :
  1. Navigation simple            (100 ex.) — prep + path + motion loop
  2. Navigation avec autorisation  ( 50 ex.) — prep + signal + motion
  3. Inspection de voie           (100 ex.) — prep + motion + inspection + qualité
  4. Inspection avec corrective    ( 50 ex.) — inspection + défaut → corrective
  5. Mesures simples               ( 50 ex.) — prep + mesure + analyse
  6. Navigation sécurisée          ( 50 ex.) — avec décélération / arrêt
  7. Missions complexes           (100 ex.) — patterns combinés
  8. Simulation                    ( 50 ex.) — avec SimulationStarted

Basé sur l'analyse de behavior_tree_example.xml (BT réel NAV4RAIL).
"""

import json
import random
from pathlib import Path

random.seed(42)

# ─── Catalogue skills réels NAV4RAIL ────────────────────────────────────────

SKILLS_DOC = """Skills disponibles (27 skills, 4 familles) :

PREPARATION :
- LoadMission                    : Charge les paramètres de la mission depuis la source
- MissionStructureValid          : Vérifie la cohérence structurelle de la mission chargée
- UpdateCurrentGeneratedActivity : Met à jour l'activité en cours de génération
- ProjectPointOnNetwork          : Projette un point sur le réseau ferroviaire
- CreatePath                     : Calcule un chemin entre deux points
- AgregatePath                   : Fusionne plusieurs segments de chemin
- MissionFullyTreated            : Vérifie si toutes les étapes sont traitées
- PassAdvancedPath               : Transmet un chemin avancé au module d'exécution
- PassMission                    : Transmet la mission au module d'exécution
- GenerateMissionSequence        : Génère la séquence d'actions pour la mission
- GenerateCorrectiveSubSequence  : Génère une sous-séquence corrective en cas de déviation
- InsertCorrectiveSubSequence    : Insère la sous-séquence corrective dans la séquence

MOTION :
- MissionTerminated              : Vérifie si la mission est terminée (critère d'arrêt)
- CheckCurrentStepType           : Vérifie le type de l'étape en cours
- PassMotionParameters           : Configure les paramètres de mouvement
- Move                           : Déplace le robot vers la cible
- UpdateCurrentExecutedStep      : Marque l'étape courante comme exécutée
- Deccelerate                    : Réduit la vitesse du robot
- MoveAndStop                    : Déplace puis stoppe le robot à la cible
- SignalAndWaitForOrder           : Émet un signal et attend une autorisation externe
- IsRobotPoseProjectionActive    : Vérifie si la projection de pose est active

INSPECTION :
- ManageMeasurements             : Lance et gère l'acquisition des mesures
- AnalyseMeasurements            : Traite et analyse les données de mesure
- MeasurementsQualityValidated   : Vérifie si la qualité des mesures est acceptable
- PassDefectsLocalization        : Transmet la localisation des défauts détectés
- MeasurementsEnforcedValidated  : Validation stricte de la qualité des mesures

SIMULATION :
- SimulationStarted              : Vérifie si le mode simulation est actif"""

SYSTEM_PROMPT = (
    "Tu es un expert en robotique ferroviaire NAV4RAIL. "
    "Génère un Behavior Tree au format XML BehaviorTree.CPP v4 "
    "correspondant exactement à la mission décrite. "
    "Utilise uniquement les skills du catalogue fourni. "
    "Réponds uniquement avec le XML, sans explication."
)

# ─── Builder XML ─────────────────────────────────────────────────────────────

def N(tag: str, name: str, *children) -> dict:
    """Crée un nœud BT (action ou nœud de contrôle)."""
    d = {"tag": tag, "name": name}
    if children:
        d["children"] = list(children)
    return d


def A(skill: str, nm: str) -> dict:
    """Crée un nœud action (skill feuille)."""
    return N(skill, nm)


def S(nm: str, *ch) -> dict:
    """Crée un nœud Sequence."""
    return N("Sequence", nm, *ch)


def F(nm: str, *ch) -> dict:
    """Crée un nœud Fallback."""
    return N("Fallback", nm, *ch)


def render(node: dict, depth: int = 0) -> str:
    """Rend récursivement un nœud en XML indenté (2 espaces par niveau)."""
    pad = "  " * depth
    tag = node["tag"]
    name = node.get("name", "")
    attrs = f' name="{name}"' if name else ""
    children = node.get("children")

    if not children:
        return f"{pad}<{tag}{attrs}/>"

    lines = [f"{pad}<{tag}{attrs}>"]
    for child in children:
        lines.append(render(child, depth + 1))
    lines.append(f"{pad}</{tag}>")
    return "\n".join(lines)


def bt(tree: dict) -> str:
    """Enveloppe un arbre dans <root><BehaviorTree>...</BehaviorTree></root>."""
    inner = render(tree, depth=2)
    return (
        '<root BTCPP_format="4">\n'
        '  <BehaviorTree ID="MainTree">\n'
        f'{inner}\n'
        '  </BehaviorTree>\n'
        '</root>'
    )


def make_entry(mission: str, xml: str) -> dict:
    instruction = f"{SYSTEM_PROMPT}\n\n{SKILLS_DOC}\n\nMission : {mission}"
    return {
        "mission": mission,
        "xml": xml,
        "prompt": f"<s>[INST] {instruction} [/INST] {xml} </s>",
    }


# ─── Blocs réutilisables (patterns du BT réel) ─────────────────────────────

def prep_base():
    """Phase de préparation commune : charge + valide + génère séquence."""
    return [
        A("LoadMission", "load_mission"),
        A("MissionStructureValid", "check_structure"),
        A("GenerateMissionSequence", "generate_sequence"),
    ]


def prep_with_path():
    """Préparation complète avec calcul de chemin."""
    return [
        A("LoadMission", "load_mission"),
        A("MissionStructureValid", "check_structure"),
        A("UpdateCurrentGeneratedActivity", "update_activity"),
        A("ProjectPointOnNetwork", "project_origin"),
        A("ProjectPointOnNetwork", "project_target"),
        A("CreatePath", "create_path"),
        A("AgregatePath", "agregate_path"),
        A("PassAdvancedPath", "pass_path"),
        A("PassMission", "pass_mission"),
        A("GenerateMissionSequence", "generate_sequence"),
    ]


def prep_short():
    """Préparation minimale sans calcul de chemin."""
    return [
        A("LoadMission", "load_mission"),
        A("MissionStructureValid", "check_structure"),
        A("PassMission", "pass_mission"),
        A("GenerateMissionSequence", "generate_sequence"),
    ]


def motion_loop():
    """Boucle de navigation : Fallback(MissionTerminated | step)."""
    return F("execution_loop",
        A("MissionTerminated", "check_terminated"),
        S("step_execution",
            A("CheckCurrentStepType", "check_step_type"),
            A("PassMotionParameters", "set_motion_params"),
            A("Move", "execute_move"),
            A("UpdateCurrentExecutedStep", "update_step"),
        ),
    )


def motion_loop_with_decel():
    """Boucle navigation avec décélération en fin d'étape."""
    return F("execution_loop",
        A("MissionTerminated", "check_terminated"),
        S("step_execution",
            A("CheckCurrentStepType", "check_step_type"),
            A("PassMotionParameters", "set_motion_params"),
            A("Move", "execute_move"),
            A("Deccelerate", "decelerate"),
            A("UpdateCurrentExecutedStep", "update_step"),
        ),
    )


def mission_loop():
    """Boucle mission complète : Fallback(MissionFullyTreated | step)."""
    return F("mission_loop",
        A("MissionFullyTreated", "check_fully_treated"),
        S("step_execution",
            A("CheckCurrentStepType", "check_step_type"),
            A("PassMotionParameters", "set_motion_params"),
            A("Move", "execute_move"),
            A("UpdateCurrentExecutedStep", "update_step"),
        ),
    )


def quality_check():
    """Vérification qualité mesures : Fallback(QualityOK | report défauts)."""
    return F("quality_check",
        A("MeasurementsQualityValidated", "check_quality"),
        A("PassDefectsLocalization", "report_defects"),
    )


def quality_check_enforced():
    """Vérification qualité stricte avec re-mesure."""
    return F("quality_check",
        A("MeasurementsQualityValidated", "check_quality"),
        S("reacquire",
            A("ManageMeasurements", "retry_measurements"),
            A("MeasurementsEnforcedValidated", "enforce_quality"),
        ),
    )


def corrective_block():
    """Bloc correctif : génère + insère une sous-séquence corrective."""
    return S("corrective",
        A("GenerateCorrectiveSubSequence", "generate_corrective"),
        A("InsertCorrectiveSubSequence", "insert_corrective"),
    )


def inspection_step():
    """Étape d'inspection : mesure + analyse + qualité."""
    return S("inspection_step",
        A("ManageMeasurements", "acquire_measurements"),
        A("AnalyseMeasurements", "analyse_measurements"),
        quality_check(),
    )


def inspection_step_enforced():
    """Étape d'inspection avec validation stricte."""
    return S("inspection_step",
        A("ManageMeasurements", "acquire_measurements"),
        A("AnalyseMeasurements", "analyse_measurements"),
        quality_check_enforced(),
    )


# ─── Templates BT par catégorie ─────────────────────────────────────────────

# === 1. NAVIGATION SIMPLE ===

def xml_nav_simple() -> str:
    """Navigation : prep + chemin + boucle motion + stop."""
    return bt(S("navigation_sequence",
        *prep_with_path(),
        motion_loop(),
        A("MoveAndStop", "final_stop"),
    ))


def xml_nav_short() -> str:
    """Navigation courte : prep minimale + motion directe."""
    return bt(S("navigation_sequence",
        *prep_short(),
        A("PassMotionParameters", "set_motion_params"),
        A("Move", "move_to_target"),
        A("MoveAndStop", "final_stop"),
    ))


def xml_nav_with_decel() -> str:
    """Navigation avec décélération dans la boucle."""
    return bt(S("navigation_sequence",
        *prep_with_path(),
        motion_loop_with_decel(),
        A("MoveAndStop", "final_stop"),
    ))


def xml_nav_direct() -> str:
    """Navigation directe sans boucle (trajet simple)."""
    return bt(S("navigation_sequence",
        *prep_base(),
        A("ProjectPointOnNetwork", "project_target"),
        A("CreatePath", "create_path"),
        A("PassMotionParameters", "set_motion_params"),
        A("Move", "move_to_target"),
        A("MoveAndStop", "final_stop"),
    ))


def xml_nav_return() -> str:
    """Retour au dépôt : prep + chemin retour + motion."""
    return bt(S("return_sequence",
        *prep_base(),
        A("ProjectPointOnNetwork", "project_origin"),
        A("ProjectPointOnNetwork", "project_depot"),
        A("CreatePath", "create_return_path"),
        A("AgregatePath", "agregate_path"),
        A("PassAdvancedPath", "pass_path"),
        A("PassMotionParameters", "set_motion_params"),
        A("Move", "return_move"),
        A("Deccelerate", "decelerate"),
        A("MoveAndStop", "final_stop"),
    ))


# === 2. NAVIGATION AVEC AUTORISATION ===

def xml_nav_signal() -> str:
    """Navigation avec autorisation : prep + signal + motion."""
    return bt(S("authorized_navigation",
        *prep_with_path(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "wait_authorization"),
        motion_loop(),
        A("MoveAndStop", "final_stop"),
    ))


def xml_nav_signal_short() -> str:
    """Navigation courte avec autorisation."""
    return bt(S("authorized_navigation",
        *prep_base(),
        A("ProjectPointOnNetwork", "project_target"),
        A("CreatePath", "create_path"),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "wait_authorization"),
        A("PassMotionParameters", "set_motion_params"),
        A("Move", "move_to_target"),
        A("MoveAndStop", "final_stop"),
    ))


def xml_nav_signal_decel() -> str:
    """Navigation avec autorisation et décélération."""
    return bt(S("authorized_navigation",
        *prep_with_path(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "wait_authorization"),
        motion_loop_with_decel(),
        A("MoveAndStop", "final_stop"),
    ))


# === 3. INSPECTION DE VOIE ===

def xml_inspect_simple() -> str:
    """Inspection basique : prep + motion + mesure + analyse + qualité."""
    return bt(S("inspection_sequence",
        *prep_with_path(),
        F("inspection_loop",
            A("MissionFullyTreated", "check_complete"),
            S("inspection_step",
                A("Move", "move_to_zone"),
                A("Deccelerate", "slow_down"),
                A("ManageMeasurements", "acquire_measurements"),
                A("AnalyseMeasurements", "analyse_measurements"),
                quality_check(),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
    ))


def xml_inspect_enforced() -> str:
    """Inspection avec validation stricte des mesures."""
    return bt(S("inspection_sequence",
        *prep_with_path(),
        F("inspection_loop",
            A("MissionFullyTreated", "check_complete"),
            S("inspection_step",
                A("Move", "move_to_zone"),
                A("Deccelerate", "slow_down"),
                A("ManageMeasurements", "acquire_measurements"),
                A("AnalyseMeasurements", "analyse_measurements"),
                quality_check_enforced(),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
    ))


def xml_inspect_with_motion_loop() -> str:
    """Inspection avec boucle motion puis phase inspection."""
    return bt(S("inspection_sequence",
        *prep_with_path(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "wait_authorization"),
        F("execution_loop",
            A("MissionTerminated", "check_terminated"),
            S("step_execution",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("Move", "execute_move"),
                A("ManageMeasurements", "acquire_measurements"),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
    ))


def xml_inspect_multi_zone() -> str:
    """Inspection multi-zones : prep + boucle (motion + inspection)."""
    return bt(S("inspection_sequence",
        *prep_with_path(),
        F("inspection_loop",
            A("MissionFullyTreated", "check_complete"),
            S("zone_step",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("Move", "move_to_zone"),
                A("Deccelerate", "slow_for_inspection"),
                A("ManageMeasurements", "acquire_measurements"),
                A("AnalyseMeasurements", "analyse_measurements"),
                quality_check(),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
    ))


# === 4. INSPECTION AVEC CORRECTIVE ===

def xml_inspect_corrective() -> str:
    """Inspection avec sous-séquence corrective si défaut détecté."""
    return bt(S("inspection_sequence",
        *prep_with_path(),
        F("inspection_loop",
            A("MissionFullyTreated", "check_complete"),
            S("inspection_step",
                A("Move", "move_to_zone"),
                A("Deccelerate", "slow_down"),
                A("ManageMeasurements", "acquire_measurements"),
                A("AnalyseMeasurements", "analyse_measurements"),
                F("quality_check",
                    A("MeasurementsQualityValidated", "check_quality"),
                    S("handle_defects",
                        A("PassDefectsLocalization", "report_defects"),
                        A("GenerateCorrectiveSubSequence", "generate_corrective"),
                        A("InsertCorrectiveSubSequence", "insert_corrective"),
                    ),
                ),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
    ))


def xml_inspect_corrective_signal() -> str:
    """Inspection corrective avec signal après correction."""
    return bt(S("inspection_sequence",
        *prep_with_path(),
        F("inspection_loop",
            A("MissionFullyTreated", "check_complete"),
            S("inspection_step",
                A("Move", "move_to_zone"),
                A("ManageMeasurements", "acquire_measurements"),
                A("AnalyseMeasurements", "analyse_measurements"),
                F("quality_check",
                    A("MeasurementsQualityValidated", "check_quality"),
                    S("handle_defects",
                        A("PassDefectsLocalization", "report_defects"),
                        A("GenerateCorrectiveSubSequence", "generate_corrective"),
                        A("InsertCorrectiveSubSequence", "insert_corrective"),
                        A("SignalAndWaitForOrder", "wait_validation"),
                    ),
                ),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
    ))


def xml_inspect_corrective_enforced() -> str:
    """Inspection corrective avec validation stricte."""
    return bt(S("inspection_sequence",
        *prep_with_path(),
        F("inspection_loop",
            A("MissionFullyTreated", "check_complete"),
            S("inspection_step",
                A("Move", "move_to_zone"),
                A("Deccelerate", "slow_down"),
                A("ManageMeasurements", "acquire_measurements"),
                A("AnalyseMeasurements", "analyse_measurements"),
                F("quality_check",
                    A("MeasurementsQualityValidated", "check_quality"),
                    S("reacquire",
                        A("ManageMeasurements", "retry_measurements"),
                        A("MeasurementsEnforcedValidated", "enforce_quality"),
                    ),
                ),
                F("defect_handling",
                    A("MeasurementsQualityValidated", "final_quality_check"),
                    S("corrective",
                        A("PassDefectsLocalization", "report_defects"),
                        A("GenerateCorrectiveSubSequence", "generate_corrective"),
                        A("InsertCorrectiveSubSequence", "insert_corrective"),
                    ),
                ),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
    ))


# === 5. MESURES SIMPLES ===

def xml_measure_simple() -> str:
    """Mesure simple : prep + déplacement + mesure + analyse."""
    return bt(S("measurement_sequence",
        *prep_base(),
        A("ProjectPointOnNetwork", "project_measurement_point"),
        A("CreatePath", "create_path"),
        A("PassMotionParameters", "set_motion_params"),
        A("Move", "move_to_measurement_point"),
        A("Deccelerate", "slow_for_measurement"),
        A("ManageMeasurements", "acquire_measurements"),
        A("AnalyseMeasurements", "analyse_measurements"),
        A("MoveAndStop", "final_stop"),
    ))


def xml_measure_with_quality() -> str:
    """Mesure avec vérification qualité."""
    return bt(S("measurement_sequence",
        *prep_base(),
        A("ProjectPointOnNetwork", "project_measurement_point"),
        A("CreatePath", "create_path"),
        A("PassMotionParameters", "set_motion_params"),
        A("Move", "move_to_measurement_point"),
        A("Deccelerate", "slow_for_measurement"),
        A("ManageMeasurements", "acquire_measurements"),
        A("AnalyseMeasurements", "analyse_measurements"),
        quality_check(),
        A("MoveAndStop", "final_stop"),
    ))


def xml_measure_multi_point() -> str:
    """Mesures multi-points : boucle sur les points de mesure."""
    return bt(S("measurement_sequence",
        *prep_with_path(),
        F("measurement_loop",
            A("MissionFullyTreated", "check_complete"),
            S("measurement_step",
                A("PassMotionParameters", "set_motion_params"),
                A("Move", "move_to_point"),
                A("Deccelerate", "slow_for_measurement"),
                A("ManageMeasurements", "acquire_measurements"),
                A("AnalyseMeasurements", "analyse_measurements"),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
    ))


def xml_measure_and_report() -> str:
    """Mesure avec rapport de défauts."""
    return bt(S("measurement_sequence",
        *prep_base(),
        A("ProjectPointOnNetwork", "project_measurement_point"),
        A("CreatePath", "create_path"),
        A("PassMotionParameters", "set_motion_params"),
        A("Move", "move_to_measurement_point"),
        A("Deccelerate", "slow_for_measurement"),
        A("ManageMeasurements", "acquire_measurements"),
        A("AnalyseMeasurements", "analyse_measurements"),
        quality_check(),
        A("PassDefectsLocalization", "report_defects"),
        A("MoveAndStop", "final_stop"),
    ))


# === 6. NAVIGATION SECURISÉE ===

def xml_nav_safe_decel() -> str:
    """Navigation avec décélération progressive et arrêt."""
    return bt(S("safe_navigation",
        *prep_with_path(),
        F("execution_loop",
            A("MissionTerminated", "check_terminated"),
            S("step_execution",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("Move", "execute_move"),
                A("Deccelerate", "decelerate"),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("Deccelerate", "final_deceleration"),
        A("MoveAndStop", "final_stop"),
    ))


def xml_nav_safe_signal() -> str:
    """Navigation sécurisée avec arrêt et signal à chaque étape."""
    return bt(S("safe_navigation",
        *prep_with_path(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "initial_authorization"),
        F("execution_loop",
            A("MissionTerminated", "check_terminated"),
            S("step_execution",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("Move", "execute_move"),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("Deccelerate", "final_deceleration"),
        A("MoveAndStop", "final_stop"),
        A("SignalAndWaitForOrder", "signal_complete"),
    ))


def xml_nav_safe_stop() -> str:
    """Navigation sécurisée avec MoveAndStop par segment."""
    return bt(S("safe_navigation",
        *prep_with_path(),
        F("execution_loop",
            A("MissionTerminated", "check_terminated"),
            S("step_execution",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("MoveAndStop", "move_and_stop"),
                A("SignalAndWaitForOrder", "wait_next_order"),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
    ))


# === 7. MISSIONS COMPLEXES ===

def xml_complex_inspect_return() -> str:
    """Inspection complète puis retour au dépôt."""
    return bt(S("inspect_and_return",
        *prep_with_path(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "wait_authorization"),
        F("inspection_loop",
            A("MissionFullyTreated", "check_complete"),
            S("inspection_step",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("Move", "move_to_zone"),
                A("Deccelerate", "slow_down"),
                A("ManageMeasurements", "acquire_measurements"),
                A("AnalyseMeasurements", "analyse_measurements"),
                quality_check(),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("ProjectPointOnNetwork", "project_depot"),
        A("CreatePath", "create_return_path"),
        A("PassMotionParameters", "set_return_params"),
        A("Move", "return_to_depot"),
        A("MoveAndStop", "final_stop"),
    ))


def xml_complex_corrective_return() -> str:
    """Inspection corrective puis retour."""
    return bt(S("corrective_and_return",
        *prep_with_path(),
        F("inspection_loop",
            A("MissionFullyTreated", "check_complete"),
            S("inspection_step",
                A("Move", "move_to_zone"),
                A("Deccelerate", "slow_down"),
                A("ManageMeasurements", "acquire_measurements"),
                A("AnalyseMeasurements", "analyse_measurements"),
                F("quality_check",
                    A("MeasurementsQualityValidated", "check_quality"),
                    S("handle_defects",
                        A("PassDefectsLocalization", "report_defects"),
                        A("GenerateCorrectiveSubSequence", "generate_corrective"),
                        A("InsertCorrectiveSubSequence", "insert_corrective"),
                    ),
                ),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("ProjectPointOnNetwork", "project_depot"),
        A("CreatePath", "create_return_path"),
        A("PassMotionParameters", "set_return_params"),
        A("Move", "return_to_depot"),
        A("MoveAndStop", "final_stop"),
    ))


def xml_complex_full_inspection() -> str:
    """Inspection complète : autorisation + motion + inspection + corrective."""
    return bt(S("full_inspection",
        *prep_with_path(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "wait_authorization"),
        F("inspection_loop",
            A("MissionFullyTreated", "check_complete"),
            S("inspection_step",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("Move", "move_to_zone"),
                A("Deccelerate", "slow_down"),
                A("ManageMeasurements", "acquire_measurements"),
                A("AnalyseMeasurements", "analyse_measurements"),
                F("quality_check",
                    A("MeasurementsQualityValidated", "check_quality"),
                    S("handle_defects",
                        A("PassDefectsLocalization", "report_defects"),
                        A("GenerateCorrectiveSubSequence", "generate_corrective"),
                        A("InsertCorrectiveSubSequence", "insert_corrective"),
                        A("SignalAndWaitForOrder", "wait_corrective_validation"),
                    ),
                ),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
        A("SignalAndWaitForOrder", "signal_mission_complete"),
    ))


def xml_complex_nav_inspect() -> str:
    """Navigation puis inspection sur zone ciblée."""
    return bt(S("nav_then_inspect",
        *prep_with_path(),
        F("navigation_loop",
            A("MissionTerminated", "check_nav_terminated"),
            S("nav_step",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("Move", "execute_move"),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("Deccelerate", "decelerate_for_inspection"),
        A("ManageMeasurements", "acquire_measurements"),
        A("AnalyseMeasurements", "analyse_measurements"),
        quality_check(),
        A("MoveAndStop", "final_stop"),
    ))


def xml_complex_patrol() -> str:
    """Patrouille : boucle motion + inspection multi-points."""
    return bt(S("patrol_sequence",
        *prep_with_path(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "wait_authorization"),
        F("patrol_loop",
            A("MissionFullyTreated", "check_complete"),
            S("patrol_step",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("Move", "move_to_checkpoint"),
                A("ManageMeasurements", "inspect_checkpoint"),
                A("AnalyseMeasurements", "analyse_checkpoint"),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
        A("SignalAndWaitForOrder", "signal_patrol_complete"),
    ))


# === 8. SIMULATION ===

def xml_simulation_nav() -> str:
    """Navigation en mode simulation."""
    return bt(S("simulation_navigation",
        A("SimulationStarted", "check_simulation"),
        *prep_with_path(),
        motion_loop(),
        A("MoveAndStop", "final_stop"),
    ))


def xml_simulation_inspect() -> str:
    """Inspection en mode simulation."""
    return bt(S("simulation_inspection",
        A("SimulationStarted", "check_simulation"),
        *prep_with_path(),
        F("inspection_loop",
            A("MissionFullyTreated", "check_complete"),
            S("inspection_step",
                A("Move", "move_to_zone"),
                A("ManageMeasurements", "acquire_measurements"),
                A("AnalyseMeasurements", "analyse_measurements"),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
    ))


def xml_simulation_full() -> str:
    """Simulation complète avec corrective."""
    return bt(S("simulation_full",
        A("SimulationStarted", "check_simulation"),
        *prep_with_path(),
        F("inspection_loop",
            A("MissionFullyTreated", "check_complete"),
            S("inspection_step",
                A("Move", "move_to_zone"),
                A("Deccelerate", "slow_down"),
                A("ManageMeasurements", "acquire_measurements"),
                A("AnalyseMeasurements", "analyse_measurements"),
                F("quality_check",
                    A("MeasurementsQualityValidated", "check_quality"),
                    S("handle_defects",
                        A("PassDefectsLocalization", "report_defects"),
                        A("GenerateCorrectiveSubSequence", "generate_corrective"),
                        A("InsertCorrectiveSubSequence", "insert_corrective"),
                    ),
                ),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
    ))


# ─── Vocabulaire missions ────────────────────────────────────────────────────

NAV_VERBS = ["Déplace-toi", "Va", "Rejoins", "Navigue jusqu'à",
             "Retourne", "Avance jusqu'au", "Positionne-toi au"]
NAV_TARGETS = ["le dépôt principal", "le point de chargement",
               "la position de départ", "le poste de maintenance",
               "la zone de stationnement", "la voie de service",
               "le terminal de ravitaillement", "la zone de remisage",
               "le poste de contrôle central", "la voie d'évitement"]
URGENCY_TGTS = ["le secteur d'urgence", "la zone d'incident",
                "le point d'intervention prioritaire"]

INSPECT_OBJS = ["la voie", "les rails", "le tunnel ferroviaire",
                "le passage à niveau", "les aiguillages", "les traverses",
                "les soudures de rails", "la signalisation", "les capteurs de voie",
                "les fixations de rails", "la géométrie de la courbe",
                "les joints de dilatation", "les éléments de sécurité"]
INSPECT_VERBS = ["Inspecte", "Contrôle", "Vérifie", "Effectue une inspection de",
                 "Réalise un contrôle de", "Examine"]
SECTIONS = ["A", "B", "C", "D", "E", "nord", "sud", "est", "ouest",
            "principale", "secondaire", "maintenance", "critique"]

MEASURE_TYPES = ["la géométrie de voie", "le nivellement", "le dévers",
                 "la largeur de voie", "l'alignement des rails",
                 "les paramètres thermiques", "le profil de voie",
                 "les paramètres au point de contrôle", "l'usure des rails",
                 "la résistance des soudures", "la vibration de voie"]
MEASURE_VERBS = ["Mesure", "Effectue des mesures de", "Enregistre",
                 "Prends des mesures de", "Réalise une mesure de",
                 "Effectue un relevé de"]

AUTH_NAV_MISSIONS = [
    "Navigue vers {} avec autorisation préalable",
    "Déplace-toi vers {} après autorisation du poste de contrôle",
    "Rejoins {} en mode supervisé avec validation opérateur",
    "Va au {} et attends l'autorisation avant de démarrer",
    "Effectue un transit supervisé vers {}",
    "Navigue vers {} avec activation de la projection de pose",
    "Déplace-toi vers {} en mode autorisé avec décélération",
    "Rejoins {} après vérification de la projection et autorisation",
]

SAFE_NAV_MISSIONS = [
    "Navigue en mode sécurisé vers {}",
    "Déplace-toi vers {} avec décélération progressive",
    "Rejoins {} en mode sécurisé avec arrêt à chaque étape",
    "Effectue une navigation sécurisée jusqu'à {}",
    "Navigue vers {} en signalant ta progression",
    "Déplace-toi pas à pas vers {} avec autorisation à chaque segment",
    "Effectue un transit sécurisé vers {} avec arrêt et signal",
]

INSPECT_MISSIONS = [
    "Inspecte {} et vérifie la qualité des mesures",
    "Effectue une inspection complète de {} avec analyse des mesures",
    "Réalise un contrôle qualité de {} avec re-mesure si nécessaire",
    "Inspecte {} en boucle jusqu'à complétion de la mission",
    "Contrôle {} avec validation de la qualité des mesures",
    "Effectue l'inspection de {} avec analyse et rapport de qualité",
]

INSPECT_CORRECTIVE_MISSIONS = [
    "Inspecte {} et corrige automatiquement les défauts détectés",
    "Effectue une inspection de {} avec sous-séquence corrective si défaut",
    "Réalise un contrôle de {} : si défaut, génère et insère une correction",
    "Inspecte {} avec gestion corrective automatique des anomalies",
    "Contrôle {} et applique les corrections nécessaires si qualité insuffisante",
    "Effectue une inspection approfondie de {} avec correction et signalement des défauts",
]

COMPLEX_MISSIONS = [
    ("Inspecte {} et reviens au dépôt après complétion", xml_complex_inspect_return),
    ("Effectue une inspection complète de {} avec correction puis retour", xml_complex_corrective_return),
    ("Réalise l'inspection intégrale de {} : autorisation, mesures, corrections, signal final", xml_complex_full_inspection),
    ("Navigue vers {} puis inspecte la zone à l'arrivée", xml_complex_nav_inspect),
    ("Effectue une patrouille d'inspection entre {} et {} avec mesures à chaque point", xml_complex_patrol),
    ("Inspecte {} en mode supervisé avec rapport de défauts et retour au dépôt", xml_complex_inspect_return),
    ("Réalise une ronde de contrôle de {} avec analyse qualité et correction automatique", xml_complex_full_inspection),
    ("Patrouille entre {} et {} en inspectant chaque checkpoint", xml_complex_patrol),
    ("Navigue vers {}, mesure la qualité de la voie et reviens", xml_complex_nav_inspect),
    ("Inspection complète de {} avec autorisation, mesures, corrections et signal de fin", xml_complex_full_inspection),
    ("Inspecte {} avec re-mesure stricte et retour au point de départ", xml_complex_corrective_return),
    ("Contrôle post-travaux de {} : inspection, validation qualité, corrections, rapport", xml_complex_full_inspection),
]

SIMULATION_MISSIONS = [
    "Simule une navigation vers {}",
    "Lance une simulation de navigation jusqu'à {}",
    "En mode simulation, navigue vers {}",
    "Teste en simulation la navigation vers {}",
    "Simule un transit vers {} pour validation",
    "Simule une inspection de {}",
    "Lance une simulation d'inspection de {}",
    "En mode simulation, inspecte {}",
    "Teste en simulation l'inspection de {}",
    "Simule l'inspection complète de {} avec corrections",
    "Lance une simulation d'inspection approfondie de {}",
    "En mode simulation, effectue le contrôle complet de {}",
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def km() -> int:
    return random.randint(0, 99)

def km_pair() -> tuple:
    a = random.randint(0, 90)
    return a, a + random.randint(2, 15)

def section() -> str:
    return random.choice(SECTIONS)

def zone() -> str:
    return random.choice(["A", "B", "C", "nord", "sud", "maintenance",
                          "urgence", "test", "critique", str(km())])

def location() -> str:
    a, b = km_pair()
    return random.choice([
        f"entre le km {a} et le km {b}",
        f"de la section {section()}",
        f"au km {a}",
        f"dans la zone {zone()}",
        f"au point PK{a}",
        f"entre les km {a} et {b}",
        f"sur {b - a} km depuis le km {a}",
    ])

def target_named() -> str:
    return random.choice(NAV_TARGETS + URGENCY_TGTS)

def inspect_target() -> str:
    obj = random.choice(INSPECT_OBJS)
    loc = location()
    return f"{obj} {loc}"


# ─── Générateurs par catégorie ───────────────────────────────────────────────

def gen_navigation(n: int) -> list:
    """Navigation simple (100 ex.)."""
    templates = [xml_nav_simple, xml_nav_short, xml_nav_with_decel,
                 xml_nav_direct, xml_nav_return]
    examples = []

    for _ in range(n // 2):
        verb = random.choice(NAV_VERBS)
        a, b = km_pair()
        mission = random.choice([
            f"{verb} au km {b} depuis le km {a}",
            f"{verb} au km {b}",
            f"Rejoins le km {b}",
            f"Avance de {b - a} km depuis le km {a}",
        ])
        examples.append(make_entry(mission, random.choice(templates)()))

    for _ in range(n - n // 2):
        verb = random.choice(NAV_VERBS)
        target = target_named()
        mission = random.choice([
            f"{verb} {target}",
            f"Déplace-toi vers {target}",
            f"Rejoins {target}",
            f"Va à {target} et attends",
        ])
        examples.append(make_entry(mission, random.choice(templates)()))

    return examples


def gen_nav_authorized(n: int) -> list:
    """Navigation avec autorisation (50 ex.)."""
    templates = [xml_nav_signal, xml_nav_signal_short, xml_nav_signal_decel]
    examples = []

    for _ in range(n):
        tpl = random.choice(AUTH_NAV_MISSIONS)
        a, b = km_pair()
        target = random.choice([
            f"le km {b}",
            f"le km {b} depuis le km {a}",
            target_named(),
            f"la zone {zone()}",
        ])
        mission = tpl.format(target)
        examples.append(make_entry(mission, random.choice(templates)()))

    return examples


def gen_inspection(n: int) -> list:
    """Inspection de voie (100 ex.)."""
    templates = [xml_inspect_simple, xml_inspect_enforced,
                 xml_inspect_with_motion_loop, xml_inspect_multi_zone]
    examples = []

    for _ in range(n):
        tpl = random.choice(INSPECT_MISSIONS)
        target = inspect_target()
        mission = tpl.format(target)
        examples.append(make_entry(mission, random.choice(templates)()))

    return examples


def gen_inspection_corrective(n: int) -> list:
    """Inspection avec corrective (50 ex.)."""
    templates = [xml_inspect_corrective, xml_inspect_corrective_signal,
                 xml_inspect_corrective_enforced]
    examples = []

    for _ in range(n):
        tpl = random.choice(INSPECT_CORRECTIVE_MISSIONS)
        target = inspect_target()
        mission = tpl.format(target)
        examples.append(make_entry(mission, random.choice(templates)()))

    return examples


def gen_measurement(n: int) -> list:
    """Mesures simples (50 ex.)."""
    templates = [xml_measure_simple, xml_measure_with_quality,
                 xml_measure_multi_point, xml_measure_and_report]
    examples = []

    for _ in range(n):
        verb = random.choice(MEASURE_VERBS)
        mtype = random.choice(MEASURE_TYPES)
        loc = location()
        mission = random.choice([
            f"{verb} {mtype} {loc}",
            f"Enregistre {mtype} {loc}",
            f"Effectue un relevé de {mtype} {loc}",
        ])
        examples.append(make_entry(mission, random.choice(templates)()))

    return examples


def gen_safe_navigation(n: int) -> list:
    """Navigation sécurisée (50 ex.)."""
    templates = [xml_nav_safe_decel, xml_nav_safe_signal, xml_nav_safe_stop]
    examples = []

    for _ in range(n):
        tpl = random.choice(SAFE_NAV_MISSIONS)
        a, b = km_pair()
        target = random.choice([
            f"le km {b}",
            target_named(),
            f"la zone {zone()}",
        ])
        mission = tpl.format(target)
        examples.append(make_entry(mission, random.choice(templates)()))

    return examples


def gen_complex(n: int) -> list:
    """Missions complexes (100 ex.)."""
    examples = []

    for _ in range(n):
        tpl_str, xml_fn = random.choice(COMPLEX_MISSIONS)
        if tpl_str.count("{}") == 2:
            a, b = km_pair()
            mission = tpl_str.format(f"km {a}", f"km {b}")
        else:
            target = random.choice([
                inspect_target(),
                f"la section {section()}",
                f"la zone {zone()}",
            ])
            mission = tpl_str.format(target)
        examples.append(make_entry(mission, xml_fn()))

    return examples


def gen_simulation(n: int) -> list:
    """Simulation (50 ex.)."""
    sim_nav_templates = [xml_simulation_nav]
    sim_inspect_templates = [xml_simulation_inspect, xml_simulation_full]
    examples = []

    for i in range(n):
        tpl = random.choice(SIMULATION_MISSIONS)
        target = random.choice([
            f"le km {km()}",
            target_named(),
            inspect_target(),
            f"la zone {zone()}",
        ])
        mission = tpl.format(target)

        if "inspect" in tpl.lower() or "contrôle" in tpl.lower():
            xml = random.choice(sim_inspect_templates)()
        else:
            xml = random.choice(sim_nav_templates + sim_inspect_templates)()

        examples.append(make_entry(mission, xml))

    return examples


# ─── Assemblage & sauvegarde ─────────────────────────────────────────────────

def main():
    import xml.etree.ElementTree as ET
    output_dir = Path(__file__).parent
    out_jsonl = output_dir / "dataset_nav4rail_v4.jsonl"
    out_json = output_dir / "dataset_nav4rail_v4.json"

    counts = {
        "Navigation simple": 100,
        "Navigation autorisée": 50,
        "Inspection": 100,
        "Inspection corrective": 50,
        "Mesures": 50,
        "Navigation sécurisée": 50,
        "Complexe": 100,
        "Simulation": 50,
    }

    print("Génération du dataset NAV4RAIL v4 (27 skills réels)...")
    dataset = (
        gen_navigation(counts["Navigation simple"])
        + gen_nav_authorized(counts["Navigation autorisée"])
        + gen_inspection(counts["Inspection"])
        + gen_inspection_corrective(counts["Inspection corrective"])
        + gen_measurement(counts["Mesures"])
        + gen_safe_navigation(counts["Navigation sécurisée"])
        + gen_complex(counts["Complexe"])
        + gen_simulation(counts["Simulation"])
    )
    random.shuffle(dataset)

    # Validation XML
    errors = 0
    for i, entry in enumerate(dataset):
        try:
            ET.fromstring(entry["xml"])
        except ET.ParseError as e:
            print(f"  [ERREUR XML] exemple {i}: {e}")
            print(entry["xml"][:300])
            errors += 1

    # Vérification couverture des 27 skills
    all_skills = set()
    for entry in dataset:
        import re
        tags = re.findall(r'<(\w+)\s+name=', entry["xml"])
        all_skills.update(t for t in tags if t not in ("root", "BehaviorTree", "Sequence", "Fallback"))
    expected_skills = {
        "LoadMission", "MissionStructureValid", "UpdateCurrentGeneratedActivity",
        "ProjectPointOnNetwork", "CreatePath", "AgregatePath", "MissionFullyTreated",
        "PassAdvancedPath", "PassMission", "GenerateMissionSequence",
        "GenerateCorrectiveSubSequence", "InsertCorrectiveSubSequence",
        "MissionTerminated", "CheckCurrentStepType", "PassMotionParameters",
        "Move", "UpdateCurrentExecutedStep", "Deccelerate", "MoveAndStop",
        "SignalAndWaitForOrder", "IsRobotPoseProjectionActive",
        "ManageMeasurements", "AnalyseMeasurements", "MeasurementsQualityValidated",
        "PassDefectsLocalization", "MeasurementsEnforcedValidated",
        "SimulationStarted",
    }
    missing = expected_skills - all_skills
    extra = all_skills - expected_skills

    # Sauvegarde
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nDataset v4 : {len(dataset)} exemples")
    for cat, n in counts.items():
        print(f"  {cat:<25}: {n}")
    print(f"\nValidation XML : {'OK — 0 erreur' if errors == 0 else f'{errors} ERREURS'}")
    print(f"\nCouverture skills : {len(all_skills)}/{len(expected_skills)}")
    if missing:
        print(f"  Skills MANQUANTS : {missing}")
    if extra:
        print(f"  Skills INCONNUS  : {extra}")
    print(f"\n  → {out_jsonl}")
    print(f"  → {out_json}")

    # Exemple
    print("\n── Exemple XML ─────────────────────────────────────────────────────────")
    sample = next(e for e in dataset if "Fallback" in e["xml"] and "corrective" in e["xml"].lower())
    print(f"Mission : {sample['mission']}")
    print(sample["xml"])


if __name__ == "__main__":
    main()
