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

Catégories (2000 exemples) :
  1. Navigation simple            (350 ex.) — prep + path + motion loop
  2. Navigation avec autorisation (150 ex.) — prep + signal + motion
  3. Inspection de voie           (350 ex.) — prep + motion + inspection + qualité
  4. Inspection avec corrective   (200 ex.) — inspection + défaut → corrective
  5. Mesures simples              (150 ex.) — prep + mesure + analyse
  6. Navigation sécurisée         (150 ex.) — avec décélération / arrêt
  7. Missions complexes           (400 ex.) — patterns combinés
  8. Simulation                   (250 ex.) — avec SimulationStarted

Patterns fortement inspirés de behavior_tree_example.xml (BT réel NAV4RAIL).
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
    """Préparation complète avec calcul de chemin (pattern BT réel : get_mission + calculate_path + pass)."""
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


def prep_with_path_loop():
    """Préparation avec boucle de calcul de chemin (pattern réel calculate_path subtree).

    BT réel (lignes 31-44) :
      Fallback(
        Repeat(UpdateActivity → Project × 2 → CreatePath → AgregatePath)
        MissionFullyTreated
      )
    """
    return [
        A("LoadMission", "load_mission"),
        A("MissionStructureValid", "check_structure"),
        F("path_calculation",
            S("path_loop",
                A("UpdateCurrentGeneratedActivity", "update_activity"),
                A("ProjectPointOnNetwork", "project_origin"),
                A("ProjectPointOnNetwork", "project_target"),
                A("CreatePath", "create_path"),
                A("AgregatePath", "agregate_path"),
            ),
            A("MissionFullyTreated", "all_paths_calculated"),
        ),
        A("PassAdvancedPath", "pass_path"),
        A("PassMission", "pass_mission"),
        A("GenerateMissionSequence", "generate_sequence"),
    ]


def prep_with_validation():
    """Préparation avec double validation (structure + projection)."""
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
        A("IsRobotPoseProjectionActive", "check_projection"),
    ]


def motion_loop():
    """Boucle de navigation : Fallback(MissionTerminated | step).
    Pattern BT réel execute subtree (lignes 45-62)."""
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
    """Boucle navigation avec décélération en fin d'étape.
    Pattern BT réel deccelerate subtree (lignes 82-89)."""
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


def motion_loop_reach_and_stop():
    """Boucle navigation avec MoveAndStop et signal à chaque étape.
    Pattern BT réel reach_and_stop subtree (lignes 98-106)."""
    return F("execution_loop",
        A("MissionTerminated", "check_terminated"),
        S("step_execution",
            A("CheckCurrentStepType", "check_step_type"),
            A("PassMotionParameters", "set_motion_params"),
            A("MoveAndStop", "reach_and_stop"),
            A("SignalAndWaitForOrder", "wait_authorization"),
            A("UpdateCurrentExecutedStep", "update_step"),
        ),
    )


def motion_loop_move_and_inspect():
    """Boucle motion avec inspection pendant le mouvement.
    Pattern BT réel move_and_inspect subtree (lignes 72-81)."""
    return F("execution_loop",
        A("MissionTerminated", "check_terminated"),
        S("step_execution",
            A("CheckCurrentStepType", "check_step_type"),
            A("PassMotionParameters", "set_motion_params"),
            A("ManageMeasurements", "start_inspection"),
            A("Move", "execute_move"),
            A("UpdateCurrentExecutedStep", "update_step"),
        ),
    )


def motion_loop_reach_stop_inspect():
    """Boucle motion avec arrêt et inspection post-mouvement.
    Pattern BT réel reach_and_stop_inspecting subtree (lignes 107-124)."""
    return F("execution_loop",
        A("MissionTerminated", "check_terminated"),
        S("step_execution",
            A("CheckCurrentStepType", "check_step_type"),
            A("PassMotionParameters", "set_motion_params"),
            A("MoveAndStop", "reach_and_stop"),
            A("ManageMeasurements", "stop_inspection"),
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
    )


def motion_loop_pass_stop_inspect():
    """Boucle motion pass-through avec arrêt inspection et validation stricte.
    Pattern BT réel pass_and_stop_inspecting subtree (lignes 133-145)."""
    return F("execution_loop",
        A("MissionTerminated", "check_terminated"),
        S("step_execution",
            A("CheckCurrentStepType", "check_step_type"),
            A("PassMotionParameters", "set_motion_params"),
            A("Move", "pass_through"),
            A("ManageMeasurements", "stop_inspection"),
            F("enforced_analysis",
                A("AnalyseMeasurements", "analyse_measurements"),
                A("MeasurementsEnforcedValidated", "enforced_validation"),
            ),
            A("UpdateCurrentExecutedStep", "update_step"),
        ),
    )


def motion_multi_selector():
    """Sélecteur multi-motion : Fallback entre différents types de mouvement.
    Pattern BT réel execute subtree (lignes 46-62) — version aplatie."""
    return F("execution_loop",
        A("MissionTerminated", "check_terminated"),
        F("motion_selector",
            S("move_step",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("Move", "execute_move"),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
            S("decelerate_step",
                A("CheckCurrentStepType", "check_step_decel"),
                A("PassMotionParameters", "set_decel_params"),
                A("Deccelerate", "decelerate"),
                A("UpdateCurrentExecutedStep", "update_step_decel"),
            ),
            S("reach_and_stop_step",
                A("CheckCurrentStepType", "check_step_stop"),
                A("PassMotionParameters", "set_stop_params"),
                A("MoveAndStop", "reach_and_stop"),
                A("SignalAndWaitForOrder", "wait_next_order"),
                A("UpdateCurrentExecutedStep", "update_step_stop"),
            ),
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


def mission_loop_with_decel():
    """Boucle mission avec décélération."""
    return F("mission_loop",
        A("MissionFullyTreated", "check_fully_treated"),
        S("step_execution",
            A("CheckCurrentStepType", "check_step_type"),
            A("PassMotionParameters", "set_motion_params"),
            A("Move", "execute_move"),
            A("Deccelerate", "decelerate"),
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
    """Vérification qualité stricte avec re-mesure.
    Pattern BT réel pass_and_stop_inspecting (lignes 139-142)."""
    return F("quality_check",
        A("MeasurementsQualityValidated", "check_quality"),
        S("reacquire",
            A("ManageMeasurements", "retry_measurements"),
            A("MeasurementsEnforcedValidated", "enforce_quality"),
        ),
    )


def enforced_analysis():
    """Analyse avec validation stricte.
    Pattern BT réel pass_and_stop_inspecting Fallback(Analyse | Enforced)."""
    return F("enforced_analysis",
        A("AnalyseMeasurements", "analyse_measurements"),
        A("MeasurementsEnforcedValidated", "enforced_validation"),
    )


def corrective_block():
    """Bloc correctif : génère + insère une sous-séquence corrective."""
    return S("corrective",
        A("GenerateCorrectiveSubSequence", "generate_corrective"),
        A("InsertCorrectiveSubSequence", "insert_corrective"),
    )


def corrective_full():
    """Bloc correctif complet : report + génère + insère.
    Pattern BT réel reach_stop_inspecting_dont_wait (lignes 161-167)."""
    return S("handle_defects",
        A("PassDefectsLocalization", "report_defects"),
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


def inspection_stop_and_analyse():
    """Arrête l'inspection et analyse les mesures.
    Pattern BT réel reach_and_stop_inspecting (lignes 112-121)."""
    return S("stop_and_analyse",
        A("ManageMeasurements", "stop_inspection"),
        A("AnalyseMeasurements", "analyse_measurements"),
        F("corrective_sequence",
            A("MeasurementsQualityValidated", "check_quality"),
            A("PassDefectsLocalization", "report_defects"),
        ),
        corrective_block(),
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


def xml_nav_path_loop() -> str:
    """Navigation avec boucle de calcul de chemin (pattern réel calculate_path)."""
    return bt(S("navigation_sequence",
        *prep_with_path_loop(),
        motion_loop(),
        A("MoveAndStop", "final_stop"),
    ))


def xml_nav_multi_motion() -> str:
    """Navigation avec sélecteur multi-motion (pattern réel execute)."""
    return bt(S("navigation_sequence",
        *prep_with_path(),
        motion_multi_selector(),
        A("MoveAndStop", "final_stop"),
    ))


def xml_nav_mission_loop() -> str:
    """Navigation avec boucle MissionFullyTreated."""
    return bt(S("navigation_sequence",
        *prep_with_path(),
        mission_loop(),
        A("MoveAndStop", "final_stop"),
    ))


def xml_nav_mission_loop_decel() -> str:
    """Navigation avec boucle MissionFullyTreated et décélération."""
    return bt(S("navigation_sequence",
        *prep_with_path(),
        mission_loop_with_decel(),
        A("MoveAndStop", "final_stop"),
    ))


def xml_nav_with_projection() -> str:
    """Navigation avec vérification de projection de pose."""
    return bt(S("navigation_sequence",
        *prep_with_validation(),
        motion_loop(),
        A("MoveAndStop", "final_stop"),
    ))


# === 2. NAVIGATION AVEC AUTORISATION ===

def xml_nav_signal() -> str:
    """Navigation avec autorisation : prep + signal + motion.
    Pattern BT réel real_preparation (lignes 9-14)."""
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


def xml_nav_signal_reach_stop() -> str:
    """Navigation autorisée avec reach-and-stop à chaque étape.
    Pattern BT réel reach_and_stop (lignes 98-106)."""
    return bt(S("authorized_navigation",
        *prep_with_path(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "wait_authorization"),
        motion_loop_reach_and_stop(),
        A("MoveAndStop", "final_stop"),
    ))


def xml_nav_signal_path_loop() -> str:
    """Navigation autorisée avec boucle de calcul de chemin."""
    return bt(S("authorized_navigation",
        *prep_with_path_loop(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "wait_authorization"),
        motion_loop(),
        A("MoveAndStop", "final_stop"),
    ))


def xml_nav_signal_multi_motion() -> str:
    """Navigation autorisée avec multi-motion selector."""
    return bt(S("authorized_navigation",
        *prep_with_path(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "wait_authorization"),
        motion_multi_selector(),
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


def xml_inspect_move_and_inspect() -> str:
    """Inspection en mouvement : démarrage de l'inspection avant le mouvement.
    Pattern BT réel move_and_inspect (lignes 72-81)."""
    return bt(S("inspection_sequence",
        *prep_with_path(),
        F("inspection_loop",
            A("MissionTerminated", "check_terminated"),
            S("move_and_inspect_step",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("ManageMeasurements", "start_inspection"),
                A("Move", "execute_move"),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
    ))


def xml_inspect_reach_stop_inspect() -> str:
    """Inspection avec arrêt : arrive, stoppe, puis inspecte et analyse.
    Pattern BT réel reach_and_stop_inspecting (lignes 107-124)."""
    return bt(S("inspection_sequence",
        *prep_with_path(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "wait_authorization"),
        F("inspection_loop",
            A("MissionTerminated", "check_terminated"),
            S("reach_stop_inspect_step",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("MoveAndStop", "reach_and_stop"),
                A("ManageMeasurements", "stop_inspection"),
                A("AnalyseMeasurements", "analyse_measurements"),
                F("corrective_sequence",
                    A("MeasurementsQualityValidated", "check_quality"),
                    A("PassDefectsLocalization", "report_defects"),
                ),
                corrective_block(),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
    ))


def xml_inspect_pass_stop_inspect() -> str:
    """Inspection pass-through avec arrêt et validation stricte.
    Pattern BT réel pass_and_stop_inspecting (lignes 133-145)."""
    return bt(S("inspection_sequence",
        *prep_with_path(),
        F("inspection_loop",
            A("MissionTerminated", "check_terminated"),
            S("pass_and_inspect_step",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("Move", "pass_through"),
                A("ManageMeasurements", "stop_inspection"),
                F("enforced_analysis",
                    A("AnalyseMeasurements", "analyse_measurements"),
                    A("MeasurementsEnforcedValidated", "enforced_validation"),
                ),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
    ))


def xml_inspect_path_loop() -> str:
    """Inspection avec boucle de calcul de chemin."""
    return bt(S("inspection_sequence",
        *prep_with_path_loop(),
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


def xml_inspect_reach_stop_no_wait() -> str:
    """Inspection reach-stop sans attente + analyse corrective complète.
    Pattern BT réel reach_stop_inspecting_dont_wait (lignes 154-171)."""
    return bt(S("inspection_sequence",
        *prep_with_path(),
        F("inspection_loop",
            A("MissionTerminated", "check_terminated"),
            S("reach_stop_inspect_step",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("MoveAndStop", "reach_and_stop"),
                A("ManageMeasurements", "stop_inspection"),
                A("AnalyseMeasurements", "analyse_measurements"),
                F("corrective_sequence",
                    A("MeasurementsQualityValidated", "check_quality"),
                    corrective_full(),
                ),
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


def xml_inspect_corrective_reach_stop() -> str:
    """Inspection corrective avec reach-and-stop.
    Pattern BT réel reach_and_stop_inspecting (lignes 107-124)."""
    return bt(S("inspection_sequence",
        *prep_with_path(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "wait_authorization"),
        F("inspection_loop",
            A("MissionTerminated", "check_terminated"),
            S("inspection_step",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("MoveAndStop", "reach_and_stop"),
                A("ManageMeasurements", "stop_inspection"),
                A("AnalyseMeasurements", "analyse_measurements"),
                F("corrective_sequence",
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


def xml_inspect_corrective_reach_stop_full() -> str:
    """Inspection corrective reach-stop avec séquence corrective complète.
    Pattern BT réel reach_stop_inspecting_dont_wait (lignes 154-171)."""
    return bt(S("inspection_sequence",
        *prep_with_path(),
        F("inspection_loop",
            A("MissionTerminated", "check_terminated"),
            S("inspection_step",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("MoveAndStop", "reach_and_stop"),
                A("ManageMeasurements", "stop_inspection"),
                A("AnalyseMeasurements", "analyse_measurements"),
                F("corrective_sequence",
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


def xml_inspect_corrective_path_loop() -> str:
    """Inspection corrective avec boucle de calcul de chemin."""
    return bt(S("inspection_sequence",
        *prep_with_path_loop(),
        F("inspection_loop",
            A("MissionFullyTreated", "check_complete"),
            S("inspection_step",
                A("Move", "move_to_zone"),
                A("Deccelerate", "slow_down"),
                A("ManageMeasurements", "acquire_measurements"),
                A("AnalyseMeasurements", "analyse_measurements"),
                F("quality_check",
                    A("MeasurementsQualityValidated", "check_quality"),
                    corrective_full(),
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


def xml_measure_enforced() -> str:
    """Mesure avec validation stricte."""
    return bt(S("measurement_sequence",
        *prep_base(),
        A("ProjectPointOnNetwork", "project_measurement_point"),
        A("CreatePath", "create_path"),
        A("PassMotionParameters", "set_motion_params"),
        A("Move", "move_to_measurement_point"),
        A("Deccelerate", "slow_for_measurement"),
        A("ManageMeasurements", "acquire_measurements"),
        A("AnalyseMeasurements", "analyse_measurements"),
        quality_check_enforced(),
        A("MoveAndStop", "final_stop"),
    ))


def xml_measure_reach_stop() -> str:
    """Mesure avec arrêt complet avant acquisition."""
    return bt(S("measurement_sequence",
        *prep_with_path(),
        F("measurement_loop",
            A("MissionFullyTreated", "check_complete"),
            S("measurement_step",
                A("PassMotionParameters", "set_motion_params"),
                A("MoveAndStop", "reach_measurement_point"),
                A("ManageMeasurements", "acquire_measurements"),
                A("AnalyseMeasurements", "analyse_measurements"),
                quality_check(),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
    ))


def xml_measure_multi_with_report() -> str:
    """Mesures multi-points avec rapport de défauts à chaque point."""
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
                quality_check(),
                A("PassDefectsLocalization", "report_defects"),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
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
    """Navigation sécurisée avec MoveAndStop par segment.
    Pattern BT réel reach_and_stop (lignes 98-106)."""
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


def xml_nav_safe_reach_stop() -> str:
    """Navigation sécurisée reach-and-stop avec autorisation initiale."""
    return bt(S("safe_navigation",
        *prep_with_path(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "initial_authorization"),
        motion_loop_reach_and_stop(),
        A("MoveAndStop", "final_stop"),
        A("SignalAndWaitForOrder", "signal_complete"),
    ))


def xml_nav_safe_multi_motion() -> str:
    """Navigation sécurisée avec sélecteur multi-motion."""
    return bt(S("safe_navigation",
        *prep_with_path(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "initial_authorization"),
        motion_multi_selector(),
        A("Deccelerate", "final_deceleration"),
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


def xml_complex_reach_stop_inspect_return() -> str:
    """Inspection reach-stop complète puis retour.
    Combine reach_and_stop_inspecting + return."""
    return bt(S("inspect_and_return",
        *prep_with_path(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "wait_authorization"),
        F("inspection_loop",
            A("MissionTerminated", "check_terminated"),
            S("reach_stop_inspect_step",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("MoveAndStop", "reach_and_stop"),
                A("ManageMeasurements", "stop_inspection"),
                A("AnalyseMeasurements", "analyse_measurements"),
                F("corrective_sequence",
                    A("MeasurementsQualityValidated", "check_quality"),
                    corrective_full(),
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


def xml_complex_multi_motion_inspect() -> str:
    """Navigation multi-motion suivie d'inspection."""
    return bt(S("multi_motion_inspection",
        *prep_with_path(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "wait_authorization"),
        motion_multi_selector(),
        A("Deccelerate", "decelerate_for_inspection"),
        A("ManageMeasurements", "acquire_measurements"),
        A("AnalyseMeasurements", "analyse_measurements"),
        quality_check(),
        A("MoveAndStop", "final_stop"),
        A("SignalAndWaitForOrder", "signal_complete"),
    ))


def xml_complex_path_loop_corrective() -> str:
    """Calcul de chemin en boucle + inspection corrective."""
    return bt(S("path_loop_corrective",
        *prep_with_path_loop(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "wait_authorization"),
        F("inspection_loop",
            A("MissionFullyTreated", "check_complete"),
            S("inspection_step",
                A("Move", "move_to_zone"),
                A("Deccelerate", "slow_down"),
                A("ManageMeasurements", "acquire_measurements"),
                A("AnalyseMeasurements", "analyse_measurements"),
                F("quality_check",
                    A("MeasurementsQualityValidated", "check_quality"),
                    corrective_full(),
                ),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
    ))


def xml_complex_enforced_patrol() -> str:
    """Patrouille avec validation stricte à chaque point."""
    return bt(S("enforced_patrol",
        *prep_with_path(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "wait_authorization"),
        F("patrol_loop",
            A("MissionFullyTreated", "check_complete"),
            S("patrol_step",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("Move", "move_to_checkpoint"),
                A("Deccelerate", "slow_for_inspection"),
                A("ManageMeasurements", "acquire_measurements"),
                A("AnalyseMeasurements", "analyse_measurements"),
                quality_check_enforced(),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
        A("SignalAndWaitForOrder", "signal_patrol_complete"),
    ))


def xml_complex_move_inspect_corrective() -> str:
    """Move-and-inspect suivi de phase corrective.
    Combine move_and_inspect + corrective."""
    return bt(S("move_inspect_corrective",
        *prep_with_path(),
        F("inspection_loop",
            A("MissionTerminated", "check_terminated"),
            S("move_and_inspect_step",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("ManageMeasurements", "start_inspection"),
                A("Move", "execute_move"),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("ManageMeasurements", "stop_inspection"),
        A("AnalyseMeasurements", "analyse_measurements"),
        F("quality_check",
            A("MeasurementsQualityValidated", "check_quality"),
            corrective_full(),
        ),
        A("MoveAndStop", "final_stop"),
    ))


def xml_complex_nav_measure_return() -> str:
    """Navigation + mesure + retour au dépôt."""
    return bt(S("nav_measure_return",
        *prep_with_path(),
        motion_loop(),
        A("Deccelerate", "decelerate_for_measurement"),
        A("ManageMeasurements", "acquire_measurements"),
        A("AnalyseMeasurements", "analyse_measurements"),
        quality_check(),
        A("ProjectPointOnNetwork", "project_depot"),
        A("CreatePath", "create_return_path"),
        A("PassMotionParameters", "set_return_params"),
        A("Move", "return_to_depot"),
        A("MoveAndStop", "final_stop"),
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


def xml_simulation_nav_signal() -> str:
    """Navigation autorisée en simulation."""
    return bt(S("simulation_authorized_nav",
        A("SimulationStarted", "check_simulation"),
        *prep_with_path(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "wait_authorization"),
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


def xml_simulation_reach_stop_inspect() -> str:
    """Simulation reach-stop avec inspection."""
    return bt(S("simulation_reach_stop_inspect",
        A("SimulationStarted", "check_simulation"),
        *prep_with_path(),
        A("IsRobotPoseProjectionActive", "check_projection"),
        A("SignalAndWaitForOrder", "wait_authorization"),
        F("inspection_loop",
            A("MissionTerminated", "check_terminated"),
            S("inspection_step",
                A("CheckCurrentStepType", "check_step_type"),
                A("PassMotionParameters", "set_motion_params"),
                A("MoveAndStop", "reach_and_stop"),
                A("ManageMeasurements", "stop_inspection"),
                A("AnalyseMeasurements", "analyse_measurements"),
                F("corrective_sequence",
                    A("MeasurementsQualityValidated", "check_quality"),
                    A("PassDefectsLocalization", "report_defects"),
                ),
                A("UpdateCurrentExecutedStep", "update_step"),
            ),
        ),
        A("MoveAndStop", "final_stop"),
    ))


def xml_simulation_multi_motion() -> str:
    """Navigation multi-motion en simulation."""
    return bt(S("simulation_multi_motion",
        A("SimulationStarted", "check_simulation"),
        *prep_with_path(),
        motion_multi_selector(),
        A("MoveAndStop", "final_stop"),
    ))


def xml_simulation_enforced_inspect() -> str:
    """Inspection en simulation avec validation stricte."""
    return bt(S("simulation_enforced_inspection",
        A("SimulationStarted", "check_simulation"),
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


def xml_simulation_patrol() -> str:
    """Patrouille en mode simulation."""
    return bt(S("simulation_patrol",
        A("SimulationStarted", "check_simulation"),
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
    ))


def xml_simulation_nav_path_loop() -> str:
    """Navigation en simulation avec boucle de calcul de chemin."""
    return bt(S("simulation_path_loop_nav",
        A("SimulationStarted", "check_simulation"),
        *prep_with_path_loop(),
        motion_loop(),
        A("MoveAndStop", "final_stop"),
    ))


# ─── Vocabulaire missions ────────────────────────────────────────────────────

NAV_VERBS = [
    "Déplace-toi", "Va", "Rejoins", "Navigue jusqu'à",
    "Retourne", "Avance jusqu'au", "Positionne-toi au",
    "Rends-toi", "Dirige-toi vers", "Gagne",
]
NAV_TARGETS = [
    "le dépôt principal", "le point de chargement",
    "la position de départ", "le poste de maintenance",
    "la zone de stationnement", "la voie de service",
    "le terminal de ravitaillement", "la zone de remisage",
    "le poste de contrôle central", "la voie d'évitement",
    "le point de contrôle technique", "la gare de triage",
    "le centre de commandement", "la base opérationnelle",
    "le point de raccordement", "la zone de régulation",
]
URGENCY_TGTS = [
    "le secteur d'urgence", "la zone d'incident",
    "le point d'intervention prioritaire",
    "le secteur sinistré", "la zone de crise",
]

INSPECT_OBJS = [
    "la voie", "les rails", "le tunnel ferroviaire",
    "le passage à niveau", "les aiguillages", "les traverses",
    "les soudures de rails", "la signalisation", "les capteurs de voie",
    "les fixations de rails", "la géométrie de la courbe",
    "les joints de dilatation", "les éléments de sécurité",
    "les appareils de voie", "le ballast", "les éclisses",
    "les attaches de rail", "la plateforme ferroviaire",
    "les câbles de signalisation", "les boîtiers de détection",
]
INSPECT_VERBS = [
    "Inspecte", "Contrôle", "Vérifie", "Effectue une inspection de",
    "Réalise un contrôle de", "Examine", "Analyse l'état de",
    "Évalue", "Diagnostique l'état de",
]
SECTIONS = [
    "A", "B", "C", "D", "E", "F",
    "nord", "sud", "est", "ouest",
    "principale", "secondaire", "maintenance", "critique",
    "alpha", "bravo", "charlie",
]

MEASURE_TYPES = [
    "la géométrie de voie", "le nivellement", "le dévers",
    "la largeur de voie", "l'alignement des rails",
    "les paramètres thermiques", "le profil de voie",
    "les paramètres au point de contrôle", "l'usure des rails",
    "la résistance des soudures", "la vibration de voie",
    "l'écartement de voie", "le profil d'usure",
    "les défauts de surface", "la charge axiale",
]
MEASURE_VERBS = [
    "Mesure", "Effectue des mesures de", "Enregistre",
    "Prends des mesures de", "Réalise une mesure de",
    "Effectue un relevé de", "Acquiers les données de",
    "Relève", "Capture les mesures de",
]

AUTH_NAV_MISSIONS = [
    "Navigue vers {} avec autorisation préalable",
    "Déplace-toi vers {} après autorisation du poste de contrôle",
    "Rejoins {} en mode supervisé avec validation opérateur",
    "Va au {} et attends l'autorisation avant de démarrer",
    "Effectue un transit supervisé vers {}",
    "Navigue vers {} avec activation de la projection de pose",
    "Déplace-toi vers {} en mode autorisé avec décélération",
    "Rejoins {} après vérification de la projection et autorisation",
    "Effectue un déplacement autorisé vers {} avec arrêt à chaque étape",
    "Navigue vers {} en attente d'ordre à chaque point de passage",
    "Déplace-toi vers {} avec signalisation et autorisation externe",
    "Rejoins {} via transit supervisé avec multi-segments",
]

SAFE_NAV_MISSIONS = [
    "Navigue en mode sécurisé vers {}",
    "Déplace-toi vers {} avec décélération progressive",
    "Rejoins {} en mode sécurisé avec arrêt à chaque étape",
    "Effectue une navigation sécurisée jusqu'à {}",
    "Navigue vers {} en signalant ta progression",
    "Déplace-toi pas à pas vers {} avec autorisation à chaque segment",
    "Effectue un transit sécurisé vers {} avec arrêt et signal",
    "Navigue vers {} en mode haute sécurité avec multi-arrêts",
    "Rejoins {} avec arrêt complet et autorisation entre chaque segment",
    "Déplace-toi prudemment vers {} avec signalisation continue",
]

INSPECT_MISSIONS = [
    "Inspecte {} et vérifie la qualité des mesures",
    "Effectue une inspection complète de {} avec analyse des mesures",
    "Réalise un contrôle qualité de {} avec re-mesure si nécessaire",
    "Inspecte {} en boucle jusqu'à complétion de la mission",
    "Contrôle {} avec validation de la qualité des mesures",
    "Effectue l'inspection de {} avec analyse et rapport de qualité",
    "Inspecte {} en mouvement continu avec acquisition de mesures",
    "Réalise l'inspection de {} avec arrêt à chaque point de contrôle",
    "Effectue une inspection pass-through de {} avec validation stricte",
    "Inspecte {} avec acquisition continue et analyse à chaque arrêt",
    "Réalise l'inspection de {} avec boucle de calcul de chemin",
    "Contrôle {} zone par zone avec arrêt, mesure et correction",
]

INSPECT_CORRECTIVE_MISSIONS = [
    "Inspecte {} et corrige automatiquement les défauts détectés",
    "Effectue une inspection de {} avec sous-séquence corrective si défaut",
    "Réalise un contrôle de {} : si défaut, génère et insère une correction",
    "Inspecte {} avec gestion corrective automatique des anomalies",
    "Contrôle {} et applique les corrections nécessaires si qualité insuffisante",
    "Effectue une inspection approfondie de {} avec correction et signalement des défauts",
    "Inspecte {} avec arrêt à chaque point, analyse et correction si nécessaire",
    "Réalise un contrôle de {} avec reach-and-stop et séquence corrective complète",
    "Contrôle {} avec validation stricte et correction automatique",
    "Inspecte {} en mode reach-stop avec analyse, défauts et corrections",
    "Effectue le contrôle de {} avec boucle de calcul de chemin et corrections",
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
    ("Inspecte {} en reach-stop avec analyse corrective puis retour au dépôt", xml_complex_reach_stop_inspect_return),
    ("Navigation multi-motion vers {} puis inspection avec analyse qualité", xml_complex_multi_motion_inspect),
    ("Calcul de chemin en boucle puis inspection corrective de {}", xml_complex_path_loop_corrective),
    ("Patrouille de {} avec validation stricte à chaque point de contrôle", xml_complex_enforced_patrol),
    ("Inspecte {} en mouvement continu puis phase corrective en fin", xml_complex_move_inspect_corrective),
    ("Navigue vers {}, mesure et analyse, puis retourne au dépôt", xml_complex_nav_measure_return),
    ("Inspection reach-stop de {} avec corrections automatiques et retour", xml_complex_reach_stop_inspect_return),
    ("Ronde d'inspection entre {} et {} avec validation stricte à chaque arrêt", xml_complex_enforced_patrol),
]

SIMULATION_MISSIONS = [
    "Simule une navigation vers {}",
    "Lance une simulation de navigation jusqu'à {}",
    "En mode simulation, navigue vers {}",
    "Teste en simulation la navigation vers {}",
    "Simule un transit vers {} pour validation",
    "Simule une navigation autorisée vers {}",
    "En mode simulation, effectue un transit supervisé vers {}",
    "Simule une inspection de {}",
    "Lance une simulation d'inspection de {}",
    "En mode simulation, inspecte {}",
    "Teste en simulation l'inspection de {}",
    "Simule l'inspection complète de {} avec corrections",
    "Lance une simulation d'inspection approfondie de {}",
    "En mode simulation, effectue le contrôle complet de {}",
    "Simule une inspection reach-stop de {} avec analyse qualité",
    "En mode simulation, effectue la navigation multi-motion vers {}",
    "Simule l'inspection de {} avec validation stricte des mesures",
    "En mode simulation, patrouille entre {} et {} avec inspection",
    "Simule la navigation avec boucle de calcul de chemin vers {}",
    "Lance une simulation de patrouille d'inspection de {}",
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
                          "urgence", "test", "critique", str(km()),
                          "alpha", "bravo", "delta"])

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
        f"du secteur {section()} entre km {a} et km {b}",
    ])

def target_named() -> str:
    return random.choice(NAV_TARGETS + URGENCY_TGTS)

def inspect_target() -> str:
    obj = random.choice(INSPECT_OBJS)
    loc = location()
    return f"{obj} {loc}"


# ─── Générateurs par catégorie ───────────────────────────────────────────────

def gen_navigation(n: int) -> list:
    """Navigation simple (350 ex.)."""
    templates = [xml_nav_simple, xml_nav_short, xml_nav_with_decel,
                 xml_nav_direct, xml_nav_return, xml_nav_path_loop,
                 xml_nav_multi_motion, xml_nav_mission_loop,
                 xml_nav_mission_loop_decel, xml_nav_with_projection]
    examples = []

    for _ in range(n // 2):
        verb = random.choice(NAV_VERBS)
        a, b = km_pair()
        mission = random.choice([
            f"{verb} au km {b} depuis le km {a}",
            f"{verb} au km {b}",
            f"Rejoins le km {b}",
            f"Avance de {b - a} km depuis le km {a}",
            f"{verb} du km {a} au km {b}",
            f"Transit du km {a} vers le km {b}",
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
            f"Effectue un transit vers {target}",
        ])
        examples.append(make_entry(mission, random.choice(templates)()))

    return examples


def gen_nav_authorized(n: int) -> list:
    """Navigation avec autorisation (150 ex.)."""
    templates = [xml_nav_signal, xml_nav_signal_short, xml_nav_signal_decel,
                 xml_nav_signal_reach_stop, xml_nav_signal_path_loop,
                 xml_nav_signal_multi_motion]
    examples = []

    for _ in range(n):
        tpl = random.choice(AUTH_NAV_MISSIONS)
        a, b = km_pair()
        target = random.choice([
            f"le km {b}",
            f"le km {b} depuis le km {a}",
            target_named(),
            f"la zone {zone()}",
            f"la section {section()}",
        ])
        mission = tpl.format(target)
        examples.append(make_entry(mission, random.choice(templates)()))

    return examples


def gen_inspection(n: int) -> list:
    """Inspection de voie (350 ex.)."""
    templates = [xml_inspect_simple, xml_inspect_enforced,
                 xml_inspect_with_motion_loop, xml_inspect_multi_zone,
                 xml_inspect_move_and_inspect, xml_inspect_reach_stop_inspect,
                 xml_inspect_pass_stop_inspect, xml_inspect_path_loop,
                 xml_inspect_reach_stop_no_wait]
    examples = []

    for _ in range(n):
        tpl = random.choice(INSPECT_MISSIONS)
        target = inspect_target()
        mission = tpl.format(target)
        examples.append(make_entry(mission, random.choice(templates)()))

    return examples


def gen_inspection_corrective(n: int) -> list:
    """Inspection avec corrective (200 ex.)."""
    templates = [xml_inspect_corrective, xml_inspect_corrective_signal,
                 xml_inspect_corrective_enforced,
                 xml_inspect_corrective_reach_stop,
                 xml_inspect_corrective_reach_stop_full,
                 xml_inspect_corrective_path_loop]
    examples = []

    for _ in range(n):
        tpl = random.choice(INSPECT_CORRECTIVE_MISSIONS)
        target = inspect_target()
        mission = tpl.format(target)
        examples.append(make_entry(mission, random.choice(templates)()))

    return examples


def gen_measurement(n: int) -> list:
    """Mesures simples (150 ex.)."""
    templates = [xml_measure_simple, xml_measure_with_quality,
                 xml_measure_multi_point, xml_measure_and_report,
                 xml_measure_enforced, xml_measure_reach_stop,
                 xml_measure_multi_with_report]
    examples = []

    for _ in range(n):
        verb = random.choice(MEASURE_VERBS)
        mtype = random.choice(MEASURE_TYPES)
        loc = location()
        mission = random.choice([
            f"{verb} {mtype} {loc}",
            f"Enregistre {mtype} {loc}",
            f"Effectue un relevé de {mtype} {loc}",
            f"Acquiers les données de {mtype} {loc}",
        ])
        examples.append(make_entry(mission, random.choice(templates)()))

    return examples


def gen_safe_navigation(n: int) -> list:
    """Navigation sécurisée (150 ex.)."""
    templates = [xml_nav_safe_decel, xml_nav_safe_signal, xml_nav_safe_stop,
                 xml_nav_safe_reach_stop, xml_nav_safe_multi_motion]
    examples = []

    for _ in range(n):
        tpl = random.choice(SAFE_NAV_MISSIONS)
        a, b = km_pair()
        target = random.choice([
            f"le km {b}",
            target_named(),
            f"la zone {zone()}",
            f"la section {section()}",
        ])
        mission = tpl.format(target)
        examples.append(make_entry(mission, random.choice(templates)()))

    return examples


def gen_complex(n: int) -> list:
    """Missions complexes (400 ex.)."""
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
    """Simulation (250 ex.)."""
    sim_nav_templates = [xml_simulation_nav, xml_simulation_nav_signal,
                         xml_simulation_multi_motion, xml_simulation_nav_path_loop]
    sim_inspect_templates = [xml_simulation_inspect, xml_simulation_full,
                             xml_simulation_reach_stop_inspect,
                             xml_simulation_enforced_inspect,
                             xml_simulation_patrol]
    examples = []

    for _ in range(n):
        tpl = random.choice(SIMULATION_MISSIONS)
        if tpl.count("{}") == 2:
            a, b = km_pair()
            target1 = f"km {a}"
            target2 = f"km {b}"
            mission = tpl.format(target1, target2)
        else:
            target = random.choice([
                f"le km {km()}",
                target_named(),
                inspect_target(),
                f"la zone {zone()}",
            ])
            mission = tpl.format(target)

        if "inspect" in tpl.lower() or "contrôle" in tpl.lower() or "patrouille" in tpl.lower():
            xml = random.choice(sim_inspect_templates)()
        elif "multi-motion" in tpl.lower():
            xml = xml_simulation_multi_motion()
        elif "boucle" in tpl.lower() or "calcul" in tpl.lower():
            xml = xml_simulation_nav_path_loop()
        elif "autorisé" in tpl.lower() or "supervisé" in tpl.lower():
            xml = xml_simulation_nav_signal()
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
        "Navigation simple": 350,
        "Navigation autorisée": 150,
        "Inspection": 350,
        "Inspection corrective": 200,
        "Mesures": 150,
        "Navigation sécurisée": 150,
        "Complexe": 400,
        "Simulation": 250,
    }

    print("Génération du dataset NAV4RAIL v4 (27 skills réels, 2000 exemples)...")
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
    import re
    all_skills = set()
    for entry in dataset:
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
    print(f"\n  Total : {sum(counts.values())}")
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
