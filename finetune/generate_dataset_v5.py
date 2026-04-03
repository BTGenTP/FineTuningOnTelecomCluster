"""
Generateur de dataset NAV4RAIL v5 : fidele a behavior_tree_example.xml

Format identique a la reference :
  - <Action name="HUMAN NAME" ID="SkillName" port="{var}"/>
  - <Condition name="HUMAN NAME" ID="SkillName" port="{var}"/>
  - <SubTreePlus name="HUMAN NAME" ID="subtree_id" __autoremap="true"/>
  - Multi-BehaviorTree : main -> preparation -> base_preparation -> get_mission + calculate_path
  - ReactiveFallback dans execute
  - Repeat num_cycles="-1" dans calculate_path
  - Fallback MOTION SELECTOR avec subtrees par type (0-4, 10-14)

27 skills reels, 2000 exemples, 8 categories.
"""

import json
import random
from pathlib import Path

random.seed(42)

# --- Skills doc (updated with format info) ----------------------------------

SKILLS_DOC = """Skills disponibles (27 skills, 4 familles) :

PREPARATION :
- LoadMission                    : Charge les parametres de la mission depuis la source
- MissionStructureValid          : Verifie la coherence structurelle de la mission chargee
- UpdateCurrentGeneratedActivity : Met a jour l'activite en cours de generation
- ProjectPointOnNetwork          : Projette un point sur le reseau ferroviaire
- CreatePath                     : Calcule un chemin entre deux points
- AgregatePath                   : Fusionne plusieurs segments de chemin
- MissionFullyTreated            : Verifie si toutes les etapes sont traitees
- PassAdvancedPath               : Transmet un chemin avance au module d'execution
- PassMission                    : Transmet la mission au module d'execution
- GenerateMissionSequence        : Genere la sequence d'actions pour la mission
- GenerateCorrectiveSubSequence  : Genere une sous-sequence corrective
- InsertCorrectiveSubSequence    : Insere la sous-sequence corrective

MOTION :
- MissionTerminated              : Verifie si la mission est terminee
- CheckCurrentStepType           : Verifie le type de l'etape (type_to_be_checked: 0=move, 1=decel, 2=reach_stop, 3=pass, 4=reach_stop_no_wait, 10-14=variantes inspection)
- PassMotionParameters           : Configure les parametres de mouvement
- Move                           : Deplace le robot (threshold_type: 1=normal, 3=pass-through)
- UpdateCurrentExecutedStep      : Marque l'etape comme executee
- Deccelerate                    : Reduit la vitesse du robot
- MoveAndStop                    : Deplace puis stoppe le robot
- SignalAndWaitForOrder           : Signal et attente d'autorisation externe
- IsRobotPoseProjectionActive    : Verifie si la projection de pose est active

INSPECTION :
- ManageMeasurements             : Lance/arrete l'acquisition des mesures
- AnalyseMeasurements            : Analyse les donnees de mesure
- MeasurementsQualityValidated   : Verifie la qualite des mesures
- PassDefectsLocalization        : Transmet la localisation des defauts
- MeasurementsEnforcedValidated  : Validation stricte des mesures

SIMULATION :
- SimulationStarted              : Verifie si le mode simulation est actif

Format XML BehaviorTree.CPP :
- Noeuds feuilles : <Action name="NOM" ID="Skill" port="{var}"/>
                     <Condition name="NOM" ID="Skill" port="{var}"/>
- Sous-arbres     : <SubTreePlus name="NOM" ID="subtree_id" __autoremap="true"/>
- Controle        : Sequence, Fallback, ReactiveFallback, Repeat
- Ports blackboard: {variable} pour la communication entre noeuds"""

SYSTEM_PROMPT = (
    "Tu es un expert en robotique ferroviaire NAV4RAIL. "
    "Genere un Behavior Tree au format XML BehaviorTree.CPP "
    "correspondant exactement a la mission decrite. "
    "Utilise le format multi-subtree avec Action/Condition/SubTreePlus. "
    "Inclus les ports blackboard sur chaque noeud. "
    "Reponds uniquement avec le XML, sans explication."
)

# --- Condition skills (vs Action) -------------------------------------------

CONDITION_SKILLS = {
    "MissionStructureValid", "IsRobotPoseProjectionActive", "CheckCurrentStepType",
    "MissionTerminated", "MissionFullyTreated", "MeasurementsQualityValidated",
    "MeasurementsEnforcedValidated", "SimulationStarted",
}

# --- Node builders ----------------------------------------------------------


def Act(skill_id: str, name: str, **attrs) -> dict:
    """Action leaf: <Action name="NAME" ID="Skill" attrs/>"""
    return {"tag": "Action", "name": name, "id": skill_id, "attrs": attrs}


def Cond(skill_id: str, name: str, **attrs) -> dict:
    """Condition leaf: <Condition name="NAME" ID="Skill" attrs/>"""
    return {"tag": "Condition", "name": name, "id": skill_id, "attrs": attrs}


def Sub(subtree_id: str, name: str, **attrs) -> dict:
    """SubTreePlus ref: <SubTreePlus name="NAME" ID="id" __autoremap="true" attrs/>"""
    return {"tag": "SubTreePlus", "name": name, "id": subtree_id,
            "attrs": {"__autoremap": "true", **attrs}}


def S(name: str, *children) -> dict:
    """Sequence control node."""
    return {"tag": "Sequence", "name": name, "children": list(children)}


def F(name: str, *children) -> dict:
    """Fallback control node."""
    return {"tag": "Fallback", "name": name, "children": list(children)}


def RF(name: str, *children) -> dict:
    """ReactiveFallback control node."""
    return {"tag": "ReactiveFallback", "name": name, "children": list(children)}


def R(name: str, *children, num_cycles: str = "-1") -> dict:
    """Repeat control node."""
    return {"tag": "Repeat", "name": name, "children": list(children),
            "attrs": {"num_cycles": num_cycles}}


# --- Render -----------------------------------------------------------------


def render(node: dict, depth: int = 0) -> str:
    """Render a node tree to indented XML."""
    pad = "  " * depth
    tag = node["tag"]

    # Leaf nodes: Action, Condition, SubTreePlus
    if tag in ("Action", "Condition", "SubTreePlus"):
        parts = [f'{pad}<{tag} name="{node["name"]}"']
        parts.append(f' ID="{node["id"]}"')
        for k, v in node.get("attrs", {}).items():
            parts.append(f' {k}="{v}"')
        return "".join(parts) + "/>"

    # Control nodes
    name = node.get("name", "")
    attr_str = f' name="{name}"' if name else ""
    for k, v in node.get("attrs", {}).items():
        attr_str += f' {k}="{v}"'

    children = node.get("children")
    if not children:
        return f"{pad}<{tag}{attr_str}/>"

    lines = [f"{pad}<{tag}{attr_str}>"]
    for child in children:
        lines.append(render(child, depth + 1))
    lines.append(f"{pad}</{tag}>")
    return "\n".join(lines)


def bt_multi(main_id: str, subtrees: list) -> str:
    """Assemble multiple BehaviorTree blocks into a complete XML document.
    subtrees: list of (bt_id, tree_node) tuples.
    """
    lines = [f'<root main_tree_to_execute="{main_id}">']
    for bt_id, tree in subtrees:
        lines.append(f'  <BehaviorTree ID="{bt_id}">')
        lines.append(render(tree, depth=2))
        lines.append(f'  </BehaviorTree>')
    lines.append('</root>')
    return "\n".join(lines)


def make_entry(mission: str, xml: str) -> dict:
    instruction = f"{SYSTEM_PROMPT}\n\n{SKILLS_DOC}\n\nMission : {mission}"
    return {
        "mission": mission,
        "xml": xml,
        "prompt": f"<s>[INST] {instruction} [/INST] {xml} </s>",
    }


# === FIXED SUBTREES (always the same structure) =============================

def st_get_mission():
    """get_mission subtree: load + validate."""
    return ("get_mission", S("GET MISSION",
        Act("LoadMission", "LOAD MISSION", mission_file_path="{mission_file_path}"),
        Cond("MissionStructureValid", "IS MISSION STRUCTURE VALID"),
    ))


def st_calculate_path():
    """calculate_path subtree: Repeat loop over activities."""
    return ("calculate_path", F("PATH CALCULATION",
        R("LOOP",
            S("ACTIVITY",
                Act("UpdateCurrentGeneratedActivity", "UPDATE CURRENT ACTIVITY",
                    type="{type}", origin_sph="{origin_sph}",
                    target_sph="{target_sph}", forbidden_atoms_out="{forbidden_atoms}"),
                Act("ProjectPointOnNetwork", "PROJECT ORIGIN",
                    point_in="{origin_sph}", point_out="{origin}"),
                Act("ProjectPointOnNetwork", "PROJECT TARGET",
                    point_in="{target_sph}", point_out="{target}"),
                Act("CreatePath", "CREATE PATH",
                    origin="{origin}", target="{target}",
                    forbidden_atoms="{forbidden_atoms}", path="{path}"),
                Act("AgregatePath", "AGREGATE PATH", path="{path}"),
            ),
        ),
        Cond("MissionFullyTreated", "MISSION FULLY TREATED", type="{type}"),
    ))


def st_base_preparation():
    """base_preparation subtree: get_mission + calculate_path + pass."""
    return ("base_preparation", S("BASE PREPARATION",
        Sub("get_mission", "GET MISSION", mission_file_path="{mission_file}"),
        Sub("calculate_path", "CALCULATE PATH"),
        Act("PassAdvancedPath", "PASS ADVANCED PATH", adv_path="{adv_path}"),
        Act("PassMission", "PASS MISSION", mission="{mission}"),
        Act("GenerateMissionSequence", "GENERATE MOTION SEQUENCE",
            mission="{mission}", mission_sequence="{mission_sequence}"),
    ))


# === PREPARATION VARIANTS ===================================================

def prep_simple():
    """Simple preparation: just base_preparation."""
    return [
        ("preparation", S("PREPARATION",
            Sub("base_preparation", "BASE PREPARATION"),
        )),
        st_base_preparation(),
        st_get_mission(),
        st_calculate_path(),
    ]


def prep_authorized():
    """Authorized preparation: base_preparation + projection + signal."""
    return [
        ("preparation", S("AUTHORIZED PREPARATION",
            Sub("base_preparation", "BASE PREPARATION",
                mission_file="default", adv_path="{adv_path}",
                mission_sequence="{mission_sequence}"),
            Cond("IsRobotPoseProjectionActive", "IS ROBOT PROJECTION ACTIVE",
                 adv_path="{adv_path}", pub_proj="true"),
            Act("SignalAndWaitForOrder", "SIGNAL AND WAIT",
                message="ready to move - wait for authorization"),
        )),
        st_base_preparation(),
        st_get_mission(),
        st_calculate_path(),
    ]


# === MOTION SUBTREES (10 types, faithful to reference) ======================

def st_move():
    """move subtree (type=0): simple movement."""
    return ("move", S("MOVE",
        Cond("CheckCurrentStepType", "IS CURRENT STEP MOVE",
             type_to_be_checked="0"),
        Act("PassMotionParameters", "PASS MOTION PARAMETERS",
            motion_params="{motion_params}"),
        Act("Move", "MOVE", threshold_type="1", motion_params="{motion_params}"),
        Act("UpdateCurrentExecutedStep", "UPDATE CURRENT STEP"),
    ))


def st_move_and_inspect():
    """move_and_inspect subtree (type=10): inspect during movement."""
    return ("move_and_inspect", S("MOVE AND INSPECT",
        Cond("CheckCurrentStepType", "IS CURRENT STEP MOVE AND INSPECT",
             type_to_be_checked="10"),
        Act("PassMotionParameters", "PASS MOTION PARAMETERS",
            motion_params="{motion_params}"),
        Act("ManageMeasurements", "START INSPECTION"),
        Act("Move", "MOVE", threshold_type="1", motion_params="{motion_params}"),
        Act("UpdateCurrentExecutedStep", "UPDATE CURRENT STEP"),
    ))


def st_deccelerate():
    """deccelerate subtree (type=1): slow down."""
    return ("deccelerate", S("DECCELERATE",
        Cond("CheckCurrentStepType", "IS CURRENT STEP DECCELERATE",
             type_to_be_checked="1"),
        Act("PassMotionParameters", "PASS MOTION PARAMETERS",
            motion_params="{motion_params}"),
        Act("Deccelerate", "DECCELERATE", motion_params="{motion_params}"),
        Act("UpdateCurrentExecutedStep", "UPDATE CURRENT STEP"),
    ))


def st_deccelerate_and_inspect():
    """deccelerate_and_inspect subtree (type=11): slow down during inspection."""
    return ("deccelerate_and_inspect", S("DECCELERATE AND INSPECT",
        Cond("CheckCurrentStepType", "IS CURRENT STEP DECCELERATE AND INSPECT",
             type_to_be_checked="11"),
        Act("PassMotionParameters", "PASS MOTION PARAMETERS",
            motion_params="{motion_params}"),
        Act("Deccelerate", "DECCELERATE", motion_params="{motion_params}"),
        Act("UpdateCurrentExecutedStep", "UPDATE CURRENT STEP"),
    ))


def st_reach_and_stop():
    """reach_and_stop subtree (type=2): stop and wait for authorization (PN)."""
    return ("reach_and_stop", S("REACH AND STOP",
        Cond("CheckCurrentStepType", "IS CURRENT STEP REACH AND STOP",
             type_to_be_checked="2"),
        Act("PassMotionParameters", "PASS MOTION PARAMETERS",
            motion_params="{motion_params}"),
        Act("MoveAndStop", "MOVE AND STOP", motion_params="{motion_params}"),
        Act("SignalAndWaitForOrder", "SIGNAL AND WAIT FOR AUTHORIZATION",
            message="need authorization to go further"),
        Act("UpdateCurrentExecutedStep", "UPDATE CURRENT STEP"),
    ))


def st_reach_and_stop_inspecting():
    """reach_and_stop_inspecting subtree (type=12): stop, inspect, corrective."""
    return ("reach_and_stop_inspecting", S("REACH AND STOP INSPECTING",
        Cond("CheckCurrentStepType", "IS CURRENT STEP REACH AND STOP INSPECTING",
             type_to_be_checked="12"),
        Act("PassMotionParameters", "PASS MOTION PARAMETERS",
            motion_params="{motion_params}"),
        Act("MoveAndStop", "MOVE AND STOP", motion_params="{motion_params}"),
        Act("ManageMeasurements", "STOP INSPECTION"),
        Act("AnalyseMeasurements", "ANALYSE MEASUREMENTS"),
        S("REACT ON QUALITY",
            F("CORRECTIVE SEQUENCE",
                Cond("MeasurementsQualityValidated", "IS MEASUREMENT QUALITY OK"),
                Act("PassDefectsLocalization", "PASS DEFECTS LOCALIZATION",
                    defects="{defects}"),
            ),
            Act("GenerateCorrectiveSubSequence", "GENERATE CORRECTIVE SEQUENCE",
                defects="{defects}"),
            Act("InsertCorrectiveSubSequence", "INSERT CORRECTIVE SEQUENCE"),
        ),
        Act("UpdateCurrentExecutedStep", "UPDATE CURRENT STEP"),
    ))


def st_pass():
    """pass subtree (type=3): pass-through movement."""
    return ("pass", S("PASS",
        Cond("CheckCurrentStepType", "IS CURRENT STEP PASS",
             type_to_be_checked="3"),
        Act("PassMotionParameters", "PASS MOTION PARAMETERS",
            motion_params="{motion_params}"),
        Act("Move", "MOVE", threshold_type="3", motion_params="{motion_params}"),
        Act("UpdateCurrentExecutedStep", "UPDATE CURRENT STEP"),
    ))


def st_pass_and_stop_inspecting():
    """pass_and_stop_inspecting subtree (type=13): pass-through then inspect."""
    return ("pass_and_stop_inspecting", S("PASS AND STOP INSPECTING",
        Cond("CheckCurrentStepType", "IS CURRENT STEP PASS AND STOP INSPECTING",
             type_to_be_checked="13"),
        Act("PassMotionParameters", "PASS MOTION PARAMETERS",
            motion_params="{motion_params}"),
        Act("Move", "MOVE", threshold_type="3", motion_params="{motion_params}"),
        Act("ManageMeasurements", "STOP INSPECTION"),
        F("ENFORCED ANALYSIS",
            Act("AnalyseMeasurements", "ANALYSE MEASUREMENTS"),
            Cond("MeasurementsEnforcedValidated", "ENFORCED VALIDATION"),
        ),
        Act("UpdateCurrentExecutedStep", "UPDATE CURRENT STEP"),
    ))


def st_reach_stop_and_dont_wait():
    """reach_stop_and_dont_wait subtree (type=4): stop without waiting."""
    return ("reach_stop_and_dont_wait", S("REACH STOP AND DONT WAIT",
        Cond("CheckCurrentStepType", "IS CURRENT STEP REACH STOP AND DONT WAIT",
             type_to_be_checked="4"),
        Act("PassMotionParameters", "PASS MOTION PARAMETERS",
            motion_params="{motion_params}"),
        Act("MoveAndStop", "MOVE AND STOP", motion_params="{motion_params}"),
        Act("UpdateCurrentExecutedStep", "UPDATE CURRENT STEP"),
    ))


def st_reach_stop_inspecting_dont_wait():
    """reach_stop_inspecting_dont_wait subtree (type=14): stop, inspect, no wait."""
    return ("reach_stop_inspecting_dont_wait", S("REACH AND STOP INSPECTING DONT WAIT",
        Cond("CheckCurrentStepType",
             "IS CURRENT STEP REACH AND STOP INSPECTING DONT WAIT",
             type_to_be_checked="14"),
        Act("PassMotionParameters", "PASS MOTION PARAMETERS",
            motion_params="{motion_params}"),
        Act("MoveAndStop", "MOVE AND STOP", motion_params="{motion_params}"),
        Act("ManageMeasurements", "STOP INSPECTION"),
        Act("AnalyseMeasurements", "ANALYSE MEASUREMENTS"),
        F("CORRECTIVE SEQUENCE",
            Cond("MeasurementsQualityValidated", "IS MEASUREMENTS QUALITY OK"),
            S("REACT ON MEASUREMENTS QUALITY",
                Act("PassDefectsLocalization", "PASS DEFECTS LOCALIZATION",
                    defects="{defects}"),
                Act("GenerateCorrectiveSubSequence", "GENERATE CORRECTIVE SEQUENCE",
                    defects="{defects}"),
                Act("InsertCorrectiveSubSequence", "INSERT CORRECTIVE SEQUENCE"),
            ),
        ),
        Act("UpdateCurrentExecutedStep", "UPDATE CURRENT STEP"),
    ))


# Registry of all motion subtree builders
ALL_MOTION_BUILDERS = {
    "move": st_move,
    "move_and_inspect": st_move_and_inspect,
    "deccelerate": st_deccelerate,
    "deccelerate_and_inspect": st_deccelerate_and_inspect,
    "reach_and_stop": st_reach_and_stop,
    "reach_and_stop_inspecting": st_reach_and_stop_inspecting,
    "pass": st_pass,
    "pass_and_stop_inspecting": st_pass_and_stop_inspecting,
    "reach_stop_and_dont_wait": st_reach_stop_and_dont_wait,
    "reach_stop_inspecting_dont_wait": st_reach_stop_inspecting_dont_wait,
}

# Human-readable names for SubTreePlus references in execute
MOTION_HUMAN_NAMES = {
    "move": "MOVE",
    "move_and_inspect": "MOVE AND INSPECT",
    "deccelerate": "DECCELERATE",
    "deccelerate_and_inspect": "DECCELERATE AND INSPECT",
    "reach_and_stop": "REACH AND STOP",
    "reach_and_stop_inspecting": "REACH AND STOP INSPECTING",
    "pass": "PASS",
    "pass_and_stop_inspecting": "PASS AND STOP INSPECTING",
    "reach_stop_and_dont_wait": "REACH AND STOP WITHOUT WAITING",
    "reach_stop_inspecting_dont_wait": "REACH AND STOP INSPECTING WITHOUT WAITING",
}


# === EXECUTE SUBTREE ========================================================

def st_execute(motion_ids: list) -> tuple:
    """Build execute subtree with ReactiveFallback + selected motion subtrees."""
    subtree_refs = [Sub(mid, MOTION_HUMAN_NAMES[mid]) for mid in motion_ids]
    return ("execute", RF("EXECUTE",
        R("SEQUENCE STEP LOOP",
            F("MOTION SELECTOR", *subtree_refs),
        ),
        Cond("MissionTerminated", "IS MISSION TERMINATED"),
    ))


# === FULL TREE ASSEMBLY =====================================================

def build_tree(main_id: str, main_name: str, prep_fn, motion_ids: list,
               pre_prep=None, post_execute=None) -> str:
    """Assemble a complete multi-BehaviorTree XML.

    Args:
        main_id: ID for the main BehaviorTree
        main_name: Human-readable name for the main Sequence
        prep_fn: prep_simple or prep_authorized
        motion_ids: list of motion subtree IDs to include
        pre_prep: optional nodes before preparation in main tree (e.g. SimulationStarted)
        post_execute: optional nodes after execute in main tree (e.g. final stop + signal)
    """
    # Build main tree children
    main_children = []
    if pre_prep:
        main_children.extend(pre_prep)
    main_children.append(Sub("preparation", "PREPARATION"))
    main_children.append(Sub("execute", "EXECUTE"))
    if post_execute:
        main_children.extend(post_execute)

    subtrees = []
    # 1. Main tree
    subtrees.append((main_id, S(main_name, *main_children)))
    # 2. Preparation subtrees
    subtrees.extend(prep_fn())
    # 3. Execute subtree
    subtrees.append(st_execute(motion_ids))
    # 4. Motion subtrees
    for mid in motion_ids:
        subtrees.append(ALL_MOTION_BUILDERS[mid]())

    return bt_multi(main_id, subtrees)


# === MOTION PROFILES (sets of motion subtrees per category) =================

NAV_PROFILES = [
    ["move"],
    ["move", "deccelerate"],
    ["move", "pass"],
    ["move", "deccelerate", "pass"],
    ["move", "deccelerate", "reach_stop_and_dont_wait"],
    ["move", "deccelerate", "reach_and_stop", "pass"],
]

NAV_AUTH_PROFILES = [
    ["move", "reach_and_stop"],
    ["move", "deccelerate", "reach_and_stop"],
    ["move", "deccelerate", "reach_and_stop", "pass"],
    ["move", "deccelerate", "reach_and_stop", "pass", "reach_stop_and_dont_wait"],
]

INSPECT_PROFILES = [
    ["move_and_inspect"],
    ["move_and_inspect", "deccelerate_and_inspect"],
    ["move_and_inspect", "reach_and_stop_inspecting"],
    ["move_and_inspect", "deccelerate_and_inspect", "reach_and_stop_inspecting"],
    ["move_and_inspect", "pass_and_stop_inspecting"],
    ["move", "move_and_inspect", "reach_and_stop_inspecting"],
]

INSPECT_CORRECTIVE_PROFILES = [
    ["move_and_inspect", "reach_and_stop_inspecting"],
    ["move_and_inspect", "reach_and_stop_inspecting", "reach_stop_inspecting_dont_wait"],
    ["move_and_inspect", "deccelerate_and_inspect", "reach_and_stop_inspecting"],
    ["move_and_inspect", "reach_and_stop_inspecting", "pass_and_stop_inspecting"],
    ["move", "move_and_inspect", "reach_and_stop_inspecting",
     "reach_stop_inspecting_dont_wait"],
]

MEASURE_PROFILES = [
    ["move", "reach_stop_and_dont_wait"],
    ["move", "reach_stop_inspecting_dont_wait"],
    ["move_and_inspect", "reach_stop_inspecting_dont_wait"],
    ["move", "deccelerate", "reach_stop_inspecting_dont_wait"],
]

SAFE_NAV_PROFILES = [
    ["move", "deccelerate", "reach_and_stop"],
    ["move", "deccelerate", "reach_and_stop", "pass"],
    ["move", "deccelerate", "reach_and_stop", "pass", "reach_stop_and_dont_wait"],
]

COMPLEX_PROFILES = [
    ["move", "move_and_inspect", "deccelerate", "reach_and_stop",
     "reach_and_stop_inspecting"],
    ["move", "move_and_inspect", "deccelerate", "deccelerate_and_inspect",
     "reach_and_stop", "reach_and_stop_inspecting", "pass"],
    ["move", "move_and_inspect", "deccelerate", "deccelerate_and_inspect",
     "reach_and_stop", "reach_and_stop_inspecting", "pass",
     "pass_and_stop_inspecting", "reach_stop_and_dont_wait",
     "reach_stop_inspecting_dont_wait"],
    ["move", "move_and_inspect", "deccelerate", "reach_and_stop",
     "reach_and_stop_inspecting", "pass", "pass_and_stop_inspecting"],
    ["move", "deccelerate", "reach_and_stop", "pass",
     "reach_stop_and_dont_wait", "reach_stop_inspecting_dont_wait"],
]

SIM_NAV_PROFILES = NAV_PROFILES
SIM_INSPECT_PROFILES = INSPECT_PROFILES


# === POST-EXECUTE VARIANTS ==================================================

def post_stop():
    return [Act("MoveAndStop", "FINAL STOP", motion_params="{motion_params}")]


def post_stop_signal():
    return [
        Act("MoveAndStop", "FINAL STOP", motion_params="{motion_params}"),
        Act("SignalAndWaitForOrder", "SIGNAL MISSION COMPLETE",
            message="mission complete"),
    ]


# === CATEGORY GENERATORS ====================================================

# --- Vocabulary -------------------------------------------------------------

NAV_VERBS = [
    "Deplace-toi", "Va", "Rejoins", "Navigue jusqu'a",
    "Retourne", "Avance jusqu'au", "Positionne-toi au",
    "Rends-toi", "Dirige-toi vers", "Gagne",
]
NAV_TARGETS = [
    "le depot principal", "le point de chargement",
    "la position de depart", "le poste de maintenance",
    "la zone de stationnement", "la voie de service",
    "le terminal de ravitaillement", "la zone de remisage",
    "le poste de controle central", "la voie d'evitement",
    "le point de controle technique", "la gare de triage",
    "le centre de commandement", "la base operationnelle",
    "le point de raccordement", "la zone de regulation",
]
URGENCY_TGTS = [
    "le secteur d'urgence", "la zone d'incident",
    "le point d'intervention prioritaire",
    "le secteur sinistre", "la zone de crise",
]
INSPECT_OBJS = [
    "la voie", "les rails", "le tunnel ferroviaire",
    "le passage a niveau", "les aiguillages", "les traverses",
    "les soudures de rails", "la signalisation", "les capteurs de voie",
    "les fixations de rails", "la geometrie de la courbe",
    "les joints de dilatation", "les elements de securite",
    "les appareils de voie", "le ballast", "les eclisses",
    "les attaches de rail", "la plateforme ferroviaire",
    "les cables de signalisation", "les boitiers de detection",
]
INSPECT_VERBS = [
    "Inspecte", "Controle", "Verifie", "Effectue une inspection de",
    "Realise un controle de", "Examine", "Analyse l'etat de",
    "Evalue", "Diagnostique l'etat de",
]
SECTIONS = [
    "A", "B", "C", "D", "E", "F",
    "nord", "sud", "est", "ouest",
    "principale", "secondaire", "maintenance", "critique",
    "alpha", "bravo", "charlie",
]
MEASURE_TYPES = [
    "la geometrie de voie", "le nivellement", "le devers",
    "la largeur de voie", "l'alignement des rails",
    "les parametres thermiques", "le profil de voie",
    "les parametres au point de controle", "l'usure des rails",
    "la resistance des soudures", "la vibration de voie",
    "l'ecartement de voie", "le profil d'usure",
    "les defauts de surface", "la charge axiale",
]
MEASURE_VERBS = [
    "Mesure", "Effectue des mesures de", "Enregistre",
    "Prends des mesures de", "Realise une mesure de",
    "Effectue un releve de", "Acquiers les donnees de",
    "Releve", "Capture les mesures de",
]

AUTH_NAV_MISSIONS = [
    "Navigue vers {} avec autorisation prealable",
    "Deplace-toi vers {} apres autorisation du poste de controle",
    "Rejoins {} en mode supervise avec validation operateur",
    "Va au {} et attends l'autorisation avant de demarrer",
    "Effectue un transit supervise vers {}",
    "Navigue vers {} avec activation de la projection de pose",
    "Deplace-toi vers {} en mode autorise avec deceleration",
    "Rejoins {} apres verification de la projection et autorisation",
    "Effectue un deplacement autorise vers {} avec arret a chaque etape",
    "Navigue vers {} en attente d'ordre a chaque point de passage",
    "Deplace-toi vers {} avec signalisation et autorisation externe",
    "Rejoins {} via transit supervise avec multi-segments",
]

SAFE_NAV_MISSIONS = [
    "Navigue en mode securise vers {}",
    "Deplace-toi vers {} avec deceleration progressive",
    "Rejoins {} en mode securise avec arret a chaque etape",
    "Effectue une navigation securisee jusqu'a {}",
    "Navigue vers {} en signalant ta progression",
    "Deplace-toi pas a pas vers {} avec autorisation a chaque segment",
    "Effectue un transit securise vers {} avec arret et signal",
    "Navigue vers {} en mode haute securite avec multi-arrets",
    "Rejoins {} avec arret complet et autorisation entre chaque segment",
    "Deplace-toi prudemment vers {} avec signalisation continue",
]

INSPECT_MISSIONS = [
    "Inspecte {} et verifie la qualite des mesures",
    "Effectue une inspection complete de {} avec analyse des mesures",
    "Realise un controle qualite de {} avec re-mesure si necessaire",
    "Inspecte {} en boucle jusqu'a completion de la mission",
    "Controle {} avec validation de la qualite des mesures",
    "Effectue l'inspection de {} avec analyse et rapport de qualite",
    "Inspecte {} en mouvement continu avec acquisition de mesures",
    "Realise l'inspection de {} avec arret a chaque point de controle",
    "Effectue une inspection pass-through de {} avec validation stricte",
    "Inspecte {} avec acquisition continue et analyse a chaque arret",
    "Realise l'inspection de {} avec boucle de calcul de chemin",
    "Controle {} zone par zone avec arret, mesure et correction",
]

INSPECT_CORRECTIVE_MISSIONS = [
    "Inspecte {} et corrige automatiquement les defauts detectes",
    "Effectue une inspection de {} avec sous-sequence corrective si defaut",
    "Realise un controle de {} : si defaut, genere et insere une correction",
    "Inspecte {} avec gestion corrective automatique des anomalies",
    "Controle {} et applique les corrections necessaires si qualite insuffisante",
    "Effectue une inspection approfondie de {} avec correction et signalement des defauts",
    "Inspecte {} avec arret a chaque point, analyse et correction si necessaire",
    "Realise un controle de {} avec reach-and-stop et sequence corrective complete",
    "Controle {} avec validation stricte et correction automatique",
    "Inspecte {} en mode reach-stop avec analyse, defauts et corrections",
    "Effectue le controle de {} avec boucle de calcul de chemin et corrections",
]

COMPLEX_MISSIONS = [
    "Inspecte {} et reviens au depot apres completion",
    "Effectue une inspection complete de {} avec correction puis retour",
    "Realise l'inspection integrale de {} : autorisation, mesures, corrections, signal final",
    "Navigue vers {} puis inspecte la zone a l'arrivee",
    "Effectue une patrouille d'inspection entre {} et {} avec mesures a chaque point",
    "Inspecte {} en mode supervise avec rapport de defauts et retour au depot",
    "Realise une ronde de controle de {} avec analyse qualite et correction automatique",
    "Patrouille entre {} et {} en inspectant chaque checkpoint",
    "Inspection complete de {} avec autorisation, mesures, corrections et signal de fin",
    "Inspecte {} avec re-mesure stricte et retour au point de depart",
    "Controle post-travaux de {} : inspection, validation qualite, corrections, rapport",
    "Navigation multi-motion vers {} puis inspection avec analyse qualite",
    "Inspecte {} en mouvement continu puis phase corrective en fin",
    "Navigue vers {}, mesure et analyse, puis retourne au depot",
    "Inspection reach-stop de {} avec corrections automatiques et retour",
    "Ronde d'inspection entre {} et {} avec validation stricte a chaque arret",
]

SIMULATION_MISSIONS = [
    "Simule une navigation vers {}",
    "Lance une simulation de navigation jusqu'a {}",
    "En mode simulation, navigue vers {}",
    "Teste en simulation la navigation vers {}",
    "Simule un transit vers {} pour validation",
    "Simule une navigation autorisee vers {}",
    "En mode simulation, effectue un transit supervise vers {}",
    "Simule une inspection de {}",
    "Lance une simulation d'inspection de {}",
    "En mode simulation, inspecte {}",
    "Teste en simulation l'inspection de {}",
    "Simule l'inspection complete de {} avec corrections",
    "Lance une simulation d'inspection approfondie de {}",
    "En mode simulation, effectue le controle complet de {}",
    "Simule une inspection reach-stop de {} avec analyse qualite",
    "En mode simulation, effectue la navigation multi-motion vers {}",
    "Simule l'inspection de {} avec validation stricte des mesures",
    "En mode simulation, patrouille entre {} et {} avec inspection",
    "Simule la navigation avec boucle de calcul de chemin vers {}",
    "Lance une simulation de patrouille d'inspection de {}",
]


# --- Helpers ----------------------------------------------------------------

def km():
    return random.randint(0, 99)

def km_pair():
    a = random.randint(0, 90)
    return a, a + random.randint(2, 15)

def section():
    return random.choice(SECTIONS)

def zone():
    return random.choice(["A", "B", "C", "nord", "sud", "maintenance",
                          "urgence", "test", "critique", str(km()),
                          "alpha", "bravo", "delta"])

def location():
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

def target_named():
    return random.choice(NAV_TARGETS + URGENCY_TGTS)

def inspect_target():
    return f"{random.choice(INSPECT_OBJS)} {location()}"


# --- Generation per category ------------------------------------------------

def gen_navigation(n: int) -> list:
    """Navigation simple (350 ex.)."""
    examples = []
    for _ in range(n):
        profile = random.choice(NAV_PROFILES)
        verb = random.choice(NAV_VERBS)
        a, b = km_pair()
        if random.random() < 0.5:
            mission = random.choice([
                f"{verb} au km {b} depuis le km {a}",
                f"{verb} au km {b}",
                f"Rejoins le km {b}",
                f"{verb} du km {a} au km {b}",
                f"Transit du km {a} vers le km {b}",
            ])
        else:
            target = target_named()
            mission = random.choice([
                f"{verb} {target}",
                f"Deplace-toi vers {target}",
                f"Rejoins {target}",
                f"Effectue un transit vers {target}",
            ])
        xml = build_tree("navigation_mission", "NAVIGATION MISSION",
                         prep_simple, profile, post_execute=post_stop())
        examples.append(make_entry(mission, xml))
    return examples


def gen_nav_authorized(n: int) -> list:
    """Navigation avec autorisation (150 ex.)."""
    examples = []
    for _ in range(n):
        profile = random.choice(NAV_AUTH_PROFILES)
        tpl = random.choice(AUTH_NAV_MISSIONS)
        a, b = km_pair()
        target = random.choice([
            f"le km {b}", f"le km {b} depuis le km {a}",
            target_named(), f"la zone {zone()}", f"la section {section()}",
        ])
        mission = tpl.format(target)
        xml = build_tree("authorized_navigation_mission",
                         "AUTHORIZED NAVIGATION MISSION",
                         prep_authorized, profile, post_execute=post_stop())
        examples.append(make_entry(mission, xml))
    return examples


def gen_inspection(n: int) -> list:
    """Inspection de voie (350 ex.)."""
    examples = []
    for _ in range(n):
        profile = random.choice(INSPECT_PROFILES)
        tpl = random.choice(INSPECT_MISSIONS)
        target = inspect_target()
        mission = tpl.format(target)
        prep = random.choice([prep_simple, prep_authorized])
        main_id = "inspection_mission"
        xml = build_tree(main_id, "INSPECTION MISSION",
                         prep, profile, post_execute=post_stop())
        examples.append(make_entry(mission, xml))
    return examples


def gen_inspection_corrective(n: int) -> list:
    """Inspection avec corrective (200 ex.)."""
    examples = []
    for _ in range(n):
        profile = random.choice(INSPECT_CORRECTIVE_PROFILES)
        tpl = random.choice(INSPECT_CORRECTIVE_MISSIONS)
        target = inspect_target()
        mission = tpl.format(target)
        prep = random.choice([prep_simple, prep_authorized])
        xml = build_tree("corrective_inspection_mission",
                         "CORRECTIVE INSPECTION MISSION",
                         prep, profile, post_execute=post_stop_signal())
        examples.append(make_entry(mission, xml))
    return examples


def gen_measurement(n: int) -> list:
    """Mesures simples (150 ex.)."""
    examples = []
    for _ in range(n):
        profile = random.choice(MEASURE_PROFILES)
        verb = random.choice(MEASURE_VERBS)
        mtype = random.choice(MEASURE_TYPES)
        loc = location()
        mission = random.choice([
            f"{verb} {mtype} {loc}",
            f"Enregistre {mtype} {loc}",
            f"Effectue un releve de {mtype} {loc}",
            f"Acquiers les donnees de {mtype} {loc}",
        ])
        xml = build_tree("measurement_mission", "MEASUREMENT MISSION",
                         prep_simple, profile, post_execute=post_stop())
        examples.append(make_entry(mission, xml))
    return examples


def gen_safe_navigation(n: int) -> list:
    """Navigation securisee (150 ex.)."""
    examples = []
    for _ in range(n):
        profile = random.choice(SAFE_NAV_PROFILES)
        tpl = random.choice(SAFE_NAV_MISSIONS)
        a, b = km_pair()
        target = random.choice([
            f"le km {b}", target_named(),
            f"la zone {zone()}", f"la section {section()}",
        ])
        mission = tpl.format(target)
        xml = build_tree("safe_navigation_mission", "SAFE NAVIGATION MISSION",
                         prep_authorized, profile,
                         post_execute=post_stop_signal())
        examples.append(make_entry(mission, xml))
    return examples


def gen_complex(n: int) -> list:
    """Missions complexes (400 ex.)."""
    examples = []
    for _ in range(n):
        profile = random.choice(COMPLEX_PROFILES)
        tpl = random.choice(COMPLEX_MISSIONS)
        if tpl.count("{}") == 2:
            a, b = km_pair()
            mission = tpl.format(f"km {a}", f"km {b}")
        else:
            target = random.choice([inspect_target(), f"la section {section()}",
                                    f"la zone {zone()}"])
            mission = tpl.format(target)
        xml = build_tree("complex_mission", "COMPLEX MISSION",
                         prep_authorized, profile,
                         post_execute=post_stop_signal())
        examples.append(make_entry(mission, xml))
    return examples


def gen_simulation(n: int) -> list:
    """Simulation (250 ex.)."""
    examples = []
    for _ in range(n):
        tpl = random.choice(SIMULATION_MISSIONS)
        if tpl.count("{}") == 2:
            a, b = km_pair()
            mission = tpl.format(f"km {a}", f"km {b}")
        else:
            target = random.choice([f"le km {km()}", target_named(),
                                    inspect_target(), f"la zone {zone()}"])
            mission = tpl.format(target)

        is_inspect = any(w in tpl.lower() for w in
                         ["inspect", "controle", "patrouille"])
        if is_inspect:
            profile = random.choice(SIM_INSPECT_PROFILES)
        else:
            profile = random.choice(SIM_NAV_PROFILES)

        prep = random.choice([prep_simple, prep_authorized])
        pre = [Cond("SimulationStarted", "IS SIMULATION ACTIVE")]
        xml = build_tree("simulation_mission", "SIMULATION MISSION",
                         prep, profile, pre_prep=pre,
                         post_execute=post_stop())
        examples.append(make_entry(mission, xml))
    return examples


# --- Assemblage & sauvegarde ------------------------------------------------

def main():
    import xml.etree.ElementTree as ET
    output_dir = Path(__file__).parent
    out_jsonl = output_dir / "dataset_nav4rail_v5.jsonl"
    out_json = output_dir / "dataset_nav4rail_v5.json"

    counts = {
        "Navigation simple": 350,
        "Navigation autorisee": 150,
        "Inspection": 350,
        "Inspection corrective": 200,
        "Mesures": 150,
        "Navigation securisee": 150,
        "Complexe": 400,
        "Simulation": 250,
    }

    print("Generation du dataset NAV4RAIL v5 (multi-subtree, fidele a la reference)...")
    dataset = (
        gen_navigation(counts["Navigation simple"])
        + gen_nav_authorized(counts["Navigation autorisee"])
        + gen_inspection(counts["Inspection"])
        + gen_inspection_corrective(counts["Inspection corrective"])
        + gen_measurement(counts["Mesures"])
        + gen_safe_navigation(counts["Navigation securisee"])
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

    # Verification couverture des 27 skills
    import re
    all_skills = set()
    for entry in dataset:
        ids = re.findall(r'ID="(\w+)"', entry["xml"])
        all_skills.update(ids)
    # Remove subtree IDs (not skills)
    subtree_ids = {"get_mission", "calculate_path", "base_preparation",
                   "preparation", "execute", "navigation_mission",
                   "authorized_navigation_mission", "inspection_mission",
                   "corrective_inspection_mission", "measurement_mission",
                   "safe_navigation_mission", "complex_mission",
                   "simulation_mission", "MainTree"}
    subtree_ids.update(ALL_MOTION_BUILDERS.keys())
    skill_ids = all_skills - subtree_ids

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
    missing = expected_skills - skill_ids
    extra = skill_ids - expected_skills

    # Verification patterns
    checks = {
        "Action ID=": any('Action name=' in e["xml"] for e in dataset),
        "Condition ID=": any('Condition name=' in e["xml"] for e in dataset),
        "SubTreePlus": any("SubTreePlus" in e["xml"] for e in dataset),
        "__autoremap": any("__autoremap" in e["xml"] for e in dataset),
        "ReactiveFallback": any("ReactiveFallback" in e["xml"] for e in dataset),
        "Repeat num_cycles": any("num_cycles" in e["xml"] for e in dataset),
        "type_to_be_checked": any("type_to_be_checked" in e["xml"] for e in dataset),
        "threshold_type": any("threshold_type" in e["xml"] for e in dataset),
        "motion_params": any("{motion_params}" in e["xml"] for e in dataset),
        "message=": any('message="' in e["xml"] for e in dataset),
        "main_tree_to_execute": any("main_tree_to_execute" in e["xml"] for e in dataset),
        "Multi BehaviorTree": any(e["xml"].count("BehaviorTree") > 2 for e in dataset),
    }

    # Sauvegarde
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Stats
    avg_len = sum(len(e["xml"]) for e in dataset) / len(dataset)
    avg_bts = sum(e["xml"].count("<BehaviorTree") for e in dataset) / len(dataset)

    print(f"\nDataset v5 : {len(dataset)} exemples")
    for cat, cnt in counts.items():
        print(f"  {cat:<25}: {cnt}")
    print(f"\n  Total : {sum(counts.values())}")
    print(f"\nValidation XML : {'OK -- 0 erreur' if errors == 0 else f'{errors} ERREURS'}")
    print(f"\nCouverture skills : {len(skill_ids)}/{len(expected_skills)}")
    if missing:
        print(f"  Skills MANQUANTS : {missing}")
    if extra:
        print(f"  Skills INCONNUS  : {extra}")
    print(f"\nPatterns reference :")
    for name, ok in checks.items():
        print(f"  {name:<25}: {'OK' if ok else 'ABSENT'}")
    print(f"\nStatistiques :")
    print(f"  Taille moyenne XML     : {avg_len:.0f} chars")
    print(f"  BehaviorTrees moyens   : {avg_bts:.1f} par exemple")
    print(f"\n  -> {out_jsonl}")
    print(f"  -> {out_json}")

    # Exemples
    print("\n== Exemple : navigation simple ==")
    s = next(e for e in dataset if "navigation_mission" in e["xml"]
             and "authorized" not in e["xml"] and "simulation" not in e["xml"])
    print(f"Mission : {s['mission']}")
    print(s["xml"][:2000])

    print("\n== Exemple : inspection corrective ==")
    s2 = next(e for e in dataset if "corrective_inspection" in e["xml"])
    print(f"Mission : {s2['mission']}")
    print(s2["xml"][:3000])


if __name__ == "__main__":
    main()
