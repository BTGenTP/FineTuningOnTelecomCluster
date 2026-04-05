"""
Générateur de dataset NAV4RAIL via LLM (vLLM) + LangGraph
==========================================================

Utilise un graphe LangGraph avec self-correction :
  1. Génère une instruction métier aléatoire
  2. Le LLM génère le XML BehaviorTree.CPP
  3. Validation multi-niveaux (validate_bt.py)
  4. En cas d'erreur → renvoie l'erreur au LLM pour correction (max 3 tentatives)

Usage :
    # Sur dataia25 (via SSH tunnel sur port 8000) :
    python generate_dataset_llm.py --url http://localhost:8000/v1 --count 50

    # Sur Vast.ai (via SSH tunnel) :
    ssh -N -L 8000:localhost:8000 -p <PORT> root@<IP>
    python generate_dataset_llm.py --url http://localhost:8000/v1 --count 50

Prérequis :
    pip install langgraph langchain-openai
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Import du validateur existant
sys.path.insert(0, str(Path(__file__).parent))
from validate_bt import validate_bt

# ─── Constantes ───────────────────────────────────────────────────────────────

MODEL_ID = "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"
MAX_RETRIES = 3

SKILLS_DOC = """Skills (28, 5 familles) :

PREPARATION :
- LoadMission (mission_file_path)
- MissionStructureValid [Condition]
- UpdateCurrentGeneratedActivity (type, origin_sph, target_sph, forbidden_atoms_out)
- ProjectPointOnNetwork (point_in, point_out)
- CreatePath (origin, target, forbidden_atoms, path)
- AgregatePath (path)
- MissionFullyTreated [Condition] (type)
- PassAdvancedPath (adv_path)
- PassMission (mission)
- GenerateMissionSequence (mission, mission_sequence)
- GenerateCorrectiveSubSequence (defects)
- InsertCorrectiveSubSequence

MOTION :
- MissionTerminated [Condition]
- CheckCurrentStepType [Condition] (type_to_be_checked: 0=move 1=decel 2=reach_stop 3=pass 4=no_wait 10-14=inspection)
- PassMotionParameters (motion_params)
- Move (threshold_type: 1=normal 3=pass, motion_params)
- UpdateCurrentExecutedStep
- Deccelerate (motion_params)
- MoveAndStop (motion_params)
- SignalAndWaitForOrder (message)
- IsRobotPoseProjectionActive [Condition] (adv_path, pub_proj)
- Pause (duration)

INSPECTION :
- ManageMeasurements
- AnalyseMeasurements
- MeasurementsQualityValidated [Condition]
- PassDefectsLocalization (defects)
- MeasurementsEnforcedValidated [Condition]

SIMULATION :
- SimulationStarted [Condition]"""

SYSTEM_PROMPT = (
    "Tu es un expert en robotique ferroviaire NAV4RAIL. "
    "Genere un Behavior Tree XML BehaviorTree.CPP pour la mission decrite.\n\n"
    "FORMAT :\n"
    '- <root BTCPP_format="4" main_tree_to_execute="nom">\n'
    '- Multi-<BehaviorTree ID="..."> interconnectes via <SubTreePlus __autoremap="true">\n'
    '- <Action name="NOM" ID="Skill" port="{var}"/>  <Condition name="NOM" ID="Skill"/>\n'
    '- Controle : Sequence, Fallback, ReactiveFallback, Repeat(num_cycles="-1")\n'
    '- Chaque noeud a name="DESCRIPTION EN MAJUSCULES", ports blackboard {variable}\n\n'
    "ARCHITECTURE :\n"
    "principal -> Sequence(preparation + execution via SubTreePlus)\n"
    "preparation -> LoadMission + MissionStructureValid + calculate_path + PassAdvancedPath + PassMission + GenerateMissionSequence\n"
    "calculate_path -> Fallback(Repeat(-1)(UpdateCurrentGeneratedActivity/ProjectPointOnNetwork/CreatePath/AgregatePath), MissionFullyTreated)\n"
    "execution -> ReactiveFallback(Repeat(-1)(Fallback motion_selector), MissionTerminated)\n\n"
    "CHOIX DES MOTION SUBTREES (CRUCIAL — adapter a la mission) :\n"
    "Transport (TOUJOURS inclure) :\n"
    "  move(type=0 Move), deccelerate(type=1 Deccelerate), reach_and_stop(type=2 MoveAndStop+SignalAndWaitForOrder), pass(type=3 Move threshold=3), reach_stop_no_wait(type=4 MoveAndStop)\n"
    "Inspection AVEC controle (si 'verifier'/'controler' les mesures) — AJOUTER :\n"
    "  move_and_inspect(type=10): Pause + ManageMeasurements(start) + Move\n"
    "  deccel_and_inspect(type=11): Deccelerate (mesures en cours)\n"
    "  reach_stop_inspecting(type=12): MoveAndStop + ManageMeasurements(stop) + AnalyseMeasurements + Fallback(MeasurementsQualityValidated, PassDefectsLocalization) + GenerateCorrectiveSubSequence + InsertCorrectiveSubSequence\n"
    "  pass_stop_inspecting(type=13): Move(pass) + ManageMeasurements(stop) + Fallback(AnalyseMeasurements, MeasurementsEnforcedValidated)\n"
    "  reach_stop_inspect_no_wait(type=14): comme type=12 sans SignalAndWaitForOrder\n"
    "Inspection SANS controle (mesures 'a la volee') — AJOUTER :\n"
    "  types 10-14 avec ManageMeasurements MAIS SANS AnalyseMeasurements/MeasurementsQualityValidated\n\n"
    "Condition dans Fallback : MeasurementsQualityValidated TOUJOURS enfant direct de Fallback.\n\n"
    "VARIETE STRUCTURELLE (IMPORTANT) :\n"
    "- Adapte le name= de chaque noeud a la mission specifique (element inspecte, km, contexte).\n"
    "- Tu PEUX varier l'ordre des subtrees dans le MOTION SELECTOR.\n"
    "- Tu PEUX ajouter des Pause(duration) entre certaines etapes quand c'est pertinent.\n"
    "- Tu PEUX omettre certains subtrees optionnels (ex: pass type=3 ou reach_stop_no_wait type=4 ne sont pas toujours necessaires).\n"
    "- Les durations de Pause peuvent varier (1.0 a 5.0).\n"
    "- Les messages de SignalAndWaitForOrder doivent refleter la mission.\n"
    "- Ajoute des commentaires XML <!-- ... --> decrivant la mission.\n"
    "Reponds uniquement avec le XML."
)

# Exemple condensé (basé sur behavior_tree_example.xml) — montre transport + inspection
FEW_SHOT_EXAMPLE = """EXEMPLE (inspection avec controle) :
<root BTCPP_format="4" main_tree_to_execute="mission">
  <BehaviorTree ID="mission">
    <Sequence name="MISSION">
      <SubTreePlus name="PREPARATION" ID="preparation" __autoremap="true"/>
      <SubTreePlus name="EXECUTE" ID="execute" __autoremap="true"/>
    </Sequence>
  </BehaviorTree>
  <BehaviorTree ID="preparation">
    <Sequence name="PREPARATION">
      <Action name="LOAD MISSION" ID="LoadMission" mission_file_path="{mission_file_path}"/>
      <Condition name="IS MISSION VALID" ID="MissionStructureValid"/>
      <SubTreePlus name="CALCULATE PATH" ID="calculate_path" __autoremap="true"/>
      <Action name="PASS ADVANCED PATH" ID="PassAdvancedPath" adv_path="{adv_path}"/>
      <Action name="PASS MISSION" ID="PassMission" mission="{mission}"/>
      <Action name="GENERATE SEQUENCE" ID="GenerateMissionSequence" mission="{mission}" mission_sequence="{mission_sequence}"/>
    </Sequence>
  </BehaviorTree>
  <BehaviorTree ID="calculate_path">
    <Fallback name="PATH CALCULATION">
      <Repeat name="LOOP" num_cycles="-1">
        <Sequence name="ACTIVITY">
          <Action name="UPDATE ACTIVITY" ID="UpdateCurrentGeneratedActivity" type="{type}" origin_sph="{origin_sph}" target_sph="{target_sph}" forbidden_atoms_out="{forbidden_atoms}"/>
          <Action name="PROJECT ORIGIN" ID="ProjectPointOnNetwork" point_in="{origin_sph}" point_out="{origin}"/>
          <Action name="PROJECT TARGET" ID="ProjectPointOnNetwork" point_in="{target_sph}" point_out="{target}"/>
          <Action name="CREATE PATH" ID="CreatePath" origin="{origin}" target="{target}" forbidden_atoms="{forbidden_atoms}" path="{path}"/>
          <Action name="AGREGATE PATH" ID="AgregatePath" path="{path}"/>
        </Sequence>
      </Repeat>
      <Condition name="MISSION FULLY TREATED" ID="MissionFullyTreated" type="{type}"/>
    </Fallback>
  </BehaviorTree>
  <BehaviorTree ID="execute">
    <ReactiveFallback name="EXECUTE">
      <Repeat name="STEP LOOP" num_cycles="-1">
        <Fallback name="MOTION SELECTOR">
          <SubTreePlus name="MOVE" ID="move" __autoremap="true"/>
          <SubTreePlus name="REACH AND STOP" ID="reach_and_stop" __autoremap="true"/>
          <SubTreePlus name="MOVE AND INSPECT" ID="move_and_inspect" __autoremap="true"/>
          <SubTreePlus name="REACH STOP INSPECTING" ID="reach_stop_inspecting" __autoremap="true"/>
        </Fallback>
      </Repeat>
      <Condition name="IS MISSION TERMINATED" ID="MissionTerminated"/>
    </ReactiveFallback>
  </BehaviorTree>
  <BehaviorTree ID="move">
    <Sequence name="MOVE">
      <Condition name="IS STEP MOVE" ID="CheckCurrentStepType" type_to_be_checked="0"/>
      <Action name="PASS MOTION PARAMS" ID="PassMotionParameters" motion_params="{motion_params}"/>
      <Action name="MOVE" ID="Move" threshold_type="1" motion_params="{motion_params}"/>
      <Action name="UPDATE STEP" ID="UpdateCurrentExecutedStep"/>
    </Sequence>
  </BehaviorTree>
  <BehaviorTree ID="reach_and_stop">
    <Sequence name="REACH AND STOP">
      <Condition name="IS STEP REACH STOP" ID="CheckCurrentStepType" type_to_be_checked="2"/>
      <Action name="PASS MOTION PARAMS" ID="PassMotionParameters" motion_params="{motion_params}"/>
      <Action name="MOVE AND STOP" ID="MoveAndStop" motion_params="{motion_params}"/>
      <Action name="SIGNAL AND WAIT" ID="SignalAndWaitForOrder" message="need authorization"/>
      <Action name="UPDATE STEP" ID="UpdateCurrentExecutedStep"/>
    </Sequence>
  </BehaviorTree>
  <BehaviorTree ID="move_and_inspect">
    <Sequence name="MOVE AND INSPECT">
      <Condition name="IS STEP MOVE INSPECT" ID="CheckCurrentStepType" type_to_be_checked="10"/>
      <Action name="PASS MOTION PARAMS" ID="PassMotionParameters" motion_params="{motion_params}"/>
      <Action name="PAUSE" ID="Pause" duration="2.0"/>
      <Action name="START INSPECTION" ID="ManageMeasurements"/>
      <Action name="MOVE" ID="Move" threshold_type="1" motion_params="{motion_params}"/>
      <Action name="UPDATE STEP" ID="UpdateCurrentExecutedStep"/>
    </Sequence>
  </BehaviorTree>
  <BehaviorTree ID="reach_stop_inspecting">
    <Sequence name="REACH STOP INSPECTING">
      <Condition name="IS STEP REACH STOP INSPECT" ID="CheckCurrentStepType" type_to_be_checked="12"/>
      <Action name="PASS MOTION PARAMS" ID="PassMotionParameters" motion_params="{motion_params}"/>
      <Action name="MOVE AND STOP" ID="MoveAndStop" motion_params="{motion_params}"/>
      <Action name="STOP INSPECTION" ID="ManageMeasurements"/>
      <Action name="ANALYSE" ID="AnalyseMeasurements"/>
      <Fallback name="QUALITY CHECK">
        <Condition name="QUALITY OK" ID="MeasurementsQualityValidated"/>
        <Action name="PASS DEFECTS" ID="PassDefectsLocalization" defects="{defects}"/>
      </Fallback>
      <Action name="GEN CORRECTIVE" ID="GenerateCorrectiveSubSequence" defects="{defects}"/>
      <Action name="INSERT CORRECTIVE" ID="InsertCorrectiveSubSequence"/>
      <Action name="UPDATE STEP" ID="UpdateCurrentExecutedStep"/>
    </Sequence>
  </BehaviorTree>
</root>"""

# Catégories de missions pour la variété
# Poids pour atteindre ~30% Transport, ~40% Inspect+ctrl, ~20% Inspect-ctrl
# Transport (6 templates): poids faibles → 30% total
# Inspect+ctrl (3 templates, incl. Mission complète): poids forts → 40% total
# Inspect-ctrl (2 templates): poids moyens → 20% total
MISSION_CATEGORIES = [
    # --- Transport (~30% cible) ---
    ("Navigation simple vers un point kilométrique (km {km})", 5),
    ("Navigation autorisée avec autorisation du poste de contrôle vers le km {km}", 5),
    ("Correction de trajectoire après détection d'anomalie au km {km}", 5),
    ("Mission simulation : test de déplacement entre km {km_start} et km {km_end}", 5),
    ("Déplacement avec arrêts multiples aux km {km_start}, {km_mid} et {km_end}", 5),
    ("Simple transport vers le km {km}. Pas de mesure à réaliser", 5),
    # --- Inspect+ctrl (~40% cible) ---
    ("Inspection des {element} entre le km {km_start} et le km {km_end}", 13),
    ("Inspection avec mesures renforcées des {element} au km {km}", 13),
    (
        "Mission complète : préparation, navigation autorisée et inspection des {element} entre km {km_start} et km {km_end}",
        14,
    ),
    # --- Inspect-ctrl (~20% cible) ---
    (
        "Inspection des {element} à la volée entre le km {km_start} et le km {km_end}, sans contrôle des mesures par le robot",
        10,
    ),
    (
        "Parcours d'acquisition des {element} entre km {km_start} et km {km_end}. Les mesures seront prises à la volée sans être contrôlées",
        10,
    ),
]

INSPECTION_ELEMENTS = [
    "rails",
    "traverses",
    "joints de dilatation",
    "aiguillages",
    "signaux lumineux",
    "caténaires",
    "ballast",
    "soudures",
    "attaches de rail",
    "éclisses",
]


def random_mission(seen: set | None = None, max_reroll: int = 200) -> str:
    """Génère une instruction de mission aléatoire, unique si `seen` est fourni."""
    templates = [t for t, _ in MISSION_CATEGORIES]
    weights = [w for _, w in MISSION_CATEGORIES]
    for _ in range(max_reroll):
        cat = random.choices(templates, weights=weights, k=1)[0]
        mission = cat.format(
            km=random.randint(1, 50),
            km_start=random.randint(1, 25),
            km_mid=random.randint(26, 35),
            km_end=random.randint(36, 50),
            element=random.choice(INSPECTION_ELEMENTS),
        )
        if seen is None or mission not in seen:
            if seen is not None:
                seen.add(mission)
            return mission
    # Fallback : retourner quand même (ne devrait pas arriver avec 12k+ combos)
    return mission


def classify_mission(instruction: str) -> str:
    """Classifie la mission et retourne les directives de motion subtrees."""
    text = instruction.lower()
    # Détection des négations
    no_measure = any(
        w in text
        for w in [
            "pas de mesure",
            "sans mesure",
            "pas d'inspection",
            "sans inspection",
            "simple transport",
            "pas de contrôle",
            "pas de controle",
        ]
    )
    # "correction" et "anomalie" seuls = transport correctif, pas inspection
    is_correction = any(w in text for w in ["correction", "corriger"])
    has_inspect = (
        not no_measure
        and not is_correction
        and any(
            w in text
            for w in [
                "inspection",
                "inspecter",
                "mesure",
                "mesurer",
                "mesures",
                "vérifier",
                "contrôler",
                "défaut",
            ]
        )
    )
    no_control = any(
        w in text
        for w in [
            "sans contrôle",
            "sans être contrôl",
            "à la volée",
            "sans controle",
            "sans etre control",
        ]
    )

    transport_subtrees = (
        "- move(type=0): PassMotionParameters + Move(threshold=1)\n"
        "- deccelerate(type=1): PassMotionParameters + Deccelerate\n"
        "- reach_and_stop(type=2): PassMotionParameters + MoveAndStop + SignalAndWaitForOrder\n"
        "- pass(type=3): PassMotionParameters + Move(threshold=3)\n"
        "- reach_stop_no_wait(type=4): PassMotionParameters + MoveAndStop"
    )

    if has_inspect and no_control:
        inspect_subtrees = (
            "\n- move_and_inspect(type=10): PassMotionParameters + Pause + ManageMeasurements(start) + Move\n"
            "- deccel_and_inspect(type=11): PassMotionParameters + Deccelerate\n"
            "- reach_stop_inspecting(type=12): PassMotionParameters + MoveAndStop + ManageMeasurements(stop)\n"
            "- pass_stop_inspecting(type=13): PassMotionParameters + Move(pass) + ManageMeasurements(stop)\n"
            "- reach_stop_inspect_no_wait(type=14): PassMotionParameters + MoveAndStop + ManageMeasurements(stop)"
        )
        return (
            f"TYPE: Inspection SANS controle (mesures a la volee).\n"
            f"MOTION SUBTREES REQUIS dans le Fallback MOTION SELECTOR :\n"
            f"{transport_subtrees}{inspect_subtrees}\n"
            f"PAS de AnalyseMeasurements ni MeasurementsQualityValidated."
        )
    elif has_inspect:
        inspect_subtrees = (
            "\n- move_and_inspect(type=10): PassMotionParameters + Pause + ManageMeasurements(start) + Move\n"
            "- deccel_and_inspect(type=11): PassMotionParameters + Deccelerate\n"
            "- reach_stop_inspecting(type=12): PassMotionParameters + MoveAndStop + ManageMeasurements(stop) + AnalyseMeasurements + Fallback(MeasurementsQualityValidated, PassDefectsLocalization) + GenerateCorrectiveSubSequence + InsertCorrectiveSubSequence\n"
            "- pass_stop_inspecting(type=13): PassMotionParameters + Move(pass) + ManageMeasurements(stop) + Fallback(AnalyseMeasurements, MeasurementsEnforcedValidated)\n"
            "- reach_stop_inspect_no_wait(type=14): comme type=12 sans SignalAndWaitForOrder"
        )
        return (
            f"TYPE: Inspection AVEC controle et verification qualite.\n"
            f"MOTION SUBTREES REQUIS dans le Fallback MOTION SELECTOR :\n"
            f"{transport_subtrees}{inspect_subtrees}"
        )
    else:
        return (
            f"TYPE: Transport simple SANS mesure.\n"
            f"MOTION SUBTREES REQUIS dans le Fallback MOTION SELECTOR :\n"
            f"{transport_subtrees}\n"
            f"PAS de subtrees inspection (types 10-14). PAS de ManageMeasurements."
        )


# ─── État du graphe ──────────────────────────────────────────────────────────


class GraphState(TypedDict):
    instruction: str
    xml_draft: str
    errors: Optional[str]
    iterations: int
    is_valid: bool
    score: float


# ─── Nœuds du graphe ─────────────────────────────────────────────────────────


def make_nodes(
    llm: ChatOpenAI, llm_creative: ChatOpenAI, seen_missions: set | None = None
):
    """Crée les fonctions nœuds du graphe avec les LLM injectés."""

    def node_generate_instruction(state: GraphState) -> dict:
        """Génère une instruction de mission aléatoire unique."""
        mission = random_mission(seen=seen_missions)
        return {
            "instruction": mission,
            "iterations": 0,
            "is_valid": False,
            "errors": None,
            "score": 0.0,
        }

    def node_generate_xml(state: GraphState) -> dict:
        """Génère le XML ou le corrige si des erreurs sont présentes."""
        instruction = state["instruction"]
        errors = state.get("errors")
        system_content = f"{SYSTEM_PROMPT}\n\n{SKILLS_DOC}\n\n{FEW_SHOT_EXAMPLE}"
        mission_guidance = classify_mission(instruction)

        if errors:
            # Tronquer le XML précédent pour rester dans la limite de tokens
            prev_xml = state.get("xml_draft", "")
            if len(prev_xml) > 1500:
                prev_xml = prev_xml[:1500] + "\n<!-- ... tronqué ... -->"
            user_prompt = (
                f"Mission : {instruction}\n{mission_guidance}\n\n"
                f"Ta précédente tentative :\n```xml\n{prev_xml}\n```\n\n"
                f"Erreur de validation :\n{errors}\n\n"
                f"Corrige le XML. Renvoie UNIQUEMENT le XML corrigé."
            )
        else:
            n_subtrees = len(mission_guidance.split("\n- ")) - 1
            # Directives de variété aléatoires
            variation_hints = random.choice(
                [
                    "Utilise des name= descriptifs et specifiques a cette mission.",
                    "Ajoute un commentaire XML en debut decrivant la mission et ses parametres.",
                    "Varie les durees de Pause si tu en utilises (entre 1.0 et 5.0 secondes).",
                    "Le message de SignalAndWaitForOrder doit etre specifique a cette mission.",
                    "Tu peux omettre pass(type=3) si la mission ne necessite pas de passage sans arret.",
                    "Tu peux omettre reach_stop_no_wait(type=4) si tous les arrets necessitent une attente.",
                    "Ajoute une Pause apres LoadMission pour simuler le chargement.",
                    "Utilise des noms de variables blackboard evocateurs pour cette mission.",
                ]
            )
            user_prompt = (
                f"Mission : {instruction}\n{mission_guidance}\n\n"
                f"Genere le XML complet. Tu DOIS inclure les {n_subtrees} motion subtrees "
                f"ci-dessus comme BehaviorTree separes, chacun reference par SubTreePlus "
                f"dans le Fallback MOTION SELECTOR.\n"
                f"{variation_hints}\n"
                f"Renvoie UNIQUEMENT le XML."
            )

        try:
            response = llm.invoke(
                [
                    SystemMessage(content=system_content),
                    HumanMessage(content=user_prompt),
                ]
            )
        except Exception as e:
            err_msg = str(e)
            if "maximum context length" in err_msg and errors:
                # Retry sans le XML précédent (trop long)
                user_prompt = (
                    f"Mission : {instruction}\n\n"
                    f"Erreur précédente : {errors}\n\n"
                    f"Génère un XML corrigé. Renvoie UNIQUEMENT le XML."
                )
                response = llm.invoke(
                    [
                        SystemMessage(content=system_content),
                        HumanMessage(content=user_prompt),
                    ]
                )
            else:
                raise

        # Nettoyer les balises markdown
        xml_clean = re.sub(r"```xml\n?", "", response.content)
        xml_clean = re.sub(r"```\n?", "", xml_clean).strip()

        return {"xml_draft": xml_clean, "iterations": state["iterations"] + 1}

    def node_validate_xml(state: GraphState) -> dict:
        """Valide le XML avec le validateur multi-niveaux NAV4RAIL + cohérence sémantique."""
        result = validate_bt(state["xml_draft"])
        errors = list(result.errors)

        # Vérification sémantique : les subtrees requis sont-ils présents ?
        guidance = classify_mission(state["instruction"])
        xml = state["xml_draft"]
        if (
            "Inspection AVEC controle" in guidance
            or "Inspection SANS controle" in guidance
        ):
            has_inspect_types = bool(re.search(r'type_to_be_checked="1[0-4]"', xml))
            has_manage = "ManageMeasurements" in xml
            if not has_inspect_types or not has_manage:
                errors.append(
                    "SEMANTIC: La mission demande de l'inspection mais le XML "
                    "ne contient pas de subtrees inspection (types 10-14 avec "
                    "ManageMeasurements). Ajoute les subtrees move_and_inspect, "
                    "reach_stop_inspecting, etc."
                )
        if "Inspection AVEC controle" in guidance:
            if "AnalyseMeasurements" not in xml:
                errors.append(
                    "SEMANTIC: Inspection avec controle requiert AnalyseMeasurements "
                    "et MeasurementsQualityValidated dans les subtrees type 12/14."
                )
        if "Transport simple SANS mesure" in guidance:
            if "ManageMeasurements" in xml:
                errors.append(
                    "SEMANTIC: Transport simple ne doit PAS contenir "
                    "ManageMeasurements ni de subtrees inspection."
                )

        is_valid = len(errors) == 0 and result.valid
        if is_valid:
            error_msg = None
        else:
            error_msg = " | ".join(errors)
            if result.warnings:
                error_msg += " | Warnings: " + " | ".join(result.warnings)

        return {
            "is_valid": result.valid,
            "errors": error_msg,
            "score": result.score,
        }

    return node_generate_instruction, node_generate_xml, node_validate_xml


def route_after_validation(state: GraphState) -> str:
    """Décide de la prochaine étape après validation."""
    if state["is_valid"]:
        return "success"
    elif state["iterations"] >= MAX_RETRIES:
        return "max_retries"
    else:
        return "retry"


# ─── Construction du graphe ──────────────────────────────────────────────────


def build_graph(
    llm: ChatOpenAI, llm_creative: ChatOpenAI, seen_missions: set | None = None
) -> StateGraph:
    gen_instr, gen_xml, validate = make_nodes(llm, llm_creative, seen_missions)

    workflow = StateGraph(GraphState)
    workflow.add_node("generate_instruction", gen_instr)
    workflow.add_node("generate_xml", gen_xml)
    workflow.add_node("validate_xml", validate)

    workflow.set_entry_point("generate_instruction")
    workflow.add_edge("generate_instruction", "generate_xml")
    workflow.add_edge("generate_xml", "validate_xml")
    workflow.add_conditional_edges(
        "validate_xml",
        route_after_validation,
        {"success": END, "max_retries": END, "retry": "generate_xml"},
    )
    return workflow.compile()


# ─── Boucle de génération ────────────────────────────────────────────────────


def generate_dataset(app, count: int, output_path: Path, resume: bool = False) -> dict:
    """Génère `count` échantillons et les écrit en JSONL."""
    # Support reprise : compter les samples existants
    start = 0
    samples = []
    if resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
                    start += 1
        print(
            f"\u21bb Reprise : {start} samples existants, génération de {count - start} restants"
        )
        if start >= count:
            print("Déjà complet !")
            return {"total": count, "valid": start, "failed": 0, "retries": 0}

    stats = {"total": count, "valid": len(samples), "failed": 0, "retries": 0}
    mode = "a" if resume and start > 0 else "w"

    with open(output_path, mode, encoding="utf-8") as f:
        for i in range(start, count):
            initial = {
                "instruction": "",
                "xml_draft": "",
                "errors": None,
                "iterations": 0,
                "is_valid": False,
                "score": 0.0,
            }

            # Retry avec backoff en cas de déconnexion réseau
            result = None
            for attempt in range(5):
                try:
                    result = app.invoke(initial)
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    if (
                        "connection" in err_str
                        or "timeout" in err_str
                        or "refused" in err_str
                    ):
                        import time

                        wait = 30 * (attempt + 1)
                        print(
                            f"  ⚠ Erreur réseau (tentative {attempt + 1}/5), retry dans {wait}s: {str(e)[:80]}"
                        )
                        time.sleep(wait)
                    else:
                        raise

            if result is None:
                stats["failed"] += 1
                print(f"[{i + 1}/{count}] ❌ Échec réseau après 5 tentatives")
                continue

            if result["is_valid"]:
                stats["valid"] += 1
                instruction = f"{SYSTEM_PROMPT}\n\n{SKILLS_DOC}\n\nMission : {result['instruction']}"
                entry = {
                    "mission": result["instruction"],
                    "xml": result["xml_draft"],
                    "prompt": f"<s>[INST] {instruction} [/INST] {result['xml_draft']} </s>",
                    "score": result["score"],
                    "iterations": result["iterations"],
                }
                samples.append(entry)
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                status = f"✅ score={result['score']:.1f} ({result['iterations']} iter)"
            else:
                stats["failed"] += 1
                status = f"❌ après {result['iterations']} tentatives: {result['errors'][:80]}"

            stats["retries"] += max(0, result["iterations"] - 1)
            print(f"[{i + 1}/{count}] {status}  —  {result['instruction'][:60]}")

    # Sauver aussi en JSON
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    return stats


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Génération dataset NAV4RAIL via LLM + LangGraph"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/v1",
        help="URL de l'API vLLM (default: http://localhost:8000/v1)",
    )
    parser.add_argument("--model", default=MODEL_ID, help="Model ID")
    parser.add_argument("--count", type=int, default=50, help="Nombre d'échantillons")
    parser.add_argument("--output", default=None, help="Fichier de sortie JSONL")
    parser.add_argument("--seed", type=int, default=42, help="Seed aléatoire")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprendre la génération depuis le fichier existant",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        args.output = f"dataset_nav4rail_llm_{args.count}_{ts}.jsonl"

    # Déduplication : charger les missions existantes si reprise
    seen_missions = set()
    if args.resume and Path(args.output).exists():
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    seen_missions.add(json.loads(line)["mission"])
        print(f"🔑 {len(seen_missions)} missions uniques chargées pour dédup")

    print(f"🔧 Config: url={args.url} model={args.model} count={args.count}")
    print(f"📁 Output: {args.output}")

    llm = ChatOpenAI(
        model=args.model,
        openai_api_base=args.url,
        openai_api_key="sk-no-key-required",
        temperature=0.6,
        max_tokens=4096,
    )
    llm_creative = ChatOpenAI(
        model=args.model,
        openai_api_base=args.url,
        openai_api_key="sk-no-key-required",
        temperature=0.7,
        max_tokens=256,
    )

    app = build_graph(llm, llm_creative, seen_missions)
    stats = generate_dataset(app, args.count, Path(args.output), resume=args.resume)

    print(f"\n{'=' * 60}")
    print(
        f"📊 Résultats: {stats['valid']}/{stats['total']} valides "
        f"({stats['valid'] / stats['total'] * 100:.0f}%)"
    )
    print(f"   Échecs: {stats['failed']} | Retries: {stats['retries']}")
    print(f"📁 Sauvé: {args.output} + {Path(args.output).with_suffix('.json')}")


if __name__ == "__main__":
    main()
