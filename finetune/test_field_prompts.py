"""Test du pipeline LangGraph avec les prompts métier réels."""

import json
import re
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

sys.path.insert(0, str(Path(__file__).parent))
from validate_bt import validate_bt
from generate_dataset_llm import (
    SYSTEM_PROMPT,
    SKILLS_DOC,
    FEW_SHOT_EXAMPLE,
    MODEL_ID,
    MAX_RETRIES,
    GraphState,
    route_after_validation,
    make_nodes,
    classify_mission,
)


def build_test_graph(llm, llm_creative):
    """Graphe qui skip generate_instruction — instruction fournie dans l'état initial."""
    _, gen_xml, validate = make_nodes(llm, llm_creative)
    workflow = StateGraph(GraphState)
    workflow.add_node("generate_xml", gen_xml)
    workflow.add_node("validate_xml", validate)
    workflow.set_entry_point("generate_xml")
    workflow.add_edge("generate_xml", "validate_xml")
    workflow.add_conditional_edges(
        "validate_xml",
        route_after_validation,
        {"success": END, "max_retries": END, "retry": "generate_xml"},
    )
    return workflow.compile()


# ─── Prompts métier terrain ──────────────────────────────────────────────────

FIELD_PROMPTS = [
    # P1 — Transport simple
    "Réaliser le parcours décrit dans le fichier de mission. "
    "Il s'agit d'un simple transport. Pas de mesure à réaliser.",
    # P2 — Inspection avec re-mesure corrective
    "A partir d'un fichier descriptif de la mission, générer une tournée d'inspection. "
    "Le robot doit vérifier les mesures. Si des problèmes sont rencontrés, "
    "le robot doit remesurer les zones problématiques à vitesse réduite.",
    # P3 — Inspection sans contrôle
    "A partir d'un fichier descriptif de la mission, générer une tournée d'inspection. "
    "Les mesures seront prises à la volée sans être contrôlées par le robot.",
]

FIELD_LABELS = [
    "Transport simple",
    "Inspection + re-mesure corrective",
    "Inspection sans contrôle",
]

URL = "http://localhost:8001/v1"


def run_prompt(app, instruction: str) -> dict:
    """Exécute le graph avec une instruction fixée."""
    initial = {
        "instruction": instruction,
        "xml_draft": "",
        "errors": None,
        "iterations": 0,
        "is_valid": False,
        "score": 0.0,
    }
    return app.invoke(initial)


def analyze(xml: str) -> dict:
    """Analyse structurelle d'un XML généré."""
    subtrees = re.findall(r'<BehaviorTree ID="([^"]+)"', xml)
    motion_types = re.findall(r'type_to_be_checked="(\d+)"', xml)
    skills = set(re.findall(r'ID="([A-Z][a-zA-Z]+)"', xml))
    has_inspect = "ManageMeasurements" in xml
    has_analyse = "AnalyseMeasurements" in xml
    has_quality = "MeasurementsQualityValidated" in xml
    has_corrective = "GenerateCorrectiveSubSequence" in xml
    has_signal = "SignalAndWaitForOrder" in xml
    has_decel = "Deccelerate" in xml
    has_pause = "Pause" in xml
    has_enforced = "MeasurementsEnforcedValidated" in xml
    return {
        "subtrees": subtrees,
        "n_subtrees": len(subtrees),
        "motion_types": motion_types,
        "skills": skills,
        "inspect": has_inspect,
        "analyse": has_analyse,
        "quality_check": has_quality,
        "corrective": has_corrective,
        "signal": has_signal,
        "decel": has_decel,
        "pause": has_pause,
        "enforced": has_enforced,
        "xml_len": len(xml),
    }


def main():
    llm = ChatOpenAI(
        model=MODEL_ID,
        openai_api_base=URL,
        openai_api_key="sk-no-key-required",
        temperature=0.1,
        max_tokens=4096,
    )
    llm_creative = ChatOpenAI(
        model=MODEL_ID,
        openai_api_base=URL,
        openai_api_key="sk-no-key-required",
        temperature=0.7,
        max_tokens=256,
    )
    app = build_test_graph(llm, llm_creative)

    results = []
    for i, (prompt, label) in enumerate(zip(FIELD_PROMPTS, FIELD_LABELS)):
        print(f"\n{'=' * 70}")
        print(f"PROMPT {i + 1}: [{label}]")
        print(f'  "{prompt[:90]}..."')
        print(f"{'=' * 70}")

        result = run_prompt(app, prompt)
        vr = validate_bt(result["xml_draft"])
        a = analyze(result["xml_draft"])

        print(
            f"\n  Score: {vr.score}  Valid: {vr.valid}  Iterations: {result['iterations']}"
        )
        print(f"  Subtrees ({a['n_subtrees']}): {a['subtrees']}")
        print(f"  Motion types: {a['motion_types']}")
        print(
            f"  Inspect={a['inspect']} Analyse={a['analyse']} QualityCheck={a['quality_check']} Corrective={a['corrective']}"
        )
        print(
            f"  Signal={a['signal']} Decel={a['decel']} Pause={a['pause']} Enforced={a['enforced']}"
        )
        if vr.errors:
            print(f"  ERRORS: {vr.errors}")
        if vr.warnings:
            print(f"  WARNINGS: {vr.warnings}")

        print(f"\n--- XML ({a['xml_len']} chars) ---")
        print(result["xml_draft"])

        results.append(
            {
                "label": label,
                "prompt": prompt,
                "xml": result["xml_draft"],
                "score": vr.score,
                "valid": vr.valid,
                "iterations": result["iterations"],
                "analysis": {
                    k: v if not isinstance(v, set) else list(v) for k, v in a.items()
                },
                "errors": vr.errors,
                "warnings": vr.warnings,
            }
        )

    # Sauvegarder
    with open("test_field_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print("RÉSUMÉ")
    print(f"{'=' * 70}")
    for r in results:
        print(
            f"  [{r['label']}] score={r['score']} valid={r['valid']} iters={r['iterations']}"
        )


if __name__ == "__main__":
    main()
