"""
NAV4RAIL BT Generator — GGUF inference engine.

Loads a quantized Mistral-7B GGUF model via llama-cpp-python and generates
Behavior Tree XML from natural language mission descriptions.
"""

import os
import re
import sys
import threading
import time
from pathlib import Path

# Import validation and grammar from finetune/
FINETUNE_DIR = Path(__file__).resolve().parent / "finetune"
sys.path.insert(0, str(FINETUNE_DIR))

from validate_bt import validate_bt  # noqa: E402
from nav4rail_grammar import NAV4RAIL_GBNF  # noqa: E402

# ─── Prompt components (from finetune_lora_xml.py) ───────────────────────────

SYSTEM_PROMPT = (
    "Tu es un expert en robotique ferroviaire NAV4RAIL. "
    "Génère un Behavior Tree au format XML BehaviorTree.CPP v4 "
    "correspondant exactement à la mission décrite. "
    "Utilise uniquement les skills du catalogue fourni. "
    "Réponds uniquement avec le XML, sans explication."
)

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

TEST_MISSIONS = [
    # transport_simple
    "Navigation simple vers un point kilométrique (km 21)",
    # inspect-ctrl (diverse inspection types)
    "Inspection des rails entre le km 12 et le km 45",
    "Inspection des ballast entre le km 7 et le km 46",
    "Inspection avec mesures renforcées des attaches de rail au km 30",
    "Mission complète : préparation, navigation autorisée et inspection des soudures entre km 5 et km 25",
    # transport_arrêts_multiples
    "Déplacement avec arrêts multiples aux km 3, 35 et 42",
    # correction_anomalie
    "Correction de trajectoire après détection d'anomalie au km 26",
    # simulation
    "Mission simulation : test de déplacement entre km 19 et km 40",
    # inspect+ctrl
    "Navigation autorisée avec autorisation du poste de contrôle vers le km 15",
    # transport with measurements
    "Parcours d'acquisition des caténaires entre km 21 et km 36. Les mesures seront prises à la volée sans être contrôlées",
]


def _build_prompt(mission: str) -> str:
    instruction = f"{SYSTEM_PROMPT}\n\n{SKILLS_DOC}\n\nMission : {mission}"
    # No <s> prefix — llama-cpp adds BOS token automatically
    return f"[INST] {instruction} [/INST]"


def _extract_xml(text: str) -> str:
    """Extract the <root>...</root> block from model output."""
    match = re.search(r"(<root\b.*?</root>)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


class Nav4RailGenerator:
    """GGUF-based BT generator with optional GBNF constrained decoding."""

    def __init__(
        self, model_path: str, n_ctx: int = 2048, n_threads: int | None = None
    ):
        from llama_cpp import Llama, LlamaGrammar

        self._lock = threading.Lock()
        self._n_threads = n_threads or os.cpu_count() or 4

        print(f"[inference] Loading GGUF model: {model_path}")
        print(f"[inference] n_ctx={n_ctx}, n_threads={self._n_threads}")
        t0 = time.time()

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=self._n_threads,
            verbose=False,
        )

        self._grammar = LlamaGrammar.from_string(NAV4RAIL_GBNF)
        print(f"[inference] Model loaded in {time.time() - t0:.1f}s")

    def generate(
        self, mission: str, use_grammar: bool = True, max_tokens: int = 800
    ) -> dict:
        prompt = _build_prompt(mission)

        with self._lock:
            t0 = time.time()
            output = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                repeat_penalty=1.1,
                grammar=self._grammar if use_grammar else None,
                echo=False,
            )
            gen_time = time.time() - t0

        raw_text = output["choices"][0]["text"]
        xml = _extract_xml(raw_text)

        vr = validate_bt(xml)

        return {
            "xml": xml,
            "valid": vr.valid,
            "score": round(vr.score, 2),
            "errors": vr.errors,
            "warnings": vr.warnings,
            "summary": vr.summary(),
            "generation_time_s": round(gen_time, 1),
        }

    @property
    def loaded(self) -> bool:
        return self._llm is not None
