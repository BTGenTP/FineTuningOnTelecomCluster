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
    "Navigue jusqu'au km 42 depuis le km 10",
    "Inspecte la voie entre le km 5 et le km 15 avec analyse qualité",
    "Effectue des mesures de géométrie de voie au point PK30",
    "Navigue vers le dépôt principal avec autorisation préalable",
    "Inspecte les rails entre km 20 et km 35 et corrige les défauts détectés",
    "Simule une inspection reach-stop de la section critique avec analyse corrective",
    "Déplace-toi en mode haute sécurité avec arrêt et signal à chaque segment",
    "Effectue une patrouille d'inspection entre km 0 et km 25 avec validation stricte",
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

    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int | None = None):
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

    def generate(self, mission: str, use_grammar: bool = True,
                 max_tokens: int = 800) -> dict:
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
