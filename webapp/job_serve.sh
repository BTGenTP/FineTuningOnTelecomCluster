#!/bin/bash
#SBATCH --job-name=nav4rail-serve
#SBATCH --output=nav4rail_serve_%j.out
#SBATCH --error=nav4rail_serve_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#
# NAV4RAIL inference server on GPU cluster (Télécom Paris)
# Reverse SSH tunnel to gpu-gw so RPi5 can reach it.
#
# Usage:
#   sbatch job_serve.sh
#   # RPi5 connects via: ssh -L 8080:localhost:8080 gpu
#

set -euo pipefail

module load python/3.11.13 cuda/12.4.1 cmake/4.1.0 gcc/11.5.0 || true

echo "[serve] host=$(hostname) jobid=${SLURM_JOB_ID:-unknown} date=$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

WORK_DIR="${WORK_DIR:-$HOME/nav4rail_serve}"
VENV_DIR="${VENV_DIR:-$WORK_DIR/venv_gpu_final}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/nav4rail-mistral-7b-q4_k_m.gguf}"
PORT="${PORT:-8080}"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# ─── Virtual environment ─────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "[serve] Creating venv..."
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Check llama-cpp-python is installed with CUDA
python3 -c "from llama_cpp import Llama; print('[serve] llama-cpp-python OK')" 2>/dev/null || {
  echo "[serve] ERROR: llama-cpp-python not installed in venv. Build it first with CUDA support."
  exit 1
}

pip install -q fastapi uvicorn 2>/dev/null || true

# ─── Inference server script ─────────────────────────────────────────────────
cat > "$WORK_DIR/serve.py" << 'SERVEPY'
"""NAV4RAIL inference server — GPU cluster edition.

Fixes for llama-cpp-python >=0.3.x:
  - Grammar object is re-created per request (sampler state not reusable)
  - Startup self-test verifies grammar actually constrains output
"""

import os
import re
import sys
import time
import threading

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

MODEL_PATH = os.environ["MODEL_PATH"]

# ─── Load model ──────────────────────────────────────────────────────────────
from llama_cpp import Llama, LlamaGrammar

print(f"[serve] Loading {MODEL_PATH} ...")
t0 = time.time()
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)
print(f"[serve] Model loaded in {time.time() - t0:.1f}s (GPU offload)")

# ─── Load GBNF grammar string ───────────────────────────────────────────────
GRAMMAR_STR = None
FINETUNE_DIR = os.environ.get("FINETUNE_DIR", ".")
sys.path.insert(0, FINETUNE_DIR)

try:
    from nav4rail_grammar import NAV4RAIL_GBNF
    GRAMMAR_STR = NAV4RAIL_GBNF
    print("[serve] GBNF grammar string loaded from nav4rail_grammar module")
except ImportError:
    gbnf_path = os.environ.get("GBNF_PATH", "")
    if gbnf_path and os.path.exists(gbnf_path):
        GRAMMAR_STR = open(gbnf_path).read()
        print(f"[serve] GBNF grammar string loaded from {gbnf_path}")
    else:
        print("[serve] WARNING: No GBNF grammar available — generation will be unconstrained!")


def _make_grammar():
    """Create a FRESH LlamaGrammar object each time.

    In llama-cpp-python >=0.3.x, the grammar wraps a llama_sampler that
    maintains internal state.  Re-using the same object across calls can
    cause the grammar to be silently ignored after the first generation.
    """
    if GRAMMAR_STR is None:
        return None
    return LlamaGrammar.from_string(GRAMMAR_STR)


# ─── Startup grammar self-test ──────────────────────────────────────────────
VALID_SKILLS = {
    "LoadMission", "MissionStructureValid", "UpdateCurrentGeneratedActivity",
    "ProjectPointOnNetwork", "CreatePath", "AgregatePath", "MissionFullyTreated",
    "PassAdvancedPath", "PassMission", "GenerateMissionSequence",
    "GenerateCorrectiveSubSequence", "InsertCorrectiveSubSequence",
    "MissionTerminated", "CheckCurrentStepType", "PassMotionParameters",
    "Move", "UpdateCurrentExecutedStep", "Deccelerate", "MoveAndStop",
    "SignalAndWaitForOrder", "IsRobotPoseProjectionActive",
    "ManageMeasurements", "AnalyseMeasurements", "MeasurementsQualityValidated",
    "PassDefectsLocalization", "MeasurementsEnforcedValidated", "SimulationStarted",
}

if GRAMMAR_STR:
    print("[serve] Running grammar self-test...")
    try:
        test_grammar = _make_grammar()
        test_out = llm(
            "[INST] Génère un Behavior Tree simple avec LoadMission et Move. [/INST]",
            max_tokens=200,
            temperature=0.0,
            grammar=test_grammar,
            echo=False,
        )
        test_text = test_out["choices"][0]["text"]
        # Check that output starts with <root (grammar forces this)
        if test_text.strip().startswith("<root"):
            # Extract skill tags and verify they're all valid
            found_tags = set(re.findall(r'<(\w+)\s+name=', test_text))
            control_tags = {"Sequence", "Fallback"}
            skill_tags = found_tags - control_tags
            invalid = skill_tags - VALID_SKILLS
            if invalid:
                print(f"[serve] ⚠ GRAMMAR NOT ENFORCED — hallucinated tags: {invalid}")
                print(f"[serve] ⚠ Output was: {test_text[:300]}")
                print("[serve] ⚠ Will still attempt per-request grammar, but results may be unconstrained")
            else:
                print(f"[serve] ✓ Grammar self-test PASSED — skills found: {skill_tags}")
        else:
            print(f"[serve] ⚠ Grammar self-test: output doesn't start with <root>")
            print(f"[serve] ⚠ Output was: {test_text[:300]}")
    except Exception as e:
        print(f"[serve] ⚠ Grammar self-test failed with error: {e}")


lock = threading.Lock()

# ─── Prompt builder ──────────────────────────────────────────────────────────
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


def build_prompt(mission: str) -> str:
    instruction = f"{SYSTEM_PROMPT}\n\n{SKILLS_DOC}\n\nMission : {mission}"
    return f"[INST] {instruction} [/INST]"


def extract_xml(text: str) -> str:
    match = re.search(r"(<root\b.*?</root>)", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


# ─── Validation (optional) ───────────────────────────────────────────────────
try:
    from validate_bt import validate_bt
    print("[serve] BT validator loaded")
except ImportError:
    def validate_bt(xml):
        class _R:
            valid = True; score = 1.0; errors = []; warnings = []
            def summary(self): return "no validator"
        return _R()
    print("[serve] WARNING: validate_bt not available, using passthrough")


# ─── API ─────────────────────────────────────────────────────────────────────

class GenRequest(BaseModel):
    mission: str
    use_grammar: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "gpu": True, "model": os.path.basename(MODEL_PATH)}


@app.post("/generate")
def generate(req: GenRequest):
    prompt = build_prompt(req.mission)

    # Create a FRESH grammar object per request (v0.3.x fix)
    grammar_obj = _make_grammar() if req.use_grammar else None
    grammar_label = "fresh-per-request" if grammar_obj else "none"
    print(f"[serve] Generating for: {req.mission[:80]}... (grammar={grammar_label})")

    with lock:
        t0 = time.time()
        output = llm(
            prompt,
            max_tokens=800,
            temperature=0.0,
            repeat_penalty=1.1,
            grammar=grammar_obj,
            echo=False,
        )
        gen_time = time.time() - t0

    raw = output["choices"][0]["text"]
    xml = extract_xml(raw)
    vr = validate_bt(xml)

    # Log grammar effectiveness
    if req.use_grammar and vr.errors:
        print(f"[serve] ⚠ Grammar was requested but errors found: {vr.errors}")
    print(f"[serve] Done in {gen_time:.1f}s — score={vr.score:.2f}")

    return {
        "xml": xml,
        "valid": vr.valid,
        "score": round(vr.score, 2),
        "errors": vr.errors,
        "warnings": vr.warnings,
        "summary": vr.summary(),
        "generation_time_s": round(gen_time, 1),
    }
SERVEPY

# ─── Copy finetune modules if available ──────────────────────────────────────
FINETUNE_SRC="${FINETUNE_SRC:-$HOME/Telecom_Projet_fil_rouge/webapp/finetune}"
if [ -d "$FINETUNE_SRC" ]; then
  cp -u "$FINETUNE_SRC/nav4rail_grammar.py" "$WORK_DIR/" 2>/dev/null || true
  cp -u "$FINETUNE_SRC/validate_bt.py" "$WORK_DIR/" 2>/dev/null || true
  echo "[serve] Copied grammar + validator from $FINETUNE_SRC"
fi

# ─── Reverse SSH tunnel to gpu-gw ───────────────────────────────────────────
echo "[serve] Opening reverse SSH tunnel to gpu-gw:${PORT}..."
ssh -fN -R "${PORT}:localhost:${PORT}" gpu-gw \
  -o StrictHostKeyChecking=no \
  -o ServerAliveInterval=30 \
  -o ExitOnForwardFailure=yes \
  && echo "[serve] Tunnel active: gpu-gw:${PORT} → $(hostname):${PORT}" \
  || echo "[serve] WARNING: tunnel failed — manual setup needed"

# ─── Launch ──────────────────────────────────────────────────────────────────
export MODEL_PATH FINETUNE_DIR="$WORK_DIR"
echo "[serve] Starting inference server on port $PORT ..."
exec uvicorn serve:app --host 0.0.0.0 --port "$PORT"
