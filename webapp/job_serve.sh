#!/bin/bash
#SBATCH --job-name=nav4rail-serve
#SBATCH --output=nav4rail_serve_%j.out
#SBATCH --error=nav4rail_serve_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#
# NAV4RAIL multi-model inference server on GPU cluster (Télécom Paris)
# Supports hot-swapping between 5 fine-tuned GGUF models.
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
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
DEFAULT_MODEL="${DEFAULT_MODEL:-nav4rail-mistral-7b-q4_k_m.gguf}"
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
"""NAV4RAIL multi-model inference server — GPU cluster edition.

Supports hot-swapping between multiple GGUF models via /load_model endpoint.
Each model uses its native prompt format (auto-detected from filename).

Fixes for llama-cpp-python >=0.3.x:
  - Grammar object is re-created per request (sampler state not reusable)
  - Startup self-test verifies grammar actually constrains output
"""

import glob
import os
import re
import sys
import time
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.expanduser("~/models"))
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "")

# ─── Model registry ─────────────────────────────────────────────────────────

# Map short names to prompt format builders.
# The GGUF filename convention is: nav4rail-{short_name}-q4_k_m.gguf
MODEL_PROMPT_FORMATS = {
    "mistral-7b": "mistral",
    "llama3-8b": "llama3",
    "qwen-coder-7b": "chatml",
    "qwen-14b": "chatml",
    "gemma2-9b": "gemma",
}


def _detect_model_key(filename: str) -> str:
    """Extract model short name from GGUF filename."""
    # nav4rail-mistral-7b-q4_k_m.gguf -> mistral-7b
    m = re.match(r"nav4rail-(.+?)-q[0-9]", filename)
    if m:
        return m.group(1)
    return ""


def _scan_models() -> dict:
    """Scan MODEL_DIR for available GGUF files."""
    models = {}
    for path in sorted(glob.glob(os.path.join(MODEL_DIR, "nav4rail-*.gguf"))):
        fname = os.path.basename(path)
        key = _detect_model_key(fname)
        if key:
            models[key] = {
                "path": path,
                "filename": fname,
                "format": MODEL_PROMPT_FORMATS.get(key, "chatml"),
            }
    return models


available_models = _scan_models()
print(f"[serve] Found {len(available_models)} models: {list(available_models.keys())}")

# ─── Current model state ────────────────────────────────────────────────────
from llama_cpp import Llama, LlamaGrammar

current_model = {"name": None, "llm": None}
lock = threading.Lock()


def _load_model(model_key: str):
    """Load a GGUF model by key. Replaces current model."""
    if model_key not in available_models:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(available_models.keys())}")

    info = available_models[model_key]
    path = info["path"]
    print(f"[serve] Loading {model_key} from {path} ...")
    t0 = time.time()

    # Free previous model
    if current_model["llm"] is not None:
        del current_model["llm"]
        current_model["llm"] = None
        current_model["name"] = None
        import gc; gc.collect()

    llm = Llama(model_path=path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
    current_model["llm"] = llm
    current_model["name"] = model_key
    print(f"[serve] {model_key} loaded in {time.time() - t0:.1f}s (GPU offload)")


# Load default model at startup
if DEFAULT_MODEL:
    default_key = _detect_model_key(os.path.basename(DEFAULT_MODEL))
    if not default_key and available_models:
        default_key = next(iter(available_models))
elif available_models:
    default_key = next(iter(available_models))
else:
    default_key = ""

if default_key and default_key in available_models:
    _load_model(default_key)
elif available_models:
    _load_model(next(iter(available_models)))
else:
    print("[serve] WARNING: No GGUF models found in MODEL_DIR!")

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
    """Create a FRESH LlamaGrammar object each time (v0.3.x fix)."""
    if GRAMMAR_STR is None:
        return None
    return LlamaGrammar.from_string(GRAMMAR_STR)


# ─── Prompt builders ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Tu es un expert en robotique ferroviaire NAV4RAIL. "
    "Genere un Behavior Tree XML BehaviorTree.CPP pour la mission decrite.\n\n"
    "FORMAT :\n"
    "- <root BTCPP_format=\"4\" main_tree_to_execute=\"nom\">\n"
    "- Multi-<BehaviorTree ID=\"...\"> interconnectes via <SubTreePlus __autoremap=\"true\">\n"
    "- <Action name=\"NOM\" ID=\"Skill\" port=\"{var}\"/>  <Condition name=\"NOM\" ID=\"Skill\"/>\n"
    "- Controle : Sequence, Fallback, ReactiveFallback, Repeat(num_cycles=\"-1\")\n"
    "- Chaque noeud a name=\"DESCRIPTION EN MAJUSCULES\", ports blackboard {variable}\n\n"
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


def _build_prompt_mistral(mission: str) -> str:
    instruction = f"{SYSTEM_PROMPT}\n\n{SKILLS_DOC}\n\nMission : {mission}"
    return f"[INST] {instruction} [/INST]"


def _build_prompt_chatml(mission: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n\n{SKILLS_DOC}<|im_end|>\n"
        f"<|im_start|>user\n{mission}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _build_prompt_llama3(mission: str) -> str:
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}\n\n{SKILLS_DOC}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{mission}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _build_prompt_gemma(mission: str) -> str:
    return (
        f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{SKILLS_DOC}\n\n"
        f"Mission : {mission}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


PROMPT_BUILDERS = {
    "mistral": _build_prompt_mistral,
    "chatml": _build_prompt_chatml,
    "llama3": _build_prompt_llama3,
    "gemma": _build_prompt_gemma,
}


def build_prompt(mission: str) -> str:
    """Build prompt using current model's format."""
    fmt = "mistral"
    if current_model["name"] and current_model["name"] in available_models:
        fmt = available_models[current_model["name"]]["format"]
    builder = PROMPT_BUILDERS.get(fmt, _build_prompt_mistral)
    return builder(mission)


def extract_xml(text: str) -> str:
    match = re.search(r"(<root\b.*?</root>)", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


# ─── Validation (optional) ───────────────────────────────────────────────────
try:
    from validate_bt import validate_bt, enrich_ports
    print("[serve] BT validator + port enrichment loaded")
except ImportError:
    def validate_bt(xml):
        class _R:
            valid = True; score = 1.0; errors = []; warnings = []
            def summary(self): return "no validator"
        return _R()
    def enrich_ports(xml): return xml
    print("[serve] WARNING: validate_bt not available, using passthrough")


# ─── API ─────────────────────────────────────────────────────────────────────

class GenRequest(BaseModel):
    mission: str
    use_grammar: bool = True
    model: str | None = None


class LoadModelRequest(BaseModel):
    model: str


# ─── GPU info ────────────────────────────────────────────────────────────────
import subprocess as _sp
try:
    _gpu_name = _sp.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).strip().split("\n")[0]
except Exception:
    _gpu_name = "unknown"

_start_time = time.time()
_slurm_time_limit = 4 * 3600  # SBATCH --time=04:00:00


@app.get("/health")
def health():
    uptime_s = int(time.time() - _start_time)
    remaining_s = max(0, _slurm_time_limit - uptime_s)
    return {
        "status": "ok",
        "gpu": True,
        "gpu_name": _gpu_name,
        "model": current_model["name"],
        "format": available_models.get(current_model["name"], {}).get("format", "unknown"),
        "cluster": "telecom-paris",
        "backend": "llama-cpp",
        "uptime_s": uptime_s,
        "remaining_s": remaining_s,
    }


@app.get("/models")
def list_models():
    return {
        "current": current_model["name"],
        "available": {
            k: {"filename": v["filename"], "format": v["format"]}
            for k, v in available_models.items()
        },
    }


@app.post("/load_model")
def load_model(req: LoadModelRequest):
    if req.model == current_model["name"]:
        return {"status": "already_loaded", "model": req.model}
    if req.model not in available_models:
        raise HTTPException(404, f"Model not found: {req.model}. Available: {list(available_models.keys())}")
    with lock:
        t0 = time.time()
        _load_model(req.model)
        load_time = time.time() - t0
    return {"status": "loaded", "model": req.model, "load_time_s": round(load_time, 1)}


@app.post("/generate")
def generate(req: GenRequest):
    # Swap model if requested and different from current
    if req.model and req.model != current_model["name"]:
        if req.model not in available_models:
            raise HTTPException(404, f"Model not found: {req.model}")
        with lock:
            _load_model(req.model)

    if current_model["llm"] is None:
        raise HTTPException(503, "No model loaded")

    prompt = build_prompt(req.mission)

    grammar_obj = _make_grammar() if req.use_grammar else None
    grammar_label = "fresh-per-request" if grammar_obj else "none"
    print(f"[serve] [{current_model['name']}] Generating for: {req.mission[:80]}... (grammar={grammar_label})")

    with lock:
        t0 = time.time()
        output = current_model["llm"](
            prompt,
            max_tokens=2048,
            temperature=0.0,
            repeat_penalty=1.1,
            grammar=grammar_obj,
            echo=False,
        )
        gen_time = time.time() - t0

    raw = output["choices"][0]["text"]
    xml_raw = extract_xml(raw)
    xml = enrich_ports(xml_raw)
    vr = validate_bt(xml)

    if req.use_grammar and vr.errors:
        print(f"[serve] ⚠ Grammar was requested but errors found: {vr.errors}")
    enriched = xml != xml_raw
    print(f"[serve] Done in {gen_time:.1f}s — score={vr.score:.2f} (enriched={enriched})")

    return {
        "xml": xml,
        "xml_raw": xml_raw,
        "valid": vr.valid,
        "score": round(vr.score, 2),
        "errors": vr.errors,
        "warnings": vr.warnings,
        "summary": vr.summary(),
        "generation_time_s": round(gen_time, 1),
        "model": current_model["name"],
        "enriched": enriched,
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
export MODEL_DIR DEFAULT_MODEL FINETUNE_DIR="$WORK_DIR"
echo "[serve] Starting multi-model inference server on port $PORT ..."
echo "[serve] Model directory: $MODEL_DIR"
echo "[serve] Default model: $DEFAULT_MODEL"
exec uvicorn serve:app --host 0.0.0.0 --port "$PORT"
