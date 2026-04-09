#!/bin/bash
#SBATCH --job-name=nav4rail-hf-mistral
#SBATCH --output=nav4rail_serve_%j.out
#SBATCH --error=nav4rail_serve_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#
# NAV4RAIL — Serve fine-tuned Mistral 7B (merged fp16, NOT GGUF)
# Uses HuggingFace transformers on RTX 3090 (24 GB VRAM).
#
# Pre-requisite: copy LoRA adapters to ~/nav4rail_adapters/mistral_7b/
#   scp -r rpi5:~/nav4rail_adapters_final/mistral_7b ~/nav4rail_adapters/mistral_7b
#
# Usage:
#   sbatch job_serve_hf_mistral.sh
#

set -euo pipefail

module load python/3.11.13 cuda/12.4.1 gcc/11.5.0 || true

echo "[serve-hf] host=$(hostname) jobid=${SLURM_JOB_ID:-unknown} date=$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

WORK_DIR="${WORK_DIR:-$HOME/nav4rail_serve}"
VENV_DIR="${VENV_DIR:-$WORK_DIR/venv_gpu_final}"
ADAPTER_DIR="${ADAPTER_DIR:-$HOME/nav4rail_adapters/mistral_7b}"
MERGED_DIR="${MERGED_DIR:-$HOME/nav4rail_merged/mistral_7b}"
BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
PORT="${PORT:-8080}"

cd "$WORK_DIR"

# ─── Virtual environment ─────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "[serve-hf] Creating venv..."
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install -q torch transformers peft accelerate fastapi uvicorn 2>/dev/null || true

# ─── Merge LoRA if not already done ─────────────────────────────────────────
if [ ! -f "$MERGED_DIR/config.json" ]; then
  echo "[serve-hf] Merging LoRA adapter into base model..."
  echo "[serve-hf] Adapter: $ADAPTER_DIR"
  echo "[serve-hf] Base: $BASE_MODEL"
  echo "[serve-hf] Output: $MERGED_DIR"

  if [ ! -d "$ADAPTER_DIR" ] || [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
    echo "[serve-hf] ERROR: adapter not found at $ADAPTER_DIR"
    echo "[serve-hf] Copy them first: scp -r rpi5:~/nav4rail_adapters_final/mistral_7b $ADAPTER_DIR"
    exit 1
  fi

  python3 -c "
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained('$BASE_MODEL', torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, '$ADAPTER_DIR')
model = model.merge_and_unload()
model.save_pretrained('$MERGED_DIR')
tokenizer = AutoTokenizer.from_pretrained('$ADAPTER_DIR')
tokenizer.save_pretrained('$MERGED_DIR')
print('[serve-hf] Merge complete')
"
  echo "[serve-hf] Merged model saved to $MERGED_DIR"
else
  echo "[serve-hf] Using existing merged model at $MERGED_DIR"
fi

# ─── Inference server (HF transformers) ─────────────────────────────────────
cat > "$WORK_DIR/serve_hf.py" << 'SERVEPY'
"""NAV4RAIL inference server — HuggingFace transformers edition (Mistral 7B merged fp16).

Serves the full-precision merged model instead of GGUF quantized version.
API is compatible with the GGUF serve.py (/health, /generate, /models).
"""

import os
import re
import sys
import time
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

MERGED_DIR = os.environ.get("MERGED_DIR", os.path.expanduser("~/nav4rail_merged/mistral_7b"))

# ─── Load model ──────────────────────────────────────────────────────────────
print(f"[serve-hf] Loading merged model from {MERGED_DIR} ...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MERGED_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
load_time = time.time() - t0
print(f"[serve-hf] Model loaded in {load_time:.1f}s — device: {model.device}")
lock = threading.Lock()


# ─── Prompt ──────────────────────────────────────────────────────────────────

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


def build_prompt(mission: str) -> str:
    instruction = f"{SYSTEM_PROMPT}\n\n{SKILLS_DOC}\n\nMission : {mission}"
    return f"[INST] {instruction} [/INST]"


def extract_xml(text: str) -> str:
    match = re.search(r"(<root\b.*?</root>)", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


# ─── Validation ──────────────────────────────────────────────────────────────
FINETUNE_DIR = os.environ.get("FINETUNE_DIR", ".")
sys.path.insert(0, FINETUNE_DIR)

try:
    from validate_bt import validate_bt, enrich_ports
    print("[serve-hf] BT validator + port enrichment loaded")
except ImportError:
    def validate_bt(xml):
        class _R:
            valid = True; score = 1.0; errors = []; warnings = []
            def summary(self): return "no validator"
        return _R()
    def enrich_ports(xml): return xml
    print("[serve-hf] WARNING: validate_bt not available")


# ─── API ─────────────────────────────────────────────────────────────────────

class GenRequest(BaseModel):
    mission: str
    use_grammar: bool = False  # grammar not supported in HF mode
    model: str | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu": True,
        "model": "mistral-7b-merged-fp16",
        "format": "mistral",
        "backend": "transformers",
    }


@app.get("/models")
def list_models():
    return {
        "current": "mistral-7b-merged-fp16",
        "available": {
            "mistral-7b-merged-fp16": {
                "filename": "merged (fp16)",
                "format": "mistral",
                "backend": "transformers",
            }
        },
    }


@app.post("/generate")
def generate(req: GenRequest):
    prompt = build_prompt(req.mission)
    print(f"[serve-hf] Generating for: {req.mission[:80]}...")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with lock:
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=1.0,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_time = time.time() - t0

    generated_ids = outputs[0][input_len:]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
    xml_raw = extract_xml(raw)
    xml = enrich_ports(xml_raw)
    vr = validate_bt(xml)

    enriched = xml != xml_raw
    print(f"[serve-hf] Done in {gen_time:.1f}s — score={vr.score:.2f} (enriched={enriched})")

    return {
        "xml": xml,
        "xml_raw": xml_raw,
        "valid": vr.valid,
        "score": round(vr.score, 2),
        "errors": vr.errors,
        "warnings": vr.warnings,
        "summary": vr.summary(),
        "generation_time_s": round(gen_time, 1),
        "model": "mistral-7b-merged-fp16",
        "enriched": enriched,
    }
SERVEPY

# ─── Copy finetune modules ──────────────────────────────────────────────────
FINETUNE_SRC="${FINETUNE_SRC:-$HOME/Telecom_Projet_fil_rouge/webapp/finetune}"
if [ -d "$FINETUNE_SRC" ]; then
  cp -u "$FINETUNE_SRC/nav4rail_grammar.py" "$WORK_DIR/" 2>/dev/null || true
  cp -u "$FINETUNE_SRC/validate_bt.py" "$WORK_DIR/" 2>/dev/null || true
  echo "[serve-hf] Copied grammar + validator from $FINETUNE_SRC"
fi

# ─── Reverse SSH tunnel to gpu-gw ───────────────────────────────────────────
echo "[serve-hf] Opening reverse SSH tunnel to gpu-gw:${PORT}..."
ssh -fN -R "${PORT}:localhost:${PORT}" gpu-gw \
  -o StrictHostKeyChecking=no \
  -o ServerAliveInterval=30 \
  -o ExitOnForwardFailure=yes \
  && echo "[serve-hf] Tunnel active: gpu-gw:${PORT} → $(hostname):${PORT}" \
  || echo "[serve-hf] WARNING: tunnel failed"

# ─── Launch ──────────────────────────────────────────────────────────────────
export MERGED_DIR FINETUNE_DIR="$WORK_DIR"
echo "[serve-hf] Starting Mistral 7B merged (fp16) on port $PORT ..."
exec uvicorn serve_hf:app --host 0.0.0.0 --port "$PORT"
