#!/bin/bash
#SBATCH --job-name=nav4rail-serve
#SBATCH --output=nav4rail_serve_%j.out
#SBATCH --error=nav4rail_serve_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#
# NAV4RAIL fp16 inference server on Télécom Paris GPU cluster.
# Supports all fine-tuned models via in-memory LoRA merge.
#
# Usage:
#   MODEL_KEY=mistral_7b sbatch --partition=P100 --gres=gpu:1 --mem=16G job_serve_fp16.sh
#   MODEL_KEY=qwen25_coder_7b sbatch --partition=3090 --gres=gpu:1 --mem=32G job_serve_fp16.sh
#
# The webapp submits with the right partition/mem via SSH.
#

set -euo pipefail

MODEL_KEY="${MODEL_KEY:-mistral_7b}"

module load python/3.11.13 cuda/12.4.1 gcc/11.5.0 2>/dev/null || true

echo "[serve-hf] host=$(hostname) jobid=${SLURM_JOB_ID:-unknown} date=$(date -Iseconds)"
echo "[serve-hf] partition=${SLURM_JOB_PARTITION:-unknown} model=$MODEL_KEY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

WORK_DIR="${WORK_DIR:-$HOME/nav4rail_serve}"
VENV_DIR="$WORK_DIR/venv_hf"
ADAPTER_DIR="$HOME/nav4rail_adapters/$MODEL_KEY"
PORT="${PORT:-8080}"

# Record SLURM time info for health endpoint
_slurm_time_limit=$((4 * 3600))
_start_time=$(date +%s)

cd "$WORK_DIR"

# ─── Virtual environment with torch + transformers + peft ─────────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "[serve-hf] Creating venv_hf..."
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install deps if missing
python3 -c "import torch, transformers, peft, fastapi, uvicorn" 2>/dev/null || {
  echo "[serve-hf] Installing dependencies..."
  pip install -q --upgrade pip
  pip install -q torch --index-url https://download.pytorch.org/whl/cu124
  pip install -q transformers accelerate peft fastapi uvicorn
  # Remove torchvision if accidentally pulled (triggers bz2 crash on cluster)
  pip uninstall -yq torchvision 2>/dev/null || true
}

python3 -c "import torch; print(f'[serve-hf] torch={torch.__version__} cuda={torch.cuda.is_available()}')"

# ─── Model registry ─────────────────────────────────────────────────────────
declare -A BASE_MODELS=(
  [mistral_7b]="mistralai/Mistral-7B-Instruct-v0.2"
  [llama3_8b]="NousResearch/Meta-Llama-3.1-8B-Instruct"
  [qwen25_coder_7b]="Qwen/Qwen2.5-Coder-7B-Instruct"
  [gemma2_9b]="google/gemma-2-9b-it"
  [qwen25_14b]="Qwen/Qwen2.5-14B-Instruct"
)
declare -A API_KEYS=(
  [mistral_7b]="mistral-7b-merged-fp16"
  [llama3_8b]="llama3-8b-merged-fp16"
  [qwen25_coder_7b]="qwen-coder-7b-merged-fp16"
  [gemma2_9b]="gemma2-9b-merged-fp16"
  [qwen25_14b]="qwen-14b-merged-fp16"
)
declare -A PROMPT_FORMATS=(
  [mistral_7b]="mistral"
  [llama3_8b]="llama3"
  [qwen25_coder_7b]="chatml"
  [gemma2_9b]="gemma"
  [qwen25_14b]="chatml"
)
declare -A REP_PENALTIES=(
  [mistral_7b]="1.1"
  [llama3_8b]="1.0"
  [qwen25_coder_7b]="1.1"
  [gemma2_9b]="1.1"
  [qwen25_14b]="1.1"
)

BASE_MODEL="${BASE_MODELS[$MODEL_KEY]}"
API_KEY="${API_KEYS[$MODEL_KEY]}"
PROMPT_FMT="${PROMPT_FORMATS[$MODEL_KEY]}"
REP_PENALTY="${REP_PENALTIES[$MODEL_KEY]:-1.1}"

echo "================================================================"
echo "[serve-hf] model=$MODEL_KEY  base=$BASE_MODEL"
echo "[serve-hf] adapter=$ADAPTER_DIR"
echo "[serve-hf] prompt_format=$PROMPT_FMT  api_key=$API_KEY  rep_penalty=$REP_PENALTY"
echo "================================================================"

if [ ! -d "$ADAPTER_DIR" ]; then
  echo "[serve-hf] WARNING: Adapter not found: $ADAPTER_DIR (continuing with pre-merged)"
fi

# ─── Pre-merged model directory ──────────────────────────────────────────────
MERGED_DIR="$HOME/nav4rail_merged/$MODEL_KEY"
if [ ! -f "$MERGED_DIR/config.json" ] || [ -z "$(ls $MERGED_DIR/*.safetensors 2>/dev/null)" ]; then
  echo "[serve-hf] ERROR: Pre-merged model not found at $MERGED_DIR"
  echo "[serve-hf] Merge the model first on a machine with enough RAM, then copy here."
  exit 1
fi
echo "[serve-hf] Using pre-merged model from $MERGED_DIR"

# ─── No reverse tunnel needed ─────────────────────────────────────────────────
# RPi5 tunnels directly to the compute node via gpu-gw:
#   ssh -L 8080:<node>:8080 gpu-gw
# This avoids the port 8080 conflict on gpu-gw (another service binds 127.0.0.1:8080).
echo "[serve-hf] Server will listen on $(hostname):${PORT}"

# ─── Inline server ──────────────────────────────────────────────────────────
export HF_HOME="$WORK_DIR/hf_cache"
mkdir -p "$HF_HOME"

FINETUNE_DIR="$WORK_DIR" MERGED_DIR="$MERGED_DIR" \
  MODEL_API_KEY="$API_KEY" PROMPT_FMT="$PROMPT_FMT" REP_PENALTY="$REP_PENALTY" \
  BASE_MODEL="$BASE_MODEL" ADAPTER_DIR="$ADAPTER_DIR" \
  SLURM_TIME_LIMIT="$_slurm_time_limit" \
  START_TIME="$_start_time" \
  PYTHONUNBUFFERED=1 \
  python3 -c "
import os, re, sys, time, threading, signal, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel as PydModel
import uvicorn

MERGED_DIR = os.environ['MERGED_DIR']
MODEL_API_KEY = os.environ['MODEL_API_KEY']
PROMPT_FMT = os.environ['PROMPT_FMT']
BASE_MODEL_ID = os.environ['BASE_MODEL']
ADAPTER_DIR = os.environ['ADAPTER_DIR']
REP_PENALTY = float(os.environ.get('REP_PENALTY', '1.1'))
_slurm_time_limit = int(os.environ.get('SLURM_TIME_LIMIT', str(4*3600)))
_start_time = float(os.environ.get('START_TIME', str(time.time())))
sys.path.insert(0, os.environ.get('FINETUNE_DIR', '.'))

try:
    from validate_bt import validate_bt, enrich_ports
    print('[serve-hf] Validator loaded')
except ImportError:
    def validate_bt(xml):
        class _R:
            valid=True; score=1.0; errors=[]; warnings=[]
            def summary(self): return 'no validator'
        return _R()
    def enrich_ports(xml): return xml

_last_activity = time.time()

# ─── Load pre-merged model directly (fast path) ─────────────────────────────
print(f'[serve-hf] Loading pre-merged model from {MERGED_DIR}...')
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MERGED_DIR, dtype=torch.float16, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
model.eval()
load_time = time.time()-t0
print(f'[serve-hf] Loaded in {load_time:.1f}s — device: {model.device}')
_last_activity = time.time()
lock = threading.Lock()

import subprocess
try:
    _gpu_info = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'], text=True).strip().split(chr(10))[0]
except Exception:
    _gpu_info = 'GPU'

# ─── Prompt templates ────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    'Tu es un expert en robotique ferroviaire NAV4RAIL. '
    'Genere un Behavior Tree XML BehaviorTree.CPP pour la mission decrite.\n\n'
    'FORMAT :\n'
    '- <root BTCPP_format=\"4\" main_tree_to_execute=\"nom\">\n'
    '- Multi-<BehaviorTree ID=\"...\"> interconnectes via <SubTreePlus __autoremap=\"true\">\n'
    '- <Action name=\"NOM\" ID=\"Skill\" port=\"{var}\"/>  <Condition name=\"NOM\" ID=\"Skill\"/>\n'
    '- Controle : Sequence, Fallback, ReactiveFallback, Repeat(num_cycles=\"-1\")\n'
    '- Chaque noeud a name=\"DESCRIPTION EN MAJUSCULES\", ports blackboard {variable}\n\n'
    'ARCHITECTURE :\n'
    'principal -> Sequence(preparation + execution via SubTreePlus)\n'
    'preparation -> LoadMission + MissionStructureValid + calculate_path + PassAdvancedPath + PassMission + GenerateMissionSequence\n'
    'calculate_path -> Fallback(Repeat(-1)(UpdateCurrentGeneratedActivity/ProjectPointOnNetwork/CreatePath/AgregatePath), MissionFullyTreated)\n'
    'execution -> ReactiveFallback(Repeat(-1)(Fallback motion_selector), MissionTerminated)\n\n'
    'CHOIX DES MOTION SUBTREES (CRUCIAL) :\n'
    'Transport (TOUJOURS inclure) :\n'
    '  move(type=0 Move), deccelerate(type=1), reach_and_stop(type=2 MoveAndStop+SignalAndWaitForOrder), pass(type=3 Move threshold=3), reach_stop_no_wait(type=4 MoveAndStop)\n'
    'Inspection AVEC controle :\n'
    '  move_and_inspect(type=10), deccel_and_inspect(type=11), reach_stop_inspecting(type=12), pass_stop_inspecting(type=13), reach_stop_inspect_no_wait(type=14)\n'
    'Inspection SANS controle : types 10-14 avec ManageMeasurements SANS AnalyseMeasurements\n\n'
    'Reponds uniquement avec le XML.'
)

SKILLS_DOC = '''Skills (28) :
PREPARATION: LoadMission(mission_file_path), MissionStructureValid[C], UpdateCurrentGeneratedActivity, ProjectPointOnNetwork(point_in,point_out), CreatePath(origin,target,forbidden_atoms,path), AgregatePath(path), MissionFullyTreated[C](type), PassAdvancedPath(adv_path), PassMission(mission), GenerateMissionSequence(mission,mission_sequence), GenerateCorrectiveSubSequence(defects), InsertCorrectiveSubSequence
MOTION: MissionTerminated[C], CheckCurrentStepType[C](type_to_be_checked), PassMotionParameters(motion_params), Move(threshold_type,motion_params), UpdateCurrentExecutedStep, Deccelerate(motion_params), MoveAndStop(motion_params), SignalAndWaitForOrder(message), IsRobotPoseProjectionActive[C](adv_path,pub_proj), Pause(duration)
INSPECTION: ManageMeasurements, AnalyseMeasurements, MeasurementsQualityValidated[C], PassDefectsLocalization(defects), MeasurementsEnforcedValidated[C]
SIMULATION: SimulationStarted[C]'''

def build_prompt(mission):
    body = f'{SYSTEM_PROMPT}\n\n{SKILLS_DOC}\n\nMission : {mission}'
    if PROMPT_FMT == 'mistral':
        return f'[INST] {body} [/INST]'
    elif PROMPT_FMT == 'llama3':
        return f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}\n\n{SKILLS_DOC}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nMission : {mission}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    elif PROMPT_FMT == 'chatml':
        return f'<|im_start|>system\n{SYSTEM_PROMPT}\n\n{SKILLS_DOC}<|im_end|>\n<|im_start|>user\nMission : {mission}<|im_end|>\n<|im_start|>assistant\n'
    elif PROMPT_FMT == 'gemma':
        return f'<start_of_turn>user\n{body}<end_of_turn>\n<start_of_turn>model\n'
    return f'[INST] {body} [/INST]'

def extract_xml(text):
    m = re.search(r'(<root\b.*?</root>)', text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()

app = FastAPI()

class GenReq(PydModel):
    mission: str
    use_grammar: bool = False
    model: str | None = None

@app.get('/health')
def health():
    uptime = time.time() - _start_time
    remaining = max(0, _slurm_time_limit - uptime)
    return {
        'status':'ok','gpu':True,'gpu_name':_gpu_info,
        'model': MODEL_API_KEY, 'format': PROMPT_FMT,
        'backend':'transformers','cluster':'telecom-paris',
        'remaining_s':int(remaining),'uptime_s':int(uptime),
    }

@app.get('/models')
def models():
    return {'current':MODEL_API_KEY,'available':{MODEL_API_KEY:{'filename':'merged(fp16)','format':PROMPT_FMT,'backend':'transformers'}}}

@app.post('/generate')
def generate(req: GenReq):
    global _last_activity
    _last_activity = time.time()
    prompt = build_prompt(req.mission)
    print(f'[serve-hf] [{MODEL_API_KEY}] Generating: {req.mission[:80]}...')
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_len = inputs['input_ids'].shape[1]
    with lock:
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4096, temperature=1.0, do_sample=False, repetition_penalty=REP_PENALTY, pad_token_id=tokenizer.eos_token_id)
        gen_time = time.time()-t0
    _last_activity = time.time()
    gen_tokens = out[0].shape[0] - input_len
    eos_hit = (out[0][-1].item() == tokenizer.eos_token_id)
    raw = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    xml_raw = extract_xml(raw)
    xml = enrich_ports(xml_raw)
    vr = validate_bt(xml)
    enriched = xml != xml_raw
    print(f'[serve-hf] Done {gen_time:.1f}s score={vr.score:.2f} tokens={gen_tokens} eos={eos_hit} enriched={enriched}')
    return {'xml':xml,'xml_raw':xml_raw,'valid':vr.valid,'score':round(vr.score,2),'errors':vr.errors,'warnings':vr.warnings,'summary':vr.summary(),'generation_time_s':round(gen_time,1),'model':MODEL_API_KEY,'enriched':enriched,'debug':{'gen_tokens':gen_tokens,'input_tokens':input_len,'max_new_tokens':4096,'eos_hit':eos_hit,'rep_penalty':REP_PENALTY,'raw_len':len(raw),'xml_raw_len':len(xml_raw)}}

uvicorn.run(app, host='0.0.0.0', port=$PORT)
"
