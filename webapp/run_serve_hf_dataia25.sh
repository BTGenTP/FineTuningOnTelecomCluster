#!/bin/bash
# Generic serve script for fine-tuned models (merged fp16) on dataia25 GPU 1
# Supports: mistral_7b, llama3_8b, qwen25_coder_7b, gemma2_9b
#
# Usage:
#   bash run_serve_hf.sh llama3_8b
#   bash run_serve_hf.sh qwen25_coder_7b
#   bash run_serve_hf.sh gemma2_9b
#   bash run_serve_hf.sh mistral_7b
#
set -euo pipefail

MODEL_KEY="${1:?Usage: $0 <model_key> (mistral_7b|llama3_8b|qwen25_coder_7b|gemma2_9b)}"

export CUDA_VISIBLE_DEVICES=1
export HF_HOME="$HOME/nav4rail/hf_cache"
mkdir -p "$HF_HOME"

cd "$HOME/nav4rail"
source venv/bin/activate

# ─── Model registry ─────────────────────────────────────────────────────────
declare -A BASE_MODELS=(
  [mistral_7b]="mistralai/Mistral-7B-Instruct-v0.2"
  [llama3_8b]="NousResearch/Meta-Llama-3.1-8B-Instruct"
  [qwen25_coder_7b]="Qwen/Qwen2.5-Coder-7B-Instruct"
  [gemma2_9b]="google/gemma-2-9b-it"
)
declare -A API_KEYS=(
  [mistral_7b]="mistral-7b-merged-fp16"
  [llama3_8b]="llama3-8b-merged-fp16"
  [qwen25_coder_7b]="qwen-coder-7b-merged-fp16"
  [gemma2_9b]="gemma2-9b-merged-fp16"
)
declare -A PROMPT_FORMATS=(
  [mistral_7b]="mistral"
  [llama3_8b]="llama3"
  [qwen25_coder_7b]="chatml"
  [gemma2_9b]="gemma"
)
declare -A REP_PENALTIES=(
  [mistral_7b]="1.1"
  [llama3_8b]="1.0"
  [qwen25_coder_7b]="1.1"
  [gemma2_9b]="1.1"
)

BASE_MODEL="${BASE_MODELS[$MODEL_KEY]}"
API_KEY="${API_KEYS[$MODEL_KEY]}"
PROMPT_FMT="${PROMPT_FORMATS[$MODEL_KEY]}"
REP_PENALTY="${REP_PENALTIES[$MODEL_KEY]:-1.1}"
ADAPTER_DIR="$HOME/nav4rail/adapters/$MODEL_KEY"
MERGED_DIR="$HOME/nav4rail/merged_${MODEL_KEY}"
PORT=8080

echo "================================================================"
echo "[serve-hf] model=$MODEL_KEY  base=$BASE_MODEL"
echo "[serve-hf] adapter=$ADAPTER_DIR"
echo "[serve-hf] merged=$MERGED_DIR"
echo "[serve-hf] prompt_format=$PROMPT_FMT  api_key=$API_KEY"
echo "[serve-hf] host=$(hostname) date=$(date -Iseconds)"
echo "[serve-hf] GPU 1:"
nvidia-smi -i 1 --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo "================================================================"

# ─── Merge LoRA if not already done ─────────────────────────────────────────
if [ ! -f "$MERGED_DIR/config.json" ]; then
  echo "[serve-hf] Will merge LoRA in-memory (no disk save — disk too small)"
  echo "[serve-hf] Merge + load will happen inside the server process"
  NEED_MERGE=1
else
  echo "[serve-hf] Using pre-merged model at $MERGED_DIR"
  NEED_MERGE=0
fi

echo "[serve-hf] Starting server with $API_KEY"

# ─── Inline server ──────────────────────────────────────────────────────────
FINETUNE_DIR="$HOME/nav4rail" MERGED_DIR="$MERGED_DIR" IDLE_TIMEOUT="900" \
  MODEL_API_KEY="$API_KEY" PROMPT_FMT="$PROMPT_FMT" REP_PENALTY="$REP_PENALTY" \
  BASE_MODEL="$BASE_MODEL" ADAPTER_DIR="$ADAPTER_DIR" NEED_MERGE="$NEED_MERGE" \
  python3 -c "
import os, re, sys, time, threading, signal, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel as PydModel
import uvicorn

MERGED_DIR = os.environ['MERGED_DIR']
IDLE_TIMEOUT = int(os.environ.get('IDLE_TIMEOUT', '300'))
MODEL_API_KEY = os.environ['MODEL_API_KEY']
PROMPT_FMT = os.environ['PROMPT_FMT']
BASE_MODEL_ID = os.environ['BASE_MODEL']
ADAPTER_DIR = os.environ['ADAPTER_DIR']
NEED_MERGE = os.environ.get('NEED_MERGE', '0') == '1'
REP_PENALTY = float(os.environ.get('REP_PENALTY', '1.1'))
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

# ─── Idle auto-shutdown ─────────────────────────────────────────────────────
_last_activity = time.time()
_start_time = time.time()

def _idle_watchdog():
    while True:
        time.sleep(30)
        idle = time.time() - _last_activity
        if idle > IDLE_TIMEOUT:
            print(f'[serve-hf] Idle {idle:.0f}s > {IDLE_TIMEOUT}s — shutting down')
            os.kill(os.getpid(), signal.SIGTERM)
            break

# NOTE: watchdog started AFTER model load to avoid self-kill during download

# ─── Load model (merge in-memory if needed) ─────────────────────────────────
if NEED_MERGE:
    print(f'[serve-hf] Loading base model {BASE_MODEL_ID} for in-memory merge...')
    t0 = time.time()
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    print(f'[serve-hf] Applying LoRA from {ADAPTER_DIR}...')
    peft_model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model = peft_model.merge_and_unload()
    del base, peft_model
    import gc; gc.collect()
    # Clean HF cache to free disk (base model no longer needed)
    import shutil
    cache_dir = os.path.join(os.environ.get('HF_HOME', ''), 'hub')
    if os.path.isdir(cache_dir):
        for d in os.listdir(cache_dir):
            if d.startswith('models--'):
                p = os.path.join(cache_dir, d)
                print(f'[serve-hf] Cleaning cache: {p}')
                shutil.rmtree(p, ignore_errors=True)
    print(f'[serve-hf] Moving model to GPU...')
    model = model.to('cuda')
    model.eval()
    load_time = time.time()-t0
    print(f'[serve-hf] Merged+loaded in {load_time:.1f}s — device: {model.device}')
    _last_activity = time.time()  # reset idle timer after long load
else:
    print(f'[serve-hf] Loading pre-merged model from {MERGED_DIR} ...')
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
    model = AutoModelForCausalLM.from_pretrained(MERGED_DIR, torch_dtype=torch.float16, device_map='auto')
    model.eval()
    load_time = time.time()-t0
    print(f'[serve-hf] Loaded in {load_time:.1f}s — device: {model.device}')
    _last_activity = time.time()  # reset idle timer after long load

# Start watchdog AFTER model is loaded (avoids self-kill during long downloads)
watchdog = threading.Thread(target=_idle_watchdog, daemon=True)
watchdog.start()
print(f'[serve-hf] Idle watchdog started (timeout={IDLE_TIMEOUT}s)')
lock = threading.Lock()

import subprocess
try:
    _gpu_info = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader','-i', os.environ.get('CUDA_VISIBLE_DEVICES','0')], text=True).strip()
except Exception:
    _gpu_info = 'RTX 3090'

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
    idle = time.time() - _last_activity
    uptime = time.time() - _start_time
    remaining = max(0, IDLE_TIMEOUT - idle)
    return {
        'status':'ok','gpu':True,'gpu_name':_gpu_info,
        'model': MODEL_API_KEY, 'format': PROMPT_FMT,
        'backend':'transformers','cluster':'dataia25',
        'idle_timeout_s':IDLE_TIMEOUT,'remaining_s':int(remaining),'uptime_s':int(uptime),
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
    print(f'[serve-hf] Done {gen_time:.1f}s score={vr.score:.2f} tokens={gen_tokens} eos={eos_hit} rep_pen={REP_PENALTY} enriched={enriched}')
    return {'xml':xml,'xml_raw':xml_raw,'valid':vr.valid,'score':round(vr.score,2),'errors':vr.errors,'warnings':vr.warnings,'summary':vr.summary(),'generation_time_s':round(gen_time,1),'model':MODEL_API_KEY,'enriched':enriched,'debug':{'gen_tokens':gen_tokens,'input_tokens':input_len,'max_new_tokens':4096,'eos_hit':eos_hit,'rep_penalty':REP_PENALTY,'raw_len':len(raw),'xml_raw_len':len(xml_raw)}}

uvicorn.run(app, host='0.0.0.0', port=$PORT)
"
