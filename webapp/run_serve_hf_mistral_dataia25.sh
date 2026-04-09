#!/bin/bash
# Serve fine-tuned Mistral 7B (merged fp16, NOT GGUF) on dataia25 GPU 1
# Uses HuggingFace transformers — full precision merged model.
#
# Usage:
#   ssh dataia25
#   nohup bash ~/nav4rail/run_serve_hf_mistral.sh > ~/nav4rail/serve_hf_mistral.log 2>&1 &
#
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export HF_HOME="$HOME/nav4rail/hf_cache"
mkdir -p "$HF_HOME"

cd "$HOME/nav4rail"
source venv/bin/activate

ADAPTER_DIR="$HOME/nav4rail/adapters/mistral_7b"
MERGED_DIR="$HOME/nav4rail/merged_mistral_7b"
BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
PORT=8080

echo "================================================================"
echo "[serve-hf] host=$(hostname) date=$(date -Iseconds)"
echo "[serve-hf] GPU 1:"
nvidia-smi -i 1 --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo "================================================================"

# ─── Merge LoRA if not already done ─────────────────────────────────────────
if [ ! -f "$MERGED_DIR/config.json" ]; then
  echo "[serve-hf] Merging LoRA adapter into base model..."
  echo "[serve-hf] Adapter: $ADAPTER_DIR"
  echo "[serve-hf] Base: $BASE_MODEL (will download if not cached)"
  echo "[serve-hf] Output: $MERGED_DIR"

  python3 -c "
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

print('[merge] Loading base model...')
base = AutoModelForCausalLM.from_pretrained('$BASE_MODEL', torch_dtype=torch.float16)
print('[merge] Loading LoRA adapter...')
model = PeftModel.from_pretrained(base, '$ADAPTER_DIR')
print('[merge] Merging...')
model = model.merge_and_unload()
model.save_pretrained('$MERGED_DIR')
tokenizer = AutoTokenizer.from_pretrained('$ADAPTER_DIR')
tokenizer.save_pretrained('$MERGED_DIR')
print('[merge] Done — saved to $MERGED_DIR')
"
fi

echo "[serve-hf] Starting server with merged model at $MERGED_DIR"
echo "[serve-hf] Auto-shutdown after 15 minutes of inactivity"

# ─── Inline server ──────────────────────────────────────────────────────────
FINETUNE_DIR="$HOME/nav4rail" MERGED_DIR="$MERGED_DIR" IDLE_TIMEOUT="900" \
  python3 -c "
import os, re, sys, time, threading, signal, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel as PydModel
import uvicorn

MERGED_DIR = os.environ['MERGED_DIR']
IDLE_TIMEOUT = int(os.environ.get('IDLE_TIMEOUT', '900'))
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
            print(f'[serve-hf] Idle for {idle:.0f}s > {IDLE_TIMEOUT}s — shutting down')
            os.kill(os.getpid(), signal.SIGTERM)
            break

watchdog = threading.Thread(target=_idle_watchdog, daemon=True)
watchdog.start()

# ─── Load model ─────────────────────────────────────────────────────────────
print(f'[serve-hf] Loading merged model from {MERGED_DIR} ...')
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
model = AutoModelForCausalLM.from_pretrained(MERGED_DIR, torch_dtype=torch.float16, device_map='auto')
model.eval()
load_time = time.time()-t0
print(f'[serve-hf] Loaded in {load_time:.1f}s — device: {model.device}')
lock = threading.Lock()

# ─── GPU info ────────────────────────────────────────────────────────────────
import subprocess
try:
    _gpu_info = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader','-i','1'], text=True).strip()
except Exception:
    _gpu_info = 'RTX 3090'

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
    return f'[INST] {SYSTEM_PROMPT}\n\n{SKILLS_DOC}\n\nMission : {mission} [/INST]'

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
        'status':'ok',
        'gpu':True,
        'gpu_name': _gpu_info,
        'model':'mistral-7b-merged-fp16',
        'format':'mistral',
        'backend':'transformers',
        'cluster':'dataia25',
        'idle_timeout_s': IDLE_TIMEOUT,
        'remaining_s': int(remaining),
        'uptime_s': int(uptime),
    }

@app.get('/models')
def models():
    return {'current':'mistral-7b-merged-fp16','available':{'mistral-7b-merged-fp16':{'filename':'merged(fp16)','format':'mistral','backend':'transformers'}}}

@app.post('/generate')
def generate(req: GenReq):
    global _last_activity
    _last_activity = time.time()
    prompt = build_prompt(req.mission)
    print(f'[serve-hf] Generating: {req.mission[:80]}...')
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_len = inputs['input_ids'].shape[1]
    with lock:
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=2048, temperature=1.0, do_sample=False, repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)
        gen_time = time.time()-t0
    _last_activity = time.time()
    raw = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    xml_raw = extract_xml(raw)
    xml = enrich_ports(xml_raw)
    vr = validate_bt(xml)
    enriched = xml != xml_raw
    print(f'[serve-hf] Done {gen_time:.1f}s score={vr.score:.2f} enriched={enriched}')
    return {'xml':xml,'xml_raw':xml_raw,'valid':vr.valid,'score':round(vr.score,2),'errors':vr.errors,'warnings':vr.warnings,'summary':vr.summary(),'generation_time_s':round(gen_time,1),'model':'mistral-7b-merged-fp16','enriched':enriched}

uvicorn.run(app, host='0.0.0.0', port=$PORT)
"
