#!/bin/bash
#SBATCH --job-name=nav2_xml_merge_mistral7b
#SBATCH --output=nav2_xml_merge_mistral7b_%j.out
#SBATCH --error=nav2_xml_merge_mistral7b_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=120G

set -euo pipefail

module load python/3.11.13 cuda/12.4.1 || true

echo "[job] host=$(hostname) jobid=${SLURM_JOB_ID:-unknown} date=$(date -Iseconds)"
echo "[job] nvidia-smi:"
nvidia-smi || true
echo "[job] free -h:"
free -h || true

# Paths (override via env when submitting)
WORK_DIR="${WORK_DIR:-$HOME/models/Mistral-7B-Instruct-v02-Lora-Nav2BT-XML}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/nav4rail_nav2_xml_merge}"

BASE_ID="${BASE_ID:-mistralai/Mistral-7B-Instruct-v0.2}"
MERGED_ID="${MERGED_ID:-mlatoundji/Mistral-7B-Instruct-v0.2-Nav2BT-XML-merged}"

BASE_DIR="${BASE_DIR:-$WORK_DIR/base}"
ADAPTER_DIR="${ADAPTER_DIR:-$WORK_DIR/adapter}"
OUT_DIR="${OUT_DIR:-$WORK_DIR/merged}"
OFFLOAD_DIR="${OFFLOAD_DIR:-$WORK_DIR/offload}"

mkdir -p "$WORK_DIR" "$OFFLOAD_DIR"
cd "$WORK_DIR"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$HF_HUB_CACHE" "$PIP_CACHE_DIR"

python3 -m pip install --upgrade pip
python3 -m pip install -U "torch" "transformers" "peft" "huggingface_hub" "safetensors"

# Optionnel: download des repos HF vers BASE_DIR / ADAPTER_DIR (commenté)
# python3 - <<'PY'
# import os
# from huggingface_hub import snapshot_download
# base_id = os.environ["BASE_ID"]
# base_dir = os.environ["BASE_DIR"]
# adapter_id = os.environ.get("ADAPTER_ID")
# adapter_dir = os.environ["ADAPTER_DIR"]
# os.makedirs(base_dir, exist_ok=True)
# os.makedirs(adapter_dir, exist_ok=True)
# snapshot_download(repo_id=base_id, local_dir=base_dir, local_dir_use_symlinks=False)
# if adapter_id:
#     snapshot_download(repo_id=adapter_id, local_dir=adapter_dir, local_dir_use_symlinks=False, token=True)
# print("[job] downloads OK")
# PY

python3 - <<'PY'
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_dir = os.environ["BASE_DIR"]
adapter_dir = os.environ["ADAPTER_DIR"]
out_dir = os.environ["OUT_DIR"]
offload_dir = os.environ["OFFLOAD_DIR"]

os.makedirs(out_dir, exist_ok=True)
os.makedirs(offload_dir, exist_ok=True)

print(f"[job] loading tokenizer from {base_dir}")
tokenizer = AutoTokenizer.from_pretrained(base_dir, use_fast=True)

print(f"[job] loading base model on CPU (fp16), offload_dir={offload_dir}")
base = AutoModelForCausalLM.from_pretrained(
    base_dir,
    dtype=torch.float16,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
    offload_state_dict=True,
    offload_folder=offload_dir,
)

print(f"[job] loading adapter from {adapter_dir}")
model = PeftModel.from_pretrained(base, adapter_dir)

print("[job] merging adapter into base weights")
model = model.merge_and_unload()

print(f"[job] saving merged model to {out_dir}")
model.save_pretrained(out_dir, safe_serialization=True)
tokenizer.save_pretrained(out_dir)
print("[job] merge done")
PY

python3 - <<'PY'
import os
from huggingface_hub import HfApi

merged_id = os.environ["MERGED_ID"]
out_dir = os.environ["OUT_DIR"]

api = HfApi()
api.create_repo(merged_id, private=True, exist_ok=True, token=True)
api.upload_folder(repo_id=merged_id, folder_path=out_dir, token=True)
print("[job] uploaded merged model to", merged_id)
PY

echo "[job] done"

