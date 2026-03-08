#!/bin/bash
#SBATCH --job-name=nav2_xml_eval_hf
#SBATCH --output=nav2_xml_eval_hf_%j.out
#SBATCH --error=nav2_xml_eval_hf_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G

set -euo pipefail

module load python/3.11.13 cuda/12.4.1 || true

WORK_DIR="$HOME/code/nav4rail_finetune_nav2"
VENV_DIR="$HOME/venvs/nav4rail_nav2_xml"

echo "[job] host=$(hostname) jobid=${SLURM_JOB_ID:-unknown} date=$(date -Iseconds)"
echo "[job] nvidia-smi:"
nvidia-smi || true

cd "$WORK_DIR"

if [ ! -f "$WORK_DIR/finetune_Nav2_XML/requirements.txt" ]; then
  echo "[job] ERROR: missing $WORK_DIR/finetune_Nav2_XML/requirements.txt"
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR"

python3 -m pip install --upgrade pip
python3 -m pip install -r finetune_Nav2_XML/requirements.txt

MODEL_KEY="${MODEL_KEY:-mistral7b}"
DATASET_PATH="${DATASET_PATH:-$WORK_DIR/finetune_Nav2_XML/dataset_out/dataset_nav2_bt_xml.jsonl}"
ADAPTER_DIR="${ADAPTER_DIR:-$WORK_DIR/finetune_Nav2_XML/outputs/nav2_xml_mistral7b_lora_${SLURM_JOB_ID:-local}/lora_adapter}"
N_SAMPLES="${N_SAMPLES:-10}"
CONSTRAINED="${CONSTRAINED:-off}"   # off|regex

python3 -m finetune_Nav2_XML.eval.run_hf_eval \
  --model-key "$MODEL_KEY" \
  --adapter-dir "$ADAPTER_DIR" \
  --dataset "$DATASET_PATH" \
  --n "$N_SAMPLES" \
  --constrained "$CONSTRAINED" \
  --strict-attrs \
  --strict-blackboard

echo "[job] done"

