#!/bin/bash
#SBATCH --job-name=nav2_ft_llama3_8b
#SBATCH --output=nav2_finetune_llama3_8b_%j.out
#SBATCH --error=nav2_finetune_llama3_8b_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mem=32G

set -euo pipefail

module load python/3.11.13 cuda/12.4.1 || true

WORK_DIR="$HOME/code/nav4rail_finetune_nav2"
VENV_DIR="$HOME/venvs/nav4rail_nav2_steps"

echo "[job] host=$(hostname) jobid=${SLURM_JOB_ID:-unknown} date=$(date -Iseconds)"
echo "[job] nvidia-smi:"
nvidia-smi || true

cd "$WORK_DIR"

if [ ! -f "$WORK_DIR/finetune_Nav2/requirements.txt" ]; then
  echo "[job] ERROR: missing $WORK_DIR/finetune_Nav2/requirements.txt"
  echo "[job] You likely copied finetune_Nav2/* instead of the finetune_Nav2/ folder."
  echo "[job] Fix: scp -r <local>/finetune_Nav2 gpu:~/code/nav4rail_finetune_nav2/"
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip
python3 -m pip install -r finetune_Nav2/requirements.txt

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

DATASET_PATH="${DATASET_PATH:-$WORK_DIR/finetune_Nav2/dataset_out/dataset_nav2_steps.jsonl}"
OUT_DIR="${OUT_DIR:-$WORK_DIR/finetune_Nav2/outputs/nav2_steps_llama3_8b_lora_${SLURM_JOB_ID:-local}}"

python3 -m finetune_Nav2.train.finetune_qlora_steps_json \
  --model-key llama3_8b \
  --dataset "$DATASET_PATH" \
  --output-dir "$OUT_DIR"

echo "[job] adapter saved under: $OUT_DIR/lora_adapter"

