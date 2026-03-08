#!/bin/bash
#SBATCH --job-name=nav2_eval_oracle
#SBATCH --output=nav2_eval_oracle_%j.out
#SBATCH --error=nav2_eval_oracle_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:0
#SBATCH --time=00:15:00
#SBATCH --mem=8G

set -euo pipefail

module load python/3.11.13 cuda/12.4.1 || true

WORK_DIR="$HOME/code/nav4rail_finetune_nav2"
VENV_DIR="$HOME/venvs/nav4rail_nav2_steps"

echo "[job] host=$(hostname) jobid=${SLURM_JOB_ID:-unknown} date=$(date -Iseconds)"

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

# Stable caches (avoid re-downloading big wheels / HF shards between jobs)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR"

python3 -m pip install --upgrade pip
python3 -m pip install -r finetune_Nav2/requirements.txt

DATASET_PATH="${DATASET_PATH:-$WORK_DIR/finetune_Nav2/dataset_out/dataset_nav2_steps.jsonl}"
N_SAMPLES="${N_SAMPLES:-5}"

python3 -m finetune_Nav2.eval.run_static_eval \
  --dataset "$DATASET_PATH" \
  --n "$N_SAMPLES" \
  --strict-attrs \
  --strict-blackboard

