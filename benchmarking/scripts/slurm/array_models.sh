#!/bin/bash
# ============================================================================
# NAV4RAIL Benchmark — SLURM Job Array: All 5 Models
# ============================================================================
# Runs the same method on all 5 models in parallel.
# Usage:
#   METHOD=sft PROMPT_MODE=zero_shot sbatch scripts/slurm/array_models.sh

#SBATCH --job-name=nav4rail_array
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=runs/slurm/nav4rail_array_%A_%a/slurm_%A_%a.out

MODELS=("mistral_7b" "llama3_8b" "qwen25_coder_7b" "gemma2_9b" "qwen25_14b")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
METHOD=${METHOD:-zero_shot}
PROMPT_MODE=${PROMPT_MODE:-$METHOD}

echo "Array task $SLURM_ARRAY_TASK_ID: model=$MODEL method=$METHOD"

module load python/3.11.13 cuda/12.4.1 2>/dev/null || true

# Setup venv (with flock to avoid pip races)
VENV_DIR="$HOME/.venvs/nav4rail_bench"
if [ ! -d "$VENV_DIR" ]; then
    (
        flock -x 200
        if [ ! -d "$VENV_DIR" ]; then
            python -m venv "$VENV_DIR"
            source "$VENV_DIR/bin/activate"
            pip install --upgrade pip
            pip install -r requirements.txt
        fi
    ) 200>"$VENV_DIR.lock"
fi
source "$VENV_DIR/bin/activate"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python -m src.eval.benchmark \
    --config configs/base.yaml \
    --model "$MODEL" \
    --prompt-mode "$PROMPT_MODE" \
    --output "runs/slurm/nav4rail_${METHOD}_${MODEL}_${SLURM_ARRAY_JOB_ID}/"
