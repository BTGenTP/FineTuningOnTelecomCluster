#!/bin/bash
# ============================================================================
# NAV4RAIL Benchmark — SLURM Evaluation Job (inference only)
# ============================================================================
# Usage:
#   MODEL=gemma2_9b PROMPT_MODE=zero_shot sbatch scripts/slurm/eval.sh
#   MODEL=mistral_7b ADAPTER=runs/sft_mistral_7b/final_adapter sbatch scripts/slurm/eval.sh

#SBATCH --job-name=nav4rail_eval
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm/nav4rail_eval_%j/slurm_%j.out

MODEL=${MODEL:-mistral_7b}
CONFIG=${CONFIG:-configs/base.yaml}
PROMPT_MODE=${PROMPT_MODE:-zero_shot}
ADAPTER=${ADAPTER:-}

RUN_DIR="runs/slurm/nav4rail_eval_${PROMPT_MODE}_${MODEL}_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

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

export WANDB_PROJECT=nav4rail-bench
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

ADAPTER_FLAG=""
if [ -n "$ADAPTER" ]; then
    ADAPTER_FLAG="--adapter $ADAPTER"
fi

python -m src.eval.benchmark \
    --config "$CONFIG" \
    --model "$MODEL" \
    --prompt-mode "$PROMPT_MODE" \
    $ADAPTER_FLAG \
    --output "$RUN_DIR/" \
    2>&1 | tee "$RUN_DIR/eval.log"

echo "=== Evaluation complete ==="
