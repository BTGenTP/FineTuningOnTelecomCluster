#!/bin/bash
# ============================================================================
# NAV4RAIL Benchmark — SLURM Training Job
# ============================================================================
# Usage:
#   METHOD=sft MODEL=gemma2_9b sbatch scripts/slurm/train.sh
#   METHOD=grpo MODEL=mistral_7b CONFIG=configs/base.yaml sbatch scripts/slurm/train.sh
#
# Variables (override via --export or environment):
#   METHOD   — Training method: sft, dpo, grpo, kto, orpo (default: sft)
#   MODEL    — Model key: mistral_7b, llama3_8b, qwen25_coder_7b, gemma2_9b, qwen25_14b (default: mistral_7b)
#   CONFIG   — Config file path (default: configs/base.yaml)
#   PARTITION — SLURM partition (default: 3090)

#SBATCH --job-name=nav4rail_%x
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/nav4rail_%j/slurm_%j.out
#SBATCH --error=runs/slurm/nav4rail_%j/slurm_%j.err

# Defaults
METHOD=${METHOD:-sft}
MODEL=${MODEL:-mistral_7b}
CONFIG=${CONFIG:-configs/base.yaml}

# Run directory
RUN_DIR="runs/slurm/nav4rail_${METHOD}_${MODEL}_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

echo "=== NAV4RAIL Benchmark ==="
echo "Method: ${METHOD}"
echo "Model: ${MODEL}"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Run dir: ${RUN_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================="

# Load modules
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

# Environment
export WANDB_PROJECT=nav4rail-bench
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Train
python -m src.train.unified_trainer \
    --config "$CONFIG" \
    --model "$MODEL" \
    --method "$METHOD" \
    2>&1 | tee "$RUN_DIR/train.log"

TRAIN_EXIT=$?

# Benchmark (if training succeeded)
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "=== Running benchmark evaluation ==="
    python -m src.eval.benchmark \
        --config "$CONFIG" \
        --model "$MODEL" \
        --output "$RUN_DIR/eval_results/" \
        2>&1 | tee "$RUN_DIR/eval.log"
fi

echo "=== Job complete ==="
echo "Exit code: $TRAIN_EXIT"
