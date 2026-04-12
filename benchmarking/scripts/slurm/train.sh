#!/bin/bash
# ============================================================================
# NAV4RAIL Benchmark — SLURM Training Job
# ============================================================================
# Usage:
#   METHOD=sft MODEL=gemma2_9b sbatch scripts/slurm/train.sh
#   METHOD=grpo MODEL=mistral_7b sbatch scripts/slurm/train.sh
#   sbatch --partition=P100 scripts/slurm/train.sh   # override partition
#
# Variables (override via environment before sbatch):
#   METHOD   — Training method: sft, dpo, grpo, kto, orpo (default: sft)
#   MODEL    — Model key: mistral_7b, llama3_8b, qwen25_coder_7b, gemma2_9b, qwen25_14b
#   CONFIG   — Config file path (default: configs/base.yaml)

#SBATCH --job-name=nav4rail_train
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

echo "=== NAV4RAIL Training ==="
echo "Method: ${METHOD}"
echo "Model:  ${MODEL}"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"

# ── Common setup (venv, PYTHONPATH, diagnostics) ────────────────────────────
# Slurm executes a copy in /var/spool/slurmd/...; SLURM_SUBMIT_DIR is the dir where sbatch ran.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -r "${SLURM_SUBMIT_DIR}/scripts/slurm/_common.sh" ]; then
  # shellcheck source=scripts/slurm/_common.sh
  source "${SLURM_SUBMIT_DIR}/scripts/slurm/_common.sh"
else
  # shellcheck source=scripts/slurm/_common.sh
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
fi

# ── Run directory ───────────────────────────────────────────────────────────
RUN_DIR="runs/slurm/nav4rail_${METHOD}_${MODEL}_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

export WANDB_PROJECT=nav4rail-bench
export CUDA_VISIBLE_DEVICES=0

# ── Train ───────────────────────────────────────────────────────────────────
python -m src.train.unified_trainer \
    --config "$CONFIG" \
    --model "$MODEL" \
    --method "$METHOD" \
    2>&1 | tee "$RUN_DIR/train.log"

TRAIN_EXIT=$?

# ── Benchmark (if training succeeded) ──────────────────────────────────────
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "=== Running benchmark evaluation ==="
    python -m src.eval.benchmark \
        --config "$CONFIG" \
        --model "$MODEL" \
        --output "$RUN_DIR/eval_results/" \
        2>&1 | tee "$RUN_DIR/eval.log"
fi

echo "=== Job complete (exit: $TRAIN_EXIT) ==="
