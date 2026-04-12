#!/bin/bash
# ============================================================================
# NAV4RAIL Benchmark — SLURM Job Array: All 5 Models
# ============================================================================
# Runs the same method on all 5 models in parallel.
# Usage:
#   METHOD=sft PROMPT_MODE=zero_shot sbatch scripts/slurm/array_models.sh
#   sbatch --partition=3090 scripts/slurm/array_models.sh

#SBATCH --job-name=nav4rail_array
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=runs/slurm/nav4rail_array_%A_%a/slurm_%A_%a.out
#SBATCH --error=runs/slurm/nav4rail_array_%A_%a/slurm_%A_%a.err

MODELS=("mistral_7b" "llama3_8b" "qwen25_coder_7b" "gemma2_9b" "qwen25_14b")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
METHOD=${METHOD:-zero_shot}
PROMPT_MODE=${PROMPT_MODE:-$METHOD}

echo "Array task $SLURM_ARRAY_TASK_ID: model=$MODEL method=$METHOD"

# ── Common setup (venv, PYTHONPATH, diagnostics) ────────────────────────────
# Slurm executes a copy in /var/spool/slurmd/...; SLURM_SUBMIT_DIR is the dir where sbatch ran.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -r "${SLURM_SUBMIT_DIR}/scripts/slurm/_common.sh" ]; then
  # shellcheck source=scripts/slurm/_common.sh
  source "${SLURM_SUBMIT_DIR}/scripts/slurm/_common.sh"
else
  # shellcheck source=scripts/slurm/_common.sh
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
fi

# ── Run benchmark ───────────────────────────────────────────────────────────
python -m src.eval.benchmark \
    --config configs/base.yaml \
    --model "$MODEL" \
    --prompt-mode "$PROMPT_MODE" \
    --output "runs/slurm/nav4rail_${METHOD}_${MODEL}_${SLURM_ARRAY_JOB_ID}/"
