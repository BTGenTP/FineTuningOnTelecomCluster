#!/bin/bash
# ============================================================================
# NAV4RAIL Benchmark — SLURM Job Array: All 5 Models
# ============================================================================
# Runs the same method on all 5 models in parallel.
# Usage:
#   METHOD=sft PROMPT_MODE=zero_shot sbatch scripts/slurm/array_models.sh
#   sbatch --partition=P100 scripts/slurm/array_models.sh  # 7B models only
#
# Grammar-constrained decoding:
#   CONSTRAINT=none    METHOD=zero_shot sbatch scripts/slurm/array_models.sh   # baseline
#   CONSTRAINT=gbnf    METHOD=zero_shot sbatch scripts/slurm/array_models.sh   # transformers-cfg
#   CONSTRAINT=outlines METHOD=zero_shot sbatch scripts/slurm/array_models.sh  # outlines JSON
#   CONSTRAINT=all     METHOD=zero_shot sbatch scripts/slurm/array_models.sh   # all 3 sequentially per model

#SBATCH --job-name=nav4rail_array
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=runs/slurm/nav4rail_array_%A_%a/slurm_%A_%a.out
#SBATCH --error=runs/slurm/nav4rail_array_%A_%a/slurm_%A_%a.err

# MODELS=("mistral_7b" "llama3_8b" "qwen25_coder_7b" "gemma2_9b" "qwen25_14b") # change --array=0-4 to run all 5 models
MODELS=("mistral_7b" "llama3_8b" "qwen25_coder_7b")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
METHOD=${METHOD:-zero_shot}
PROMPT_MODE=${PROMPT_MODE:-$METHOD}
CONSTRAINT=${CONSTRAINT:-none}

echo "Array task $SLURM_ARRAY_TASK_ID: model=$MODEL method=$METHOD constraint=$CONSTRAINT"

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
    --constraint "$CONSTRAINT" \
    --output "runs/slurm/nav4rail_${METHOD}_${CONSTRAINT}_${MODEL}_${SLURM_ARRAY_JOB_ID}/"
