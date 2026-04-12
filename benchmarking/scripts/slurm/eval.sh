#!/bin/bash
# ============================================================================
# NAV4RAIL Benchmark — SLURM Evaluation Job (inference only)
# ============================================================================
# Usage:
#   MODEL=gemma2_9b PROMPT_MODE=zero_shot sbatch scripts/slurm/eval.sh
#   MODEL=mistral_7b ADAPTER=runs/sft_mistral_7b/final_adapter sbatch scripts/slurm/eval.sh
#   sbatch --partition=3090 scripts/slurm/eval.sh   # override partition

#SBATCH --job-name=nav4rail_eval
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm/nav4rail_eval_%j/slurm_%j.out
#SBATCH --error=runs/slurm/nav4rail_eval_%j/slurm_%j.err

MODEL=${MODEL:-mistral_7b}
CONFIG=${CONFIG:-configs/base.yaml}
PROMPT_MODE=${PROMPT_MODE:-zero_shot}
ADAPTER=${ADAPTER:-}

echo "=== NAV4RAIL Evaluation ==="
echo "Model:       ${MODEL}"
echo "Prompt mode: ${PROMPT_MODE}"
echo "Adapter:     ${ADAPTER:-<none>}"

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
RUN_DIR="runs/slurm/nav4rail_eval_${PROMPT_MODE}_${MODEL}_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

export WANDB_PROJECT=nav4rail-bench

# ── Adapter flag ────────────────────────────────────────────────────────────
ADAPTER_FLAG=""
if [ -n "$ADAPTER" ]; then
    ADAPTER_FLAG="--adapter $ADAPTER"
fi

# ── Run benchmark ───────────────────────────────────────────────────────────
python -m src.eval.benchmark \
    --config "$CONFIG" \
    --model "$MODEL" \
    --prompt-mode "$PROMPT_MODE" \
    $ADAPTER_FLAG \
    --output "$RUN_DIR/" \
    2>&1 | tee "$RUN_DIR/eval.log"

echo "=== Evaluation complete ==="
