#!/bin/bash
#SBATCH --job-name=nav4rail_grpo
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate || true
export PYTHONPATH="$PWD"

CONFIG="${CONFIG:-configs/grpo.yaml}"
OUTROOT="${OUTROOT:-runs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
PROMPTS="${PROMPTS:-}"
N="${N:-50}"
GROUP_SIZE="${GROUP_SIZE:-4}"
EPOCHS="${EPOCHS:-1}"
KL_COEF="${KL_COEF:-0.02}"

ARGS=(--config "$CONFIG" --output-root "$OUTROOT" --n "$N" --group-size "$GROUP_SIZE" --epochs "$EPOCHS" --kl-coef "$KL_COEF")
if [[ -n "${PROMPTS}" ]]; then
  ARGS+=(--prompts "$PROMPTS")
fi

python3 scripts/grpo_train.py "${ARGS[@]}"

