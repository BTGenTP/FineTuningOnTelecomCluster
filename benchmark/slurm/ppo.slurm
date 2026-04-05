#!/bin/bash
#SBATCH --job-name=nav4rail_ppo
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

CONFIG="${CONFIG:-configs/ppo.yaml}"
OUTROOT="${OUTROOT:-runs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
PROMPTS="${PROMPTS:-}"
N="${N:-50}"
EPOCHS="${EPOCHS:-1}"

ARGS=(--config "$CONFIG" --output-root "$OUTROOT" --n "$N" --epochs "$EPOCHS")
if [[ -n "${PROMPTS}" ]]; then
  ARGS+=(--prompts "$PROMPTS")
fi

python3 scripts/ppo_train.py "${ARGS[@]}"

