#!/bin/bash
#SBATCH --job-name=nav4rail_infer_eval
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate || true
export PYTHONPATH="$PWD"

CONFIG="${CONFIG:-configs/prompt_zero_shot.yaml}"
OUTROOT="${OUTROOT:-runs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
MISSIONS="${MISSIONS:-}"
N="${N:-50}"
ADAPTER="${ADAPTER:-}"

ARGS=(--config "$CONFIG" --output-root "$OUTROOT" --n "$N")
if [[ -n "${MISSIONS}" ]]; then
  ARGS+=(--missions "$MISSIONS")
fi
if [[ -n "${ADAPTER}" ]]; then
  ARGS+=(--adapter "$ADAPTER")
fi

python3 scripts/infer_eval.py "${ARGS[@]}"

