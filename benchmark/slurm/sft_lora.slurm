#!/bin/bash
#SBATCH --job-name=nav4rail_sft_lora
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

CONFIG="${CONFIG:-configs/sft_lora.yaml}"
OUTROOT="${OUTROOT:-runs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
DATASET="${DATASET:-}"
N_TRAIN="${N_TRAIN:-500}"
WRITE_DATASET="${WRITE_DATASET:-data/dataset_synthetic.jsonl}"

ARGS=(--config "$CONFIG" --output-root "$OUTROOT")
if [[ -n "${DATASET}" ]]; then
  ARGS+=(--dataset "$DATASET")
else
  ARGS+=(--generate-synthetic "$N_TRAIN" --write-dataset "$WRITE_DATASET")
fi

python3 scripts/sft_train.py "${ARGS[@]}"

