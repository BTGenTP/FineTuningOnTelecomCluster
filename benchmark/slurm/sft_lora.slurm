#!/bin/bash
#SBATCH --job-name=nav4rail_sft_lora
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err
# Default GPU partition (override: sbatch --partition=3090 ...; verify name with sinfo -s)
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

# Slurm copies the batch script to /var/spool/slurmd/.../slurm_script; BASH_SOURCE then points there,
# so sibling files like _telecom_env.sh are not beside the running script. Use SLURM_SUBMIT_DIR.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  _SUBMIT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
  if [[ -f "$_SUBMIT/slurm/_telecom_env.sh" ]]; then
    SCRIPT_DIR="$_SUBMIT/slurm"
  elif [[ -f "$_SUBMIT/_telecom_env.sh" ]]; then
    SCRIPT_DIR="$_SUBMIT"
  else
    echo "[job] ERROR: _telecom_env.sh not found under SLURM_SUBMIT_DIR=$_SUBMIT (run sbatch from benchmark/ or benchmark/slurm/)." >&2
    exit 1
  fi
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_telecom_env.sh"

CONFIG="${CONFIG:-configs/sft_lora_p100.yaml}"
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
