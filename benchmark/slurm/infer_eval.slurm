#!/bin/bash
#SBATCH --job-name=nav4rail_infer_eval
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

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
