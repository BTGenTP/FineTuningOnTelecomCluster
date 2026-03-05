#!/bin/bash
#SBATCH --job-name=nav2_ds
#SBATCH --output=nav2_dataset_%j.out
#SBATCH --error=nav2_dataset_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:0
#SBATCH --time=00:10:00
#SBATCH --mem=8G

set -euo pipefail

module load python/3.11.13 cuda/12.4.1 || true

WORK_DIR="$HOME/code/nav4rails/repositories/FineTuningOnTelecomCluster"
VENV_DIR="$HOME/venvs/nav4rail_nav2_steps"

echo "[job] host=$(hostname) jobid=${SLURM_JOB_ID:-unknown} date=$(date -Iseconds)"

cd "$WORK_DIR"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip
python3 -m pip install -r finetune_Nav2/requirements.txt

OUT_PATH="${OUT_PATH:-$WORK_DIR/finetune_Nav2/dataset_out/dataset_nav2_steps.jsonl}"
N_SAMPLES="${N_SAMPLES:-2000}"
SEED="${SEED:-42}"

mkdir -p "$(dirname "$OUT_PATH")"

python3 -m finetune_Nav2.dataset.generate_dataset_nav2_steps \
  --out "$OUT_PATH" \
  --n "$N_SAMPLES" \
  --seed "$SEED"

echo "[job] wrote dataset: $OUT_PATH"

