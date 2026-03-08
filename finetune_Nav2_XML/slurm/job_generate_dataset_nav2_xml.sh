#!/bin/bash
#SBATCH --job-name=nav2_xml_ds
#SBATCH --output=nav2_xml_dataset_%j.out
#SBATCH --error=nav2_xml_dataset_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:0
#SBATCH --time=00:10:00
#SBATCH --mem=8G

set -euo pipefail

module load python/3.11.13 cuda/12.4.1 || true

WORK_DIR="$HOME/code/nav4rail_finetune_nav2"
VENV_DIR="$HOME/venvs/nav4rail_nav2_xml"

echo "[job] host=$(hostname) jobid=${SLURM_JOB_ID:-unknown} date=$(date -Iseconds)"

cd "$WORK_DIR"

if [ ! -f "$WORK_DIR/finetune_Nav2_XML/requirements.txt" ]; then
  echo "[job] ERROR: missing $WORK_DIR/finetune_Nav2_XML/requirements.txt"
  echo "[job] You likely copied finetune_Nav2_XML/* instead of the finetune_Nav2_XML/ folder."
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR"

python3 -m pip install --upgrade pip
python3 -m pip install -r finetune_Nav2_XML/requirements.txt

OUT_PATH="${OUT_PATH:-$WORK_DIR/finetune_Nav2_XML/dataset_out/dataset_nav2_bt_xml.jsonl}"
N_SAMPLES="${N_SAMPLES:-2000}"
SEED="${SEED:-42}"

mkdir -p "$(dirname "$OUT_PATH")"

python3 -m finetune_Nav2_XML.dataset.generate_dataset_nav2_bt_xml \
  --out "$OUT_PATH" \
  --n "$N_SAMPLES" \
  --seed "$SEED"

echo "[job] wrote dataset: $OUT_PATH"

