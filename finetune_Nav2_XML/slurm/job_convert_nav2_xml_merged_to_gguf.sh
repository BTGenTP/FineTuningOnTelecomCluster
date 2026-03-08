#!/bin/bash
#SBATCH --job-name=nav2_xml_merged_to_gguf
#SBATCH --output=nav2_xml_gguf_%j.out
#SBATCH --error=nav2_xml_gguf_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:0
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#
# Run this job AFTER job_merge_nav2_xml_mistral7b_lora_adapter.sh (merged model must exist in OUT_DIR).
# Same WORK_DIR / OUT_DIR / VENV_DIR as the merge job (override via env when submitting).

set -euo pipefail

module load python/3.11.13 cuda/12.4.1 || true

echo "[job] host=$(hostname) jobid=${SLURM_JOB_ID:-unknown} date=$(date -Iseconds)"

WORK_DIR="${WORK_DIR:-$HOME/models/Mistral-7B-Instruct-v02-Lora-Nav2BT-XML}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/nav4rail_nav2_xml_merge}"
OUT_DIR="${OUT_DIR:-$WORK_DIR/merged}"
GGUF_OUTFILE="${GGUF_OUTFILE:-nav2_xml_mistral7b_q4_k_m.gguf}"

export OUT_DIR GGUF_OUTFILE
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ ! -d "$OUT_DIR" ] || [ ! -f "$OUT_DIR/config.json" ]; then
  echo "[job] ERROR: Merged model not found at $OUT_DIR. Run job_merge_nav2_xml_mistral7b_lora_adapter.sh first."
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

export PIP_DISABLE_PIP_VERSION_CHECK=1
python3 -m pip install --upgrade pip
python3 -m pip install -q gguf numpy sentencepiece protobuf

echo "[1/2] Setting up llama.cpp converter..."
if [ ! -d "llama.cpp" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp
fi

echo "[2/2] Converting merged model to GGUF Q4_K_M..."
python3 llama.cpp/convert_hf_to_gguf.py "$OUT_DIR" \
  --outtype q4_k_m \
  --outfile "$GGUF_OUTFILE"

echo "========================================"
echo "Done! GGUF file:"
ls -lh "$WORK_DIR/$GGUF_OUTFILE"
echo "========================================"
