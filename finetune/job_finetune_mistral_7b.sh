#!/bin/bash
#SBATCH --job-name=nav4rail_ft_mistral
#SBATCH --output=nav4rail_ft_mistral_%j.out
#SBATCH --error=nav4rail_ft_mistral_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mem=48G

set -euo pipefail

module load python/3.11.13 cuda/12.4.1 || true

WORK_DIR="$HOME/code/nav4rail_finetune"
VENV_DIR="$HOME/venvs/nav4rail_llama3"
DATASET="$WORK_DIR/dataset_nav4rail_llm_2000.jsonl"
OUT_DIR="$WORK_DIR/outputs/nav4rail_mistral_7b_lora_${SLURM_JOB_ID:-local}"

echo "================================================================"
echo "[job] host=$(hostname) jobid=${SLURM_JOB_ID:-unknown}"
echo "[job] date=$(date -Iseconds)"
echo "[job] GPU info:"
nvidia-smi || true
echo "================================================================"

cd "$WORK_DIR"

# ─── Venv (reuse llama3 venv, same deps) ─────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "[job] Creating venv: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ─── HuggingFace cache ──────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$HF_HUB_CACHE" "$PIP_CACHE_DIR"

# ─── Dependencies (already installed if llama3 ran first) ────────────────────
pip install --upgrade pip
pip install \
  "torch==2.3.0" \
  "transformers==4.44.0" \
  "peft==0.12.0" \
  "trl==0.10.1" \
  "bitsandbytes==0.43.3" \
  "datasets==2.21.0" \
  "accelerate==0.34.0" \
  "scipy" "sentencepiece" "protobuf" "rich"

# ─── Verify dataset ─────────────────────────────────────────────────────────
if [ ! -f "$DATASET" ]; then
  echo "[job] ERROR: dataset not found: $DATASET"
  exit 1
fi
NLINES=$(wc -l < "$DATASET")
echo "[job] Dataset: $DATASET ($NLINES samples)"

# ─── Train ───────────────────────────────────────────────────────────────────
echo "[job] Starting Mistral 7B fine-tuning..."
echo "[job] Output dir: $OUT_DIR"

python3 finetune_llama3_nav4rail.py \
  --model mistral_7b \
  --dataset "$DATASET" \
  --output-dir "$OUT_DIR" \
  --epochs 10 \
  --lr 2e-4 \
  --batch-size 2 \
  --grad-accum 8 \
  --max-seq-len 8192 \
  --lora-r 16

echo "================================================================"
echo "[job] Training complete!"
echo "[job] Adapter saved: $OUT_DIR/lora_adapter"
echo "[job] date=$(date -Iseconds)"
echo "================================================================"
