#!/bin/bash
#SBATCH --job-name=nav4rail_ft_llama3
#SBATCH --output=nav4rail_ft_llama3_%j.out
#SBATCH --error=nav4rail_ft_llama3_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mem=48G

set -euo pipefail

module load python/3.11.13 cuda/12.4.1 || true

WORK_DIR="$HOME/code/nav4rail_finetune"
VENV_DIR="$HOME/venvs/nav4rail_llama3"
DATASET="$WORK_DIR/dataset_nav4rail_llm_2000.jsonl"
OUT_DIR="$WORK_DIR/outputs/nav4rail_llama3_8b_lora_${SLURM_JOB_ID:-local}"

echo "================================================================"
echo "[job] host=$(hostname) jobid=${SLURM_JOB_ID:-unknown}"
echo "[job] date=$(date -Iseconds)"
echo "[job] GPU info:"
nvidia-smi || true
echo "================================================================"

cd "$WORK_DIR"

# ─── Venv ────────────────────────────────────────────────────────────────────
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

# ─── Dependencies (flock for concurrent safety) ────────────────────────────
echo "[job] Installing dependencies..."
(
  flock -w 600 200 || { echo "[job] Could not acquire pip lock"; exit 1; }
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
) 200>"$VENV_DIR/.pip_lock"

# ─── Verify dataset ─────────────────────────────────────────────────────────
if [ ! -f "$DATASET" ]; then
  echo "[job] ERROR: dataset not found: $DATASET"
  exit 1
fi
NLINES=$(wc -l < "$DATASET")
echo "[job] Dataset: $DATASET ($NLINES samples)"

# ─── Train ───────────────────────────────────────────────────────────────────
echo "[job] Starting fine-tuning..."
echo "[job] Output dir: $OUT_DIR"

python3 finetune_llama3_nav4rail.py \
  --model llama3_8b \
  --dataset "$DATASET" \
  --output-dir "$OUT_DIR" \
  --epochs 10 \
  --lr 2e-4 \
  --batch-size 1 \
  --grad-accum 16 \
  --max-seq-len 8192 \
  --lora-r 16

echo "================================================================"
echo "[job] Training complete!"
echo "[job] Adapter saved: $OUT_DIR/lora_adapter"
echo "[job] date=$(date -Iseconds)"
echo "================================================================"
