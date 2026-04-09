#!/bin/bash
# Fine-tune Gemma 2 9B IT on dataia25 GPU 1
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export HF_HOME="$HOME/nav4rail/hf_cache"
mkdir -p "$HF_HOME"

cd "$HOME/nav4rail"
source venv/bin/activate

DATASET="$HOME/nav4rail/dataset_nav4rail_llm_2000.jsonl"
OUT_DIR="$HOME/nav4rail/outputs/nav4rail_gemma2_9b_lora"

echo "================================================================"
echo "[job] host=$(hostname) date=$(date -Iseconds)"
echo "[job] GPU:"
nvidia-smi -i 1 --query-gpu=name,memory.total,temperature.gpu --format=csv,noheader
echo "[job] Dataset: $DATASET"
echo "[job] Output: $OUT_DIR"
echo "================================================================"

python3 finetune_llama3_nav4rail.py \
  --model gemma2_9b \
  --dataset "$DATASET" \
  --output-dir "$OUT_DIR" \
  --epochs 10 \
  --lr 2e-4 \
  --batch-size 1 \
  --grad-accum 16 \
  --max-seq-len 8192 \
  --lora-r 16

echo "================================================================"
echo "[job] Training complete! $(date -Iseconds)"
echo "[job] Adapter: $OUT_DIR/lora_adapter"
