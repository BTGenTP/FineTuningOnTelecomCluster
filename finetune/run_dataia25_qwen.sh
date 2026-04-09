#!/bin/bash
# Fine-tune Qwen2.5-Coder-7B-Instruct on dataia25 GPU 0
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$HOME/nav4rail/hf_cache"
mkdir -p "$HF_HOME"

cd "$HOME/nav4rail"
source venv/bin/activate

DATASET="$HOME/nav4rail/dataset_nav4rail_llm_2000.jsonl"
OUT_DIR="$HOME/nav4rail/outputs/nav4rail_qwen25_coder_7b_lora"

echo "================================================================"
echo "[job] host=$(hostname) date=$(date -Iseconds)"
echo "[job] GPU:"
nvidia-smi -i 0 --query-gpu=name,memory.total,temperature.gpu --format=csv,noheader
echo "[job] Dataset: $DATASET"
echo "[job] Output: $OUT_DIR"
echo "================================================================"

python3 finetune_llama3_nav4rail.py \
  --model qwen25_coder_7b \
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
