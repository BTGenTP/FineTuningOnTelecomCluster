#!/bin/bash
# Transfer all LoRA adapters to cluster ~/adapters/ directory.
#
# Run from dev machine (benji@BOOK-4GN31RNJVF).
# Prerequisites: SSH access to rpi5 and gpu.
#
# Adapter sources:
#   - Llama 8B:      cluster ~/code/nav4rail_finetune/outputs/nav4rail_llama3_8b_lora_767530/lora_adapter
#   - Mistral 7B:    cluster ~/code/nav4rail_finetune/outputs/nav4rail_mistral_7b_lora_767716/lora_adapter
#   - Qwen 14B:      RPi5 ~/nav4rail_finetune_results/qwen14b/output_qwen25_14b/lora_adapter
#   - Gemma 9B:      RPi5 ~/nav4rail_finetune_results/gemma9b/lora_adapter
#   - Qwen Coder 7B: dataia25 ~/nav4rail/output_qwen25_coder_7b/lora_adapter (if finished)

set -euo pipefail

CLUSTER="gpu"
RPI5="rpi5"
ADAPTERS_DIR="~/adapters"

echo "=== Setting up adapters directory on cluster ==="
ssh "$CLUSTER" "mkdir -p $ADAPTERS_DIR/{llama3_8b,mistral_7b,qwen25_coder_7b,qwen25_14b,gemma2_9b}"

# ─── 1. Cluster-local adapters (symlink) ────────────────────────────────────
echo ""
echo "=== [1/5] Llama 3.1 8B (cluster → cluster) ==="
ssh "$CLUSTER" "
  SRC=~/code/nav4rail_finetune/outputs/nav4rail_llama3_8b_lora_767530/lora_adapter
  DST=$ADAPTERS_DIR/llama3_8b/lora_adapter
  if [ -d \"\$SRC\" ] && [ ! -e \"\$DST\" ]; then
    ln -s \"\$SRC\" \"\$DST\"
    echo 'Symlinked Llama 8B adapter'
  elif [ -e \"\$DST\" ]; then
    echo 'Already exists'
  else
    echo 'ERROR: Source not found'
  fi
"

echo ""
echo "=== [2/5] Mistral 7B (cluster → cluster) ==="
ssh "$CLUSTER" "
  SRC=~/code/nav4rail_finetune/outputs/nav4rail_mistral_7b_lora_767716/lora_adapter
  DST=$ADAPTERS_DIR/mistral_7b/lora_adapter
  if [ -d \"\$SRC\" ] && [ ! -e \"\$DST\" ]; then
    ln -s \"\$SRC\" \"\$DST\"
    echo 'Symlinked Mistral 7B adapter'
  elif [ -e \"\$DST\" ]; then
    echo 'Already exists'
  else
    echo 'ERROR: Source not found'
  fi
"

# ─── 2. RPi5 adapters → cluster via SCP ─────────────────────────────────────
echo ""
echo "=== [3/5] Qwen 2.5 14B (RPi5 → cluster) ==="
echo "This will transfer ~263 MB..."
# RPi5 → dev → cluster (two-hop SCP via piping)
ssh "$RPI5" "tar czf - -C ~/nav4rail_finetune_results/qwen14b/output_qwen25_14b lora_adapter" \
  | ssh "$CLUSTER" "cd $ADAPTERS_DIR/qwen25_14b && tar xzf -"
echo "Done: Qwen 14B"

echo ""
echo "=== [4/5] Gemma 2 9B (RPi5 → cluster) ==="
echo "This will transfer ~228 MB..."
ssh "$RPI5" "tar czf - -C ~/nav4rail_finetune_results/gemma9b lora_adapter" \
  | ssh "$CLUSTER" "cd $ADAPTERS_DIR/gemma2_9b && tar xzf -"
echo "Done: Gemma 9B"

# ─── 3. Qwen Coder 7B (dataia25 → cluster if available) ─────────────────────
echo ""
echo "=== [5/5] Qwen 2.5 Coder 7B (dataia25 → cluster) ==="
# Check if training is finished (lora_adapter dir exists)
if ssh dataia25 "test -d ~/nav4rail/output_qwen25_coder_7b/lora_adapter" 2>/dev/null; then
    echo "Adapter found on dataia25, transferring ~243 MB..."
    ssh dataia25 "tar czf - -C ~/nav4rail/output_qwen25_coder_7b lora_adapter" \
      | ssh "$CLUSTER" "cd $ADAPTERS_DIR/qwen25_coder_7b && tar xzf -"
    echo "Done: Qwen Coder 7B"
else
    echo "SKIP: Qwen Coder 7B not finished or lora_adapter not found on dataia25."
    echo "Re-run this section later when training completes."
fi

echo ""
echo "=== Summary ==="
ssh "$CLUSTER" "echo 'Adapters on cluster:' && ls -la $ADAPTERS_DIR/*/lora_adapter/ 2>/dev/null | head -30"
