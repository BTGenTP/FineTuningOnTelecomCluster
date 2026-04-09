#!/bin/bash
# Merge LoRA adapters and convert to GGUF Q4_K_M on dataia25.
# dataia25 has 2× RTX 3090, 64+ GB RAM, and recent transformers.
#
# Run in tmux on dataia25:
#   tmux new -s merge
#   bash merge_all_dataia25.sh
#
# After completion, transfer GGUFs to cluster:
#   for f in ~/nav4rail/models/nav4rail-*.gguf; do
#     scp "$f" gpu:~/models/
#   done

set -uo pipefail

WORK_DIR=~/nav4rail
VENV_DIR=~/nav4rail/venv
OUTPUT_DIR=~/nav4rail/models
MERGE_SCRIPT="$WORK_DIR/merge_and_convert.py"

cd "$WORK_DIR"
source "$VENV_DIR/bin/activate"

mkdir -p "$OUTPUT_DIR"

# Install deps
pip install -q gguf numpy sentencepiece protobuf accelerate 2>/dev/null || true

# Setup llama.cpp
if [ ! -d "llama.cpp" ]; then
    echo "[setup] Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp
fi

# Build llama-quantize
if [ ! -f "llama.cpp/build/bin/llama-quantize" ]; then
    echo "[setup] Building llama-quantize..."
    cd llama.cpp && mkdir -p build && cd build
    cmake .. -DGGML_CUDA=OFF -DLLAMA_BUILD_SERVER=OFF 2>/dev/null
    cmake --build . --target llama-quantize -j$(nproc) 2>/dev/null
    cd "$WORK_DIR"
fi
QUANTIZE_BIN="$WORK_DIR/llama.cpp/build/bin/llama-quantize"

# HF token for gated models
export HF_TOKEN="${HF_TOKEN:-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}"

echo "========================================"
echo "dataia25 merge — $(date)"
echo "========================================"

# ─── Adapter locations on dataia25 ───────────────────────────────────────────
# Local:
#   - Qwen Coder 7B: ~/nav4rail/output_qwen25_coder_7b/lora_adapter
# Transferred from RPi5 → ~/nav4rail/adapters/:
#   - Mistral 7B, Llama 8B, Qwen 14B, Gemma 9B

declare -a MODELS=(
    "mistral-7b|$HOME/nav4rail/adapters/mistral_7b"
    "llama3-8b|$HOME/nav4rail/adapters/llama3_8b"
    "qwen-coder-7b|$HOME/nav4rail/output_qwen25_coder_7b/lora_adapter"
    "qwen-14b|$HOME/nav4rail/adapters/qwen25_14b"
    "gemma2-9b|$HOME/nav4rail/adapters/gemma2_9b"
)

TOTAL=${#MODELS[@]}
DONE=0
FAILED=0

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name adapter_path <<< "$entry"
    gguf_name="nav4rail-${name}-q4_k_m.gguf"
    gguf_path="$OUTPUT_DIR/$gguf_name"
    merged_dir="$WORK_DIR/merged_${name}"
    gguf_f16="$WORK_DIR/temp_${name}_f16.gguf"

    echo ""
    echo "════════════════════════════════════════"
    echo "[$((DONE + FAILED + 1))/$TOTAL] Processing: $name"
    echo "  Adapter : $adapter_path"
    echo "  Output  : $gguf_path"
    echo "════════════════════════════════════════"

    if [ -f "$gguf_path" ]; then
        echo "[skip] $gguf_name already exists."
        DONE=$((DONE + 1))
        continue
    fi

    if [ ! -d "$adapter_path" ]; then
        echo "[SKIP] Adapter not found: $adapter_path"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Step 1: Merge LoRA into base model
    echo "[1/3] Merging LoRA adapter..."
    if ! python3 "$MERGE_SCRIPT" \
        --adapter-path "$adapter_path" \
        --output-dir "$merged_dir"; then
        echo "[FAIL] Merge failed for $name"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Step 2: Convert to GGUF f16
    echo "[2/3] Converting to GGUF f16..."
    if ! python3 llama.cpp/convert_hf_to_gguf.py "$merged_dir" \
        --outtype f16 \
        --outfile "$gguf_f16"; then
        echo "[FAIL] GGUF f16 conversion failed for $name"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Free disk space: remove merged dir now that GGUF f16 exists
    rm -rf "$merged_dir"

    # Step 3: Quantize f16 → Q4_K_M
    echo "[3/3] Quantizing to Q4_K_M..."
    if [ -f "$QUANTIZE_BIN" ]; then
        if ! "$QUANTIZE_BIN" "$gguf_f16" "$gguf_path" Q4_K_M; then
            echo "[FAIL] Quantization failed for $name"
            rm -f "$gguf_f16"
            FAILED=$((FAILED + 1))
            continue
        fi
    else
        echo "[WARN] llama-quantize not available, keeping f16"
        mv "$gguf_f16" "$gguf_path"
    fi

    # Cleanup
    rm -f "$gguf_f16"

    DONE=$((DONE + 1))
    echo "[OK] $gguf_name created:"
    ls -lh "$gguf_path"
done

echo ""
echo "========================================"
echo "Summary: $DONE succeeded, $FAILED failed out of $TOTAL"
echo ""
echo "GGUF files in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR"/nav4rail-*.gguf 2>/dev/null || echo "(none)"
echo "========================================"
echo ""
echo "Next: transfer GGUFs to cluster with:"
echo "  for f in $OUTPUT_DIR/nav4rail-*.gguf; do scp \"\$f\" gpu:~/models/; done"
