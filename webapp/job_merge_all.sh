#!/bin/bash
#SBATCH --job-name=nav4rail_merge_all
#SBATCH --output=merge_all_%j.out
#SBATCH --error=merge_all_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#
# Merge all 5 LoRA adapters into base models and convert to GGUF Q4_K_M.
# Qwen 14B needs ~56 GB RAM for fp16 merge, hence --mem=64G.
#
# Usage:
#   sbatch job_merge_all.sh
#
# Prerequisites:
#   - All 5 adapters must be under ~/adapters/ (see ADAPTERS array below)
#   - pip install peft transformers torch accelerate gguf numpy sentencepiece protobuf

set -uo pipefail  # No -e: individual model failures are handled gracefully

module load python/3.11.13 cuda/12.4.1 cmake/4.1.0 gcc/11.5.0 || true

VENV=~/venv_nav4rail
source "$VENV/bin/activate"

WORK_DIR=~/code/nav4rail_finetune
OUTPUT_DIR=~/models
MERGE_SCRIPT="$WORK_DIR/merge_and_convert.py"

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Job ID   : ${SLURM_JOB_ID:-local}"
echo "Date     : $(date)"
echo "Host     : $(hostname)"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# ─── Setup llama.cpp converter ───────────────────────────────────────────────
cd "$WORK_DIR"
if [ ! -d "llama.cpp" ]; then
    echo "[setup] Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp
fi
# Ensure transformers >= 4.43 for Llama 3.1 rope_type support
pip install -q 'transformers>=4.43' accelerate gguf numpy sentencepiece protobuf 2>/dev/null || true

# HF token for gated models (Gemma, Llama)
export HF_TOKEN="${HF_TOKEN:-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}"

# Build llama-quantize if not present
if [ ! -f "llama.cpp/build/bin/llama-quantize" ]; then
    echo "[setup] Building llama-quantize..."
    cd llama.cpp && mkdir -p build && cd build
    cmake .. -DGGML_CUDA=OFF -DLLAMA_BUILD_SERVER=OFF 2>/dev/null
    make -j$(nproc) llama-quantize 2>/dev/null || cmake --build . --target llama-quantize -j$(nproc) 2>/dev/null
    cd "$WORK_DIR"
    echo "[setup] llama-quantize ready"
fi
QUANTIZE_BIN="$WORK_DIR/llama.cpp/build/bin/llama-quantize"

# ─── Model definitions ──────────────────────────────────────────────────────
# Format: "short_name|adapter_path"
# Base model is auto-detected from adapter_config.json

declare -a MODELS=(
    "mistral-7b|$HOME/adapters/mistral_7b/lora_adapter"
    "llama3-8b|$HOME/adapters/llama3_8b/lora_adapter"
    "qwen-coder-7b|$HOME/adapters/qwen25_coder_7b/lora_adapter"
    "qwen-14b|$HOME/adapters/qwen25_14b/lora_adapter"
    "gemma2-9b|$HOME/adapters/gemma2_9b/lora_adapter"
)

TOTAL=${#MODELS[@]}
DONE=0
FAILED=0

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name adapter_path <<< "$entry"
    gguf_name="nav4rail-${name}-q4_k_m.gguf"
    gguf_path="$OUTPUT_DIR/$gguf_name"
    merged_dir="$WORK_DIR/merged_${name}"

    echo ""
    echo "════════════════════════════════════════"
    echo "[$((DONE + FAILED + 1))/$TOTAL] Processing: $name"
    echo "  Adapter : $adapter_path"
    echo "  Output  : $gguf_path"
    echo "════════════════════════════════════════"

    # Skip if GGUF already exists
    if [ -f "$gguf_path" ]; then
        echo "[skip] $gguf_name already exists, skipping."
        DONE=$((DONE + 1))
        continue
    fi

    # Skip if adapter doesn't exist
    if [ ! -d "$adapter_path" ]; then
        echo "[SKIP] Adapter not found: $adapter_path"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Step 1: Merge LoRA into base model
    echo "[1/2] Merging LoRA adapter..."
    if ! python3 "$MERGE_SCRIPT" \
        --adapter-path "$adapter_path" \
        --output-dir "$merged_dir"; then
        echo "[FAIL] Merge failed for $name"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Step 2: Convert to GGUF f16 first
    gguf_f16="$WORK_DIR/temp_${name}_f16.gguf"
    echo "[2/3] Converting to GGUF f16..."
    if ! python3 llama.cpp/convert_hf_to_gguf.py "$merged_dir" \
        --outtype f16 \
        --outfile "$gguf_f16"; then
        echo "[FAIL] GGUF f16 conversion failed for $name"
        FAILED=$((FAILED + 1))
        continue
    fi

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
    rm -f "$gguf_f16"

    # Cleanup merged model (saves disk space)
    echo "[cleanup] Removing merged HF model..."
    rm -rf "$merged_dir"

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
