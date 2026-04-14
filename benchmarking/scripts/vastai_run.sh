#!/bin/bash
# ============================================================================
# vastai_run.sh — Launch NAV4RAIL benchmarks on a vast.ai GPU instance
# ============================================================================
# Requires: vastai CLI (pip install vastai), HF_TOKEN, optionally WANDB_API_KEY.
#
# Usage:
#   ./scripts/vastai_run.sh                              # all models, zero_shot
#   ./scripts/vastai_run.sh --model gemma2_9b            # single model
#   ./scripts/vastai_run.sh --method sft --model qwen25_14b
#   ./scripts/vastai_run.sh --gpu-ram 48 --query "gpu_name=RTX_4090"
#   DRY_RUN=1 ./scripts/vastai_run.sh                   # show search & commands only
# ============================================================================

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
METHOD="${METHOD:-zero_shot}"
PROMPT_MODE="${PROMPT_MODE:-$METHOD}"
MODEL="${MODEL:-}"          # empty = all models
GPU_RAM="${GPU_RAM:-24}"
DISK="${DISK:-60}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel}"
DRY_RUN="${DRY_RUN:-0}"

# vast.ai search filter: ≥$GPU_RAM GB VRAM, 1 GPU, reliable, affordable
SEARCH_QUERY="${SEARCH_QUERY:-gpu_ram>=${GPU_RAM} num_gpus=1 reliability>0.95 dph<=1.0 inet_down>200}"

ALL_MODELS=("mistral_7b" "llama3_8b" "qwen25_coder_7b" "gemma2_9b" "qwen25_14b")

# ── Parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)     MODEL="$2";     shift 2 ;;
        --method)    METHOD="$2";    PROMPT_MODE="$2"; shift 2 ;;
        --gpu-ram)   GPU_RAM="$2";   shift 2 ;;
        --disk)      DISK="$2";      shift 2 ;;
        --image)     IMAGE="$2";     shift 2 ;;
        --query)     SEARCH_QUERY="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=1;      shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Checks ──────────────────────────────────────────────────────────────────
if ! command -v vastai &>/dev/null; then
    echo "ERROR: vastai CLI not found. Install with: pip install vastai"
    echo "  Then: vastai set api-key <YOUR_KEY>"
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set — needed for gated models (gemma)."
    echo "  Export it or run: huggingface-cli login"
    exit 1
fi

# Resolve model list
if [ -n "$MODEL" ]; then
    MODELS=("$MODEL")
else
    MODELS=("${ALL_MODELS[@]}")
fi

# ── Script that runs on the instance ────────────────────────────────────────
BENCH_REPO="https://github.com/MMyster/FineTuningOnTelecomCluster.git"

make_onstart_script() {
    local model="$1"
    local method="$2"
    local prompt_mode="$3"
    cat <<ONSTART
#!/bin/bash
set -euo pipefail

echo "=== vast.ai NAV4RAIL benchmark: model=${model} method=${method} ==="

# Clone repo
cd /workspace
if [ ! -d benchmarking ]; then
    git clone --depth=1 ${BENCH_REPO} repo
    ln -s repo/benchmarking benchmarking
fi
cd benchmarking

# Venv
python -m venv /workspace/venv
source /workspace/venv/bin/activate

# Dependencies
pip install --quiet wheel setuptools
pip install --quiet -r requirements.txt

# Auth
export HF_TOKEN="${HF_TOKEN}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="nav4rail-bench"
export PYTHONPATH="\${PYTHONPATH:-}:/workspace/benchmarking"

# Run
python -m src.eval.benchmark \\
    --config configs/base.yaml \\
    --model "${model}" \\
    --prompt-mode "${prompt_mode}" \\
    --output "/workspace/results/nav4rail_${method}_${model}/"

echo "=== Done: model=${model} method=${method} ==="
# Save results artifact
tar czf /workspace/results_${model}.tar.gz -C /workspace/results .
echo "Results archived to /workspace/results_${model}.tar.gz"
ONSTART
}

# ── Launch instances ────────────────────────────────────────────────────────
echo "=== vast.ai NAV4RAIL Benchmark Launcher ==="
echo "Models:  ${MODELS[*]}"
echo "Method:  ${METHOD}"
echo "GPU RAM: >= ${GPU_RAM} GB"
echo "Image:   ${IMAGE}"
echo "Search:  ${SEARCH_QUERY}"
echo ""

# Search for offers
echo "--- Available offers ---"
vastai search offers "${SEARCH_QUERY}" --order dph --limit 5
echo ""

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN] Would launch ${#MODELS[@]} instance(s). Exiting."
    exit 0
fi

INSTANCE_IDS=()

for model in "${MODELS[@]}"; do
    echo "--- Launching instance for ${model} ---"

    onstart_script=$(make_onstart_script "$model" "$METHOD" "$PROMPT_MODE")

    instance_id=$(vastai create instance \
        --image "$IMAGE" \
        --disk "$DISK" \
        --env "-e HF_TOKEN=${HF_TOKEN} -e WANDB_API_KEY=${WANDB_API_KEY:-}" \
        --onstart-cmd "$onstart_script" \
        --search "${SEARCH_QUERY}" \
        --raw 2>&1 | grep -oP '"id":\s*\K[0-9]+' | head -1) || {
            echo "  FAILED to create instance for ${model}"
            continue
        }

    echo "  Instance ID: ${instance_id}"
    INSTANCE_IDS+=("${instance_id}:${model}")
done

echo ""
echo "=== Launched ${#INSTANCE_IDS[@]} instance(s) ==="
for entry in "${INSTANCE_IDS[@]}"; do
    IFS=: read -r id model <<< "$entry"
    echo "  ${model}: instance ${id}"
done
echo ""
echo "Monitor with:"
echo "  vastai show instances"
echo "  vastai logs <instance_id>"
echo ""
echo "Download results with:"
echo "  vastai scp <instance_id>:/workspace/results_<model>.tar.gz ."
echo ""
echo "Destroy when done:"
echo "  vastai destroy instance <instance_id>"
