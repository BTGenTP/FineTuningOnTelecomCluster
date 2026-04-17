#!/bin/bash
# ============================================================================
# vastai_run.sh — Launch NAV4RAIL benchmarks on vast.ai GPU instances
# ============================================================================
# Equivalent of SLURM array_models.sh but for vast.ai on-demand GPUs.
# Launches one instance per model in parallel, each running `python -m src.eval.benchmark`.
#
# Requires: vastai CLI (pip install vastai), HF_TOKEN, optionally WANDB_API_KEY.
#
# Usage:
#   ./scripts/vastai_run.sh                              # all 5 models, zero_shot
#   ./scripts/vastai_run.sh --model gemma2_9b            # single model
#   ./scripts/vastai_run.sh --method sft --model qwen25_14b
#   ./scripts/vastai_run.sh --gpu-ram 48 --query "gpu_name=RTX_4090"
#   ./scripts/vastai_run.sh --dry-run                    # show search, no launch
# ============================================================================

set -euo pipefail

# ── Repo context ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${BENCH_DIR}/configs/base.yaml"
INSTANCES_LOG="${BENCH_DIR}/runs/vastai/instances.log"
ONSTART_TMPDIR="${BENCH_DIR}/runs/vastai/onstart"
mkdir -p "$(dirname "$INSTANCES_LOG")" "$ONSTART_TMPDIR"

# ── Defaults ────────────────────────────────────────────────────────────────
METHOD="${METHOD:-zero_shot}"
PROMPT_MODE="${PROMPT_MODE:-$METHOD}"
MODEL="${MODEL:-}"          # empty = all models
DISK="${DISK:-60}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel}"
DRY_RUN="${DRY_RUN:-0}"
GIT_REPO="${GIT_REPO:-https://github.com/MMyster/FineTuningOnTelecomCluster.git}"
GIT_BRANCH="${GIT_BRANCH:-main}"
EXTRA_QUERY="${EXTRA_QUERY:-reliability>0.95 dph<=1.0 inet_down>200}"

ALL_MODELS=("mistral_7b" "llama3_8b" "qwen25_coder_7b" "gemma2_9b" "qwen25_14b")

# ── Parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)     MODEL="$2";     shift 2 ;;
        --method)    METHOD="$2";    PROMPT_MODE="$2"; shift 2 ;;
        --prompt-mode) PROMPT_MODE="$2"; shift 2 ;;
        --disk)      DISK="$2";      shift 2 ;;
        --image)     IMAGE="$2";     shift 2 ;;
        --query)     EXTRA_QUERY="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=1;      shift ;;
        --repo)      GIT_REPO="$2";  shift 2 ;;
        --branch)    GIT_BRANCH="$2"; shift 2 ;;
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

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 is required to parse vastai JSON output."
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set — needed for gated models (gemma)."
    echo "  Export it or run: huggingface-cli login"
    exit 1
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY not set — inference metrics won't stream to W&B."
fi

# Resolve model list
if [ -n "$MODEL" ]; then
    MODELS=("$MODEL")
else
    MODELS=("${ALL_MODELS[@]}")
fi

# ── Helper: read min_gpu_mem for a model from base.yaml ─────────────────────
get_min_gpu_mem() {
    local model_key="$1"
    python3 - <<PY
import sys
try:
    import yaml
except ImportError:
    print(24); sys.exit(0)
try:
    with open("$CONFIG_FILE") as f:
        cfg = yaml.safe_load(f)
    print(cfg.get("models", {}).get("$model_key", {}).get("min_gpu_mem", 24))
except Exception:
    print(24)
PY
}

# ── Helper: search for the cheapest available offer ─────────────────────────
# Args: gpu_ram, exclude_ids (space-separated)
# Returns: offer_id on stdout (or empty if none)
find_offer() {
    local gpu_ram="$1"
    local exclude="${2:-}"
    local query="gpu_ram>=${gpu_ram} num_gpus=1 ${EXTRA_QUERY}"

    vastai search offers "$query" --order dph --raw 2>/dev/null | python3 - "$exclude" <<'PY'
import json, sys
exclude = set(sys.argv[1].split()) if len(sys.argv) > 1 and sys.argv[1] else set()
try:
    offers = json.loads(sys.stdin.read())
except json.JSONDecodeError:
    sys.exit(0)
for offer in offers:
    if str(offer.get("id")) not in exclude:
        print(offer["id"])
        break
PY
}

# ── Helper: write onstart script to a temp file ─────────────────────────────
write_onstart() {
    local model="$1"
    local method="$2"
    local prompt_mode="$3"
    local out_file="$4"

    cat > "$out_file" <<ONSTART
#!/bin/bash
set -euo pipefail
exec > /workspace/onstart.log 2>&1

echo "=== vast.ai NAV4RAIL benchmark ==="
echo "Model:       ${model}"
echo "Method:      ${method}"
echo "Prompt mode: ${prompt_mode}"
echo "Date:        \$(date)"
echo "GPU:         \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

cd /workspace
if [ ! -d repo ]; then
    git clone --depth=1 --branch "${GIT_BRANCH}" "${GIT_REPO}" repo
fi
cd repo/benchmarking

python -m venv /workspace/venv
source /workspace/venv/bin/activate

pip install --quiet --disable-pip-version-check wheel setuptools
pip install --quiet --disable-pip-version-check -r requirements.txt

export HF_TOKEN="${HF_TOKEN}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="nav4rail-bench"
export WANDB_RUN_GROUP="vastai_${method}_\$(date +%Y%m%d)"
export PYTHONPATH="\${PYTHONPATH:-}:/workspace/repo/benchmarking"

mkdir -p /workspace/results
python -m src.eval.benchmark \\
    --config configs/base.yaml \\
    --model "${model}" \\
    --prompt-mode "${prompt_mode}" \\
    --output "/workspace/results/nav4rail_${method}_${model}/"

echo "=== Done — archiving results ==="
tar czf /workspace/results_${model}.tar.gz -C /workspace/results .
echo "Results at /workspace/results_${model}.tar.gz"
ONSTART

    chmod +x "$out_file"
}

# ── Launch loop ─────────────────────────────────────────────────────────────
echo "=== vast.ai NAV4RAIL Benchmark Launcher ==="
echo "Models:    ${MODELS[*]}"
echo "Method:    ${METHOD}"
echo "Image:     ${IMAGE}"
echo "Disk:      ${DISK} GB"
echo "Extra:     ${EXTRA_QUERY}"
echo "Log:       ${INSTANCES_LOG}"
echo ""

USED_OFFERS=""
LAUNCHED=()
FAILED=()

for model in "${MODELS[@]}"; do
    gpu_ram=$(get_min_gpu_mem "$model")
    echo "--- ${model} (needs >= ${gpu_ram} GB VRAM) ---"

    offer_id=$(find_offer "$gpu_ram" "$USED_OFFERS")
    if [ -z "$offer_id" ]; then
        echo "  No suitable offer found. Skipping."
        FAILED+=("$model:no_offer")
        continue
    fi
    echo "  Selected offer: ${offer_id}"
    USED_OFFERS="${USED_OFFERS} ${offer_id}"

    onstart_file="${ONSTART_TMPDIR}/onstart_${model}.sh"
    write_onstart "$model" "$METHOD" "$PROMPT_MODE" "$onstart_file"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY_RUN] Would launch on offer ${offer_id} with onstart ${onstart_file}"
        continue
    fi

    env_str="-e HF_TOKEN=${HF_TOKEN} -e WANDB_API_KEY=${WANDB_API_KEY:-} -e WANDB_PROJECT=nav4rail-bench"

    create_json=$(vastai create instance "$offer_id" \
        --image "$IMAGE" \
        --disk "$DISK" \
        --env "$env_str" \
        --onstart "$onstart_file" \
        --raw 2>&1) || {
            echo "  FAILED: vastai create returned non-zero"
            echo "  Output: $create_json"
            FAILED+=("$model:create_error")
            continue
        }

    instance_id=$(echo "$create_json" | python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    print(data.get('new_contract') or data.get('id') or '')
except Exception as e:
    sys.stderr.write(f'parse error: {e}\n')
" 2>/dev/null)

    if [ -z "$instance_id" ]; then
        echo "  FAILED: could not extract instance ID from response"
        echo "  Response: $create_json"
        FAILED+=("$model:parse_error")
        continue
    fi

    echo "  Instance launched: ${instance_id}"
    LAUNCHED+=("${instance_id}:${model}")
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$(date -Iseconds)" "$instance_id" "$model" "$METHOD" "$offer_id" \
        >> "$INSTANCES_LOG"
done

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Summary ==="
echo "Launched:  ${#LAUNCHED[@]}"
for entry in "${LAUNCHED[@]}"; do
    IFS=: read -r id m <<< "$entry"
    printf "  %-20s instance %s\n" "$m" "$id"
done
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Failed:    ${#FAILED[@]}"
    for entry in "${FAILED[@]}"; do
        IFS=: read -r m reason <<< "$entry"
        printf "  %-20s %s\n" "$m" "$reason"
    done
fi

if [ "${#LAUNCHED[@]}" -gt 0 ]; then
    cat <<EOF

Monitor:   vastai show instances
Logs:      vastai logs <instance_id>
Download:  vastai scp <instance_id>:/workspace/results_<model>.tar.gz ./runs/vastai/
Destroy:   vastai destroy instance <instance_id>
All at once:
  awk -F'\t' 'NR>FNR{next} {print \$2}' ${INSTANCES_LOG} | xargs -n1 vastai destroy instance
EOF
fi
