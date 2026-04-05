#!/usr/bin/env bash
# Shared environment for Telecom Paris GPU jobs (see FineTuningOnTelecomCluster/README.md).
# Source from job scripts after resolving SCRIPT_DIR (Slurm copies the .slurm file to spool — use
# SLURM_SUBMIT_DIR there; see sft_lora.slurm). Local: source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_telecom_env.sh"
#
# Optional env:
#   BENCHMARK_ROOT_OVERRIDE  — if set, cd here instead of auto-detected benchmark root (e.g. ~/benchmark)
#   BENCHMARK_VENV           — venv path (default: $BENCHMARK_ROOT/.venv)

set -euo pipefail

_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export BENCHMARK_ROOT="$(cd "$_SLURM_DIR/.." && pwd)"

if [[ -n "${BENCHMARK_ROOT_OVERRIDE:-}" ]]; then
  export BENCHMARK_ROOT="$(cd "${BENCHMARK_ROOT_OVERRIDE}" && pwd)"
fi

cd "$BENCHMARK_ROOT"

echo "[job] BENCHMARK_ROOT=$BENCHMARK_ROOT"
echo "[job] host=$(hostname) jobid=${SLURM_JOB_ID:-unknown} date=$(date -Iseconds)"

# Cluster modules (Telecom Paris)
if command -v module >/dev/null 2>&1; then
  module load python/3.11.13 cuda/12.4.1 2>/dev/null \
    || module load python/3.11.13 2>/dev/null \
    || true
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[job] GPU:"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || nvidia-smi || true
fi

REQ="$BENCHMARK_ROOT/requirements.txt"
if [[ ! -f "$REQ" ]]; then
  echo "[job] ERROR: missing requirements: $REQ"
  exit 1
fi

VENV_DIR="${BENCHMARK_VENV:-$BENCHMARK_ROOT/.venv}"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[job] creating venv: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$BENCHMARK_ROOT/runs/slurm"

echo "[job] python: $(command -v python3) $(python3 --version 2>&1)"

python3 -m pip install --upgrade pip wheel setuptools
python3 -m pip install -r "$REQ"

# PyYAML: `import yaml` — package name on pip is `pyyaml` (not `yaml`)
if ! python3 -c "import yaml" 2>/dev/null; then
  echo "[job] installing PyYAML (import yaml)"
  python3 -m pip install "pyyaml>=6"
fi

export PYTHONPATH="$BENCHMARK_ROOT"

# Minimal import checks before training / eval
if ! python3 -c "import yaml, torch, transformers, trl; import src.config_loader" 2>/dev/null; then
  echo "[job] ERROR: core imports failed after pip install. Retry: pip install -r requirements.txt"
  python3 -c "import yaml, torch, transformers, trl; import src.config_loader" || true
  exit 1
fi

echo "[job] prerequisites OK"
