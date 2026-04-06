#!/usr/bin/env bash
# Shared environment for Telecom Paris GPU jobs (see FineTuningOnTelecomCluster/README.md).
# Source from job scripts after resolving SCRIPT_DIR (Slurm copies the .slurm file to spool — use
# SLURM_SUBMIT_DIR there; see sft_lora.slurm). Local: source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_telecom_env.sh"
#
# Optional env:
#   BENCHMARK_ROOT_OVERRIDE  — if set, cd here instead of auto-detected benchmark root (e.g. ~/benchmark)
#   BENCHMARK_VENV           — venv path (default: $BENCHMARK_ROOT/.venv)
#   BENCHMARK_GPU_PROFILE    — force: p100 | auto (default auto: detect from nvidia-smi / SLURM_JOB_PARTITION)
#   TORCH_CUDA_ARCH_LIST     — only for building PyTorch from source; e.g. 6.0 for Pascal (P100)

set -euo pipefail

_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export BENCHMARK_ROOT="$(cd "$_SLURM_DIR/.." && pwd)"

if [[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi

if [[ -n "${BENCHMARK_ROOT_OVERRIDE:-}" ]]; then
  export BENCHMARK_ROOT="$(cd "${BENCHMARK_ROOT_OVERRIDE}" && pwd)"
fi

cd "$BENCHMARK_ROOT"

echo "[job] BENCHMARK_ROOT=$BENCHMARK_ROOT"
echo "[job] host=$(hostname) jobid=${SLURM_JOB_ID:-unknown} date=$(date -Iseconds)"

# --- Slurm / run metadata (also persisted in run_manifest.json by Python jobs) ---
echo "[job] manifest_kv SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}"
echo "[job] manifest_kv SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "[job] manifest_kv SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo "[job] manifest_kv SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
echo "[job] manifest_kv SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"

# Cluster modules (Telecom Paris)
if command -v module >/dev/null 2>&1; then
  module load python/3.11.13 cuda/12.4.1 2>/dev/null \
    || module load python/3.11.13 2>/dev/null \
    || true
fi

# GPU profile: P100 (Pascal) prefers fp16 and no 4-bit; wheels still include sm_60 kernels.
export BENCHMARK_GPU_PROFILE="${BENCHMARK_GPU_PROFILE:-auto}"
_GPU_LINE=""
if command -v nvidia-smi >/dev/null 2>&1; then
  _GPU_LINE="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
fi
if [[ "$BENCHMARK_GPU_PROFILE" == "auto" ]]; then
  if [[ "${SLURM_JOB_PARTITION:-}" =~ [Pp]100 ]] || [[ "${_GPU_LINE:-}" == *"P100"* ]]; then
    BENCHMARK_GPU_PROFILE="p100"
  else
    BENCHMARK_GPU_PROFILE="generic"
  fi
fi
export BENCHMARK_GPU_PROFILE
if [[ "$BENCHMARK_GPU_PROFILE" == "p100" ]]; then
  export BENCHMARK_FP16="${BENCHMARK_FP16:-1}"
  export BENCHMARK_DISABLE_4BIT="${BENCHMARK_DISABLE_4BIT:-1}"
  echo "[job] BENCHMARK_GPU_PROFILE=p100 BENCHMARK_FP16=1 BENCHMARK_DISABLE_4BIT=1"
  echo "[job] note: TORCH_CUDA_ARCH_LIST=6.0 only if compiling PyTorch from source (optional)."
else
  echo "[job] BENCHMARK_GPU_PROFILE=$BENCHMARK_GPU_PROFILE"
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
_venv_py=""
if [[ -x "$VENV_DIR/bin/python3" ]]; then
  _venv_py="$VENV_DIR/bin/python3"
elif [[ -x "$VENV_DIR/bin/python" ]]; then
  _venv_py="$VENV_DIR/bin/python"
fi
# Quota / failed pip rollbacks can leave a `.venv` directory without bin/python3 — do not skip creation.
if [[ -z "$_venv_py" ]]; then
  if [[ -d "$VENV_DIR" ]]; then
    echo "[job] WARN: venv at $VENV_DIR has no usable python; removing and recreating"
    rm -rf "$VENV_DIR"
  fi
  echo "[job] creating venv: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
  if [[ -x "$VENV_DIR/bin/python3" ]]; then
    _venv_py="$VENV_DIR/bin/python3"
  elif [[ -x "$VENV_DIR/bin/python" ]]; then
    _venv_py="$VENV_DIR/bin/python"
  else
    echo "[job] ERROR: venv creation failed (no python in $VENV_DIR/bin)"
    exit 1
  fi
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
# Bash caches the pre-venv `python3` path; without this, `python3`/`pip` can still hit the
# system interpreter → "Defaulting to user installation" and a venv that never gets packages.
hash -r 2>/dev/null || true

# Unbuffered Python so Slurm .out/.err show logs promptly (training can be long).
export PYTHONUNBUFFERED=1

# Always use the venv interpreter for installs and import checks (do not rely on PATH alone).
export BENCHMARK_PYTHON="$_venv_py"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$BENCHMARK_ROOT/runs/slurm"

echo "[job] python: $BENCHMARK_PYTHON $($BENCHMARK_PYTHON --version 2>&1)"

# On P100 clusters, unpinned `torch` on PyPI often resolves to a huge CUDA 13 stack; prefer cu121 wheels.
REQ_INSTALL="$REQ"
if [[ "$BENCHMARK_GPU_PROFILE" == "p100" && -f "$BENCHMARK_ROOT/requirements-p100-cluster.txt" ]]; then
  REQ_INSTALL="$BENCHMARK_ROOT/requirements-p100-cluster.txt"
  echo "[job] using P100 requirements: $REQ_INSTALL"
fi

"$BENCHMARK_PYTHON" -m pip install --upgrade pip wheel setuptools
"$BENCHMARK_PYTHON" -m pip install -r "$REQ_INSTALL"

# PyYAML: `import yaml` — package name on pip is `pyyaml` (not `yaml`)
if ! "$BENCHMARK_PYTHON" -c "import yaml" 2>/dev/null; then
  echo "[job] installing PyYAML (import yaml)"
  "$BENCHMARK_PYTHON" -m pip install "pyyaml>=6"
fi

export PYTHONPATH="$BENCHMARK_ROOT"

# Minimal import checks before training / eval (fail fast).
if ! "$BENCHMARK_PYTHON" -c "import yaml, torch, transformers, trl; import src.config_loader" 2>/dev/null; then
  echo "[job] ERROR: core imports failed after pip install. Retry: pip install -r requirements.txt"
  "$BENCHMARK_PYTHON" -c "import yaml, torch, transformers, trl; import src.config_loader" || true
  exit 1
fi

# TRL DPOTrainer may import `weave` (wandb). Don't fail globally; DPO jobs will error clearly if missing.
if ! "$BENCHMARK_PYTHON" -c "import weave" 2>/dev/null; then
  echo "[job] WARN: `import weave` failed. If you run DPO with TRL and it errors, ensure `weave` installs in requirements."
fi

"$BENCHMARK_PYTHON" << 'PY'
import os
import torch
print("[job] manifest_kv torch_version=" + torch.__version__)
print("[job] manifest_kv cuda_version_torch=" + (torch.version.cuda or ""))
if torch.cuda.is_available():
    print("[job] manifest_kv cuda_device_name=" + torch.cuda.get_device_name(0))
else:
    print("[job] manifest_kv cuda_device_name=")
PY

echo "[job] prerequisites OK"

# Optional rsync of runs/ at job end (only under Slurm; set BENCHMARK_SYNC_BACK_DEST to enable).
if [[ -n "${SLURM_JOB_ID:-}" && -f "$_SLURM_DIR/_telecom_sync_back.sh" ]]; then
  # shellcheck source=/dev/null
  source "$_SLURM_DIR/_telecom_sync_back.sh"
fi
