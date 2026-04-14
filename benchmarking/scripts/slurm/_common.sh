#!/bin/bash
# ============================================================================
# _common.sh — Shared SLURM environment setup for all NAV4RAIL jobs
# ============================================================================
# Source this file from batch scripts (do not use dirname(BASH_SOURCE) alone: Slurm runs a
# copy under /var/spool/slurmd/.../slurm_script, so BASH_SOURCE would miss this file):
#   if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -r "${SLURM_SUBMIT_DIR}/scripts/slurm/_common.sh" ]; then
#     source "${SLURM_SUBMIT_DIR}/scripts/slurm/_common.sh"
#   else
#     source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
#   fi
#
# What it does:
#   1. cd to benchmarking/ root (reliable cwd regardless of sbatch location)
#   2. Load CUDA module (for GPU access)
#   3. Create venv if missing, activate it
#   4. Clear PYTHONHOME (module load sets it, breaks venvs)
#   5. Ensure all requirements.txt deps are installed (idempotent, flock-serialized)
#   6. Set PYTHONPATH so `python -m src.*` works
#   7. Print diagnostic info to stdout (captured by SLURM .out)
# ============================================================================

set -euo pipefail

# ── Resolve benchmarking/ root ──────────────────────────────────────────────
# Works whether this file is sourced from scripts/slurm/ or anywhere else.
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BENCH_DIR"

VENV_DIR="${VENV_DIR:-$HOME/.venvs/nav4rail_bench}"
# Partition-scoped venv: P100 and 3090 need different package builds (torch, bitsandbytes).
if [ -n "${SLURM_JOB_PARTITION:-}" ]; then
    VENV_DIR="${VENV_DIR}_${SLURM_JOB_PARTITION}"
fi

# ── Load modules ────────────────────────────────────────────────────────────
# CUDA is needed for GPU. We load python module ONLY for venv creation (the
# compute node's system Python may be too old), then let the venv's own Python
# take over.
module load cuda/12.4.1 2>/dev/null || true

# ── Create venv if it doesn't exist ────────────────────────────────────────
# Check for bin/activate (not just the directory): python3 -m venv creates
# the directory BEFORE populating it, so concurrent tasks that see the dir
# but source a not-yet-created activate script get "No such file or directory".
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    (
        flock -x 200
        if [ ! -f "$VENV_DIR/bin/activate" ]; then
            echo "[_common.sh] Creating venv at $VENV_DIR ..."
            # Load python module for venv creation (need >= 3.10)
            module load python/3.11.13 2>/dev/null || true
            python3 -m venv "$VENV_DIR"
        fi
    ) 200>>"${VENV_DIR}.lock"
fi

# ── Activate venv ───────────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"

# ── CRITICAL FIX: Clear PYTHONHOME ──────────────────────────────────────────
# `module load python/X.Y.Z` often sets PYTHONHOME, which overrides the venv's
# sys.path discovery. This makes Python ignore the venv's site-packages and
# look for packages under the module's prefix instead.
# Result: "ModuleNotFoundError: No module named 'yaml'" even though pyyaml is
# installed in the venv.
# See: https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHOME
unset PYTHONHOME 2>/dev/null || true

# ── Hugging Face token (needed for gated repos like google/gemma-2-9b-it) ─
if [ -z "${HF_TOKEN:-}" ]; then
    for _hf_token_file in "$HOME/.cache/huggingface/token" "$HOME/.huggingface/token"; do
        if [ -f "$_hf_token_file" ]; then
            export HF_TOKEN="$(cat "$_hf_token_file")"
            break
        fi
    done
    unset _hf_token_file
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[_common.sh] WARNING: HF_TOKEN not set — gated models (e.g. gemma) will fail."
    echo "  Fix: huggingface-cli login   (run once on the cluster login node)"
fi

# ── Ensure all dependencies are installed ───────────────────────────────────
# Serialize pip against VENV_DIR.lock: a shared venv on NFS + concurrent array
# tasks causes errno 116 (Stale file handle) and half-installed envs → import yaml fails.
# This is idempotent: pip skips satisfied deps; queued jobs wait on flock.
(
    flock -x 200
    # Avoid "invalid command 'bdist_wheel'" when building sdists from cache.
    python -m pip install --quiet --disable-pip-version-check wheel setuptools
    # ── GPU-aware PyTorch: P100 (sm_60) needs torch < 2.5 ──────────────────
    # Install the right version BEFORE requirements.txt so pip sees torch as
    # already satisfied and does not upgrade to an incompatible build.
    _gpu_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$_gpu_cc" ]; then
        _cc_int=$(echo "$_gpu_cc" | tr -d '.')
        if [ "$_cc_int" -lt 75 ]; then
            echo "[_common.sh] GPU CC ${_gpu_cc} < 7.5 — pinning torch 2.4.x for sm_60 support"
            pip install --quiet --disable-pip-version-check "torch>=2.4.0,<2.5.0"
        fi
    fi
    pip install --quiet --disable-pip-version-check -r "$BENCH_DIR/requirements.txt" 2>&1 || {
        echo "[_common.sh] WARNING: pip install failed, retrying with --no-build-isolation..."
        pip install --quiet --disable-pip-version-check --no-build-isolation -r "$BENCH_DIR/requirements.txt"
    }
    # Cheap sanity check: repair if a previous crashed install left yaml missing.
    python -c "import yaml" 2>/dev/null || python -m pip install --quiet --disable-pip-version-check "pyyaml>=6.0"
) 200>>"$VENV_DIR.lock"

# ── PYTHONPATH ──────────────────────────────────────────────────────────────
export PYTHONPATH="${PYTHONPATH:-}:${BENCH_DIR}"

# ── Diagnostic output (captured in SLURM .out) ─────────────────────────────
echo "=== NAV4RAIL Environment ==="
echo "  Date:       $(date)"
echo "  Hostname:   $(hostname)"
echo "  Working dir: $(pwd)"
echo "  Python:     $(which python) ($(python --version 2>&1))"
echo "  VENV_DIR:   $VENV_DIR"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  PYTHONHOME: ${PYTHONHOME:-<unset>}"
echo "  GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  yaml check: $(python -c 'import yaml; print(yaml.__version__)' 2>&1)"
echo "============================="
