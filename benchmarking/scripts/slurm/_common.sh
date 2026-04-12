#!/bin/bash
# ============================================================================
# _common.sh — Shared SLURM environment setup for all NAV4RAIL jobs
# ============================================================================
# Source this file at the top of every SLURM script:
#   source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
#
# What it does:
#   1. cd to benchmarking/ root (reliable cwd regardless of sbatch location)
#   2. Load CUDA module (for GPU access)
#   3. Create venv if missing, activate it
#   4. Clear PYTHONHOME (module load sets it, breaks venvs)
#   5. Ensure all requirements.txt deps are installed (idempotent)
#   6. Set PYTHONPATH so `python -m src.*` works
#   7. Print diagnostic info to stdout (captured by SLURM .out)
# ============================================================================

set -euo pipefail

# ── Resolve benchmarking/ root ──────────────────────────────────────────────
# Works whether this file is sourced from scripts/slurm/ or anywhere else.
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BENCH_DIR"

VENV_DIR="${VENV_DIR:-$HOME/.venvs/nav4rail_bench}"

# ── Load modules ────────────────────────────────────────────────────────────
# CUDA is needed for GPU. We load python module ONLY for venv creation (the
# compute node's system Python may be too old), then let the venv's own Python
# take over.
module load cuda/12.4.1 2>/dev/null || true

# ── Create venv if it doesn't exist ────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[_common.sh] Creating venv at $VENV_DIR ..."
    (
        flock -x 200
        if [ ! -d "$VENV_DIR" ]; then
            # Load python module for venv creation (need >= 3.10)
            module load python/3.11.13 2>/dev/null || true
            python3 -m venv "$VENV_DIR"
        fi
    ) 200>"$VENV_DIR.lock"
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

# ── Ensure all dependencies are installed ───────────────────────────────────
# This is idempotent: pip checks versions and skips already-satisfied deps.
# Takes < 2s when everything is already installed.
# Fixes the case where the venv was created but requirements weren't installed,
# or where a new dependency was added to requirements.txt.
pip install --quiet --disable-pip-version-check -r "$BENCH_DIR/requirements.txt" 2>&1 || {
    echo "[_common.sh] WARNING: pip install failed, retrying with --no-build-isolation..."
    pip install --quiet --disable-pip-version-check --no-build-isolation -r "$BENCH_DIR/requirements.txt"
}

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
