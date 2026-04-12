#!/bin/bash
# ============================================================================
# cluster_sync_push.sh — Push benchmark sources from local machine to cluster
# ============================================================================
# Syncs the benchmarking/ directory to the remote cluster via rsync over SSH.
# Skips .venv, caches, __pycache__, runs/ and artifacts/ by default.
#
# Usage:
#   ./scripts/cluster_sync_push.sh                  # default: sync sources only
#   ./scripts/cluster_sync_push.sh --with-runs      # also sync runs/
#   ./scripts/cluster_sync_push.sh --with-data       # also sync large data files
#   ./scripts/cluster_sync_push.sh --dry-run         # preview without copying
#
# Requirements: rsync, SSH access to cluster
# SSH config expected:
#   Host gpu
#       Hostname gpu-gw.enst.fr
#       User latoundji-25
#       IdentityFile ~/.ssh/id_rsa
# ============================================================================

set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────────
REMOTE_HOST="${REMOTE_HOST:-gpu}"
REMOTE_PATH="${REMOTE_PATH:-~/benchmarking}"
LOCAL_PATH="$(cd "$(dirname "$0")/.." && pwd)/"  # benchmarking/ root

# ── Parse arguments ─────────────────────────────────────────────────────────
WITH_RUNS=false
WITH_DATA=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --with-runs)  WITH_RUNS=true ;;
        --with-data)  WITH_DATA=true ;;
        --dry-run)    DRY_RUN=true ;;
        -h|--help)
            echo "Usage: $0 [--with-runs] [--with-data] [--dry-run]"
            echo ""
            echo "Push benchmarking sources to cluster: ${REMOTE_HOST}:${REMOTE_PATH}"
            echo ""
            echo "Options:"
            echo "  --with-runs   Include runs/ directory (experiment outputs)"
            echo "  --with-data   Include large data files (datasets, etc.)"
            echo "  --dry-run     Preview what would be transferred"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

# ── Build exclude list ──────────────────────────────────────────────────────
EXCLUDES=(
    --exclude='.venv/'
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='.pytest_cache/'
    --exclude='*.egg-info/'
    --exclude='.git/'
    --exclude='wandb/'
    --exclude='*.swp'
    --exclude='*.swo'
    --exclude='.DS_Store'
)

if [ "$WITH_RUNS" = false ]; then
    EXCLUDES+=(--exclude='runs/')
fi

if [ "$WITH_DATA" = false ]; then
    EXCLUDES+=(--exclude='artifacts/')
fi

# ── Build rsync command ─────────────────────────────────────────────────────
RSYNC_OPTS=(
    -avz
    --progress
    --delete
    "${EXCLUDES[@]}"
)

if [ "$DRY_RUN" = true ]; then
    RSYNC_OPTS+=(--dry-run)
fi

# ── Execute ─────────────────────────────────────────────────────────────────
echo "=== cluster_sync_push ==="
echo "Local:  ${LOCAL_PATH}"
echo "Remote: ${REMOTE_HOST}:${REMOTE_PATH}"
echo "Options: with-runs=${WITH_RUNS} with-data=${WITH_DATA} dry-run=${DRY_RUN}"
echo "========================="

# Ensure remote directory exists
ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_PATH}"

rsync "${RSYNC_OPTS[@]}" "${LOCAL_PATH}" "${REMOTE_HOST}:${REMOTE_PATH}/"

echo ""
echo "Sync complete."
if [ "$DRY_RUN" = true ]; then
    echo "(dry-run — no files were actually transferred)"
fi
