#!/bin/bash
# ============================================================================
# cluster_sync_pull.sh — Pull experiment results from cluster to local
# ============================================================================
# Pulls runs/ and/or artifacts/ from the remote cluster. Merges with local
# dirs (rsync -a). Creates local runs/ and artifacts/ if missing.
#
# Usage:
#   ./scripts/cluster_sync_pull.sh                   # pull runs/ only (default)
#   ./scripts/cluster_sync_pull.sh --only-runs        # explicit: runs/ only
#   ./scripts/cluster_sync_pull.sh --only-artifacts   # artifacts/ only
#   ./scripts/cluster_sync_pull.sh --all              # both runs/ and artifacts/
#   ./scripts/cluster_sync_pull.sh --dry-run          # preview without copying
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
LOCAL_PATH="$(cd "$(dirname "$0")/.." && pwd)"  # benchmarking/ root

# ── Parse arguments ─────────────────────────────────────────────────────────
PULL_RUNS=true
PULL_ARTIFACTS=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --only-runs)      PULL_RUNS=true;  PULL_ARTIFACTS=false ;;
        --only-artifacts) PULL_RUNS=false; PULL_ARTIFACTS=true ;;
        --all)            PULL_RUNS=true;  PULL_ARTIFACTS=true ;;
        --dry-run)        DRY_RUN=true ;;
        -h|--help)
            echo "Usage: $0 [--only-runs|--only-artifacts|--all] [--dry-run]"
            echo ""
            echo "Pull experiment results from cluster: ${REMOTE_HOST}:${REMOTE_PATH}"
            echo ""
            echo "Options:"
            echo "  --only-runs       Pull runs/ directory only (default)"
            echo "  --only-artifacts  Pull artifacts/ directory only"
            echo "  --all             Pull both runs/ and artifacts/"
            echo "  --dry-run         Preview what would be transferred"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

# ── Build rsync base options ────────────────────────────────────────────────
RSYNC_OPTS=(
    -avz
    --progress
)

if [ "$DRY_RUN" = true ]; then
    RSYNC_OPTS+=(--dry-run)
fi

# ── Pull functions ──────────────────────────────────────────────────────────
pull_dir() {
    local dir="$1"
    local local_dir="${LOCAL_PATH}/${dir}"
    local remote_dir="${REMOTE_HOST}:${REMOTE_PATH}/${dir}"

    mkdir -p "${local_dir}"

    echo "--- Pulling ${dir}/ ---"
    echo "  From: ${remote_dir}"
    echo "  To:   ${local_dir}"

    rsync "${RSYNC_OPTS[@]}" "${remote_dir}/" "${local_dir}/" 2>/dev/null || {
        echo "  Warning: Remote directory ${dir}/ does not exist or is empty."
    }
}

# ── Execute ─────────────────────────────────────────────────────────────────
echo "=== cluster_sync_pull ==="
echo "Remote: ${REMOTE_HOST}:${REMOTE_PATH}"
echo "Local:  ${LOCAL_PATH}"
echo "Pull: runs=${PULL_RUNS} artifacts=${PULL_ARTIFACTS} dry-run=${DRY_RUN}"
echo "========================="

if [ "$PULL_RUNS" = true ]; then
    pull_dir "runs"
fi

if [ "$PULL_ARTIFACTS" = true ]; then
    pull_dir "artifacts"
fi

echo ""
echo "Pull complete."
if [ "$DRY_RUN" = true ]; then
    echo "(dry-run — no files were actually transferred)"
fi
