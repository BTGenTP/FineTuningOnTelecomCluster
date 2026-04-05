#!/usr/bin/env bash
# Pull runs/ (and optionally slurm logs under runs/slurm) from cluster → local benchmark tree.
#
# Usage:
#   export BENCHMARK_CLUSTER_SSH='latoundji-25@gpu-gw.example.fr'
#   export BENCHMARK_CLUSTER_PATH='/home/mlatoundji/studies/dev/nav4rails/repositories/FineTuningOnTelecomCluster/benchmark'
#   ./scripts/cluster_sync_pull_runs.sh [--dry-run]
#
# Or: ./scripts/cluster_sync_pull_runs.sh user@host [remote_path]
#
# Creates local runs/ if missing; merges with existing run subdirs (rsync -a).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DRY_RUN=0
SSH_TARGET="${BENCHMARK_CLUSTER_SSH:-}"
REMOTE_PATH="${BENCHMARK_CLUSTER_PATH:-~/benchmark}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      if [[ -z "$SSH_TARGET" ]]; then
        SSH_TARGET="$1"
      else
        REMOTE_PATH="$1"
      fi
      ;;
  esac
  shift
done

if [[ -z "$SSH_TARGET" ]]; then
  echo "Set BENCHMARK_CLUSTER_SSH or pass user@host as first argument." >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required." >&2
  exit 1
fi

mkdir -p "$BENCH_ROOT/runs"

RSYNC_FLAGS=(--archive --compress --human-readable --partial --stats)
if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC_FLAGS+=(--dry-run --verbose)
fi

echo "[pull] remote: ${SSH_TARGET}:${REMOTE_PATH}/runs/"
echo "[pull] local:  $BENCH_ROOT/runs/"

set -x
rsync "${RSYNC_FLAGS[@]}" \
  "${SSH_TARGET}:${REMOTE_PATH}/runs/" "$BENCH_ROOT/runs/"
set +x

echo "[pull] done."
