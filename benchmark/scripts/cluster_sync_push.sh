#!/usr/bin/env bash
# Push benchmark sources from this machine → cluster (rsync over SSH).
#
# Usage:
#   export BENCHMARK_CLUSTER_SSH='latoundji-25@gpu-gw.example.fr'
#   export BENCHMARK_CLUSTER_PATH='~/benchmark'   # optional
#   ./scripts/cluster_sync_push.sh [--dry-run] [--delete] [--with-runs] [--with-git]
#
# Or pass SSH target as first arg:
#   ./scripts/cluster_sync_push.sh latoundji-25@gpu-gw --dry-run
#
# Requires: rsync and SSH access. Does not copy .venv, caches, or runs/ unless --with-runs.
# For a one-shot without rsync: tar + scp is possible but rsync is incremental and respects excludes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/cluster_sync_common.sh"

DRY_RUN=0
DELETE=0
WITH_RUNS=0
WITH_GIT=0
SSH_TARGET="${BENCHMARK_CLUSTER_SSH:-}"
REMOTE_PATH="${BENCHMARK_CLUSTER_PATH:-~/benchmark}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --delete) DELETE=1 ;;
    --with-runs) WITH_RUNS=1 ;;
    --with-git) WITH_GIT=1 ;;
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
  echo "rsync is required. Install it or use manual scp after packing excludes." >&2
  exit 1
fi

cluster_sync_read_excludes_to_array
_EXCLUDES=("${_CLUSTER_RSYNC_EXCLUDES[@]}")
if [[ "$WITH_RUNS" -eq 0 ]]; then
  _EXCLUDES+=(--exclude=runs/)
fi
if [[ "$WITH_GIT" -eq 1 ]]; then
  _filtered=()
  for _x in "${_EXCLUDES[@]}"; do
    [[ "$_x" == --exclude=.git/ ]] && continue
    _filtered+=("$_x")
  done
  _EXCLUDES=("${_filtered[@]}")
fi

RSYNC_FLAGS=(--archive --compress --human-readable --partial --stats)
if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC_FLAGS+=(--dry-run --verbose)
fi
if [[ "$DELETE" -eq 1 ]]; then
  RSYNC_FLAGS+=(--delete)
  # Excluded paths (e.g. runs/ when not using --with-runs) are not deleted on the receiver
  # by default; still use --dry-run first.
  echo "[warn] --delete: remote files absent locally are removed (see rsync --delete)." >&2
fi

echo "[push] local:  $BENCH_ROOT/"
echo "[push] remote: ${SSH_TARGET}:${REMOTE_PATH}/"

set -x
rsync "${RSYNC_FLAGS[@]}" "${_EXCLUDES[@]}" \
  "$BENCH_ROOT/" "${SSH_TARGET}:${REMOTE_PATH}/"
set +x

echo "[push] done."
