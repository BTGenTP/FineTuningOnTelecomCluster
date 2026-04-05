#!/usr/bin/env bash
# Optional: rsync runs/ from the cluster job to another destination when the batch script exits.
# Source this from _telecom_env.sh (registers an EXIT trap).
#
# Enable before sbatch (same shell export, or #SBATCH cannot set arbitrary remote hosts easily):
#   export BENCHMARK_SYNC_BACK_DEST='user@laptop:/path/to/benchmark/'
#   # or a second NFS path on the cluster (no SSH):
#   export BENCHMARK_SYNC_BACK_DEST='/shared/proj/backups/latoundji/benchmark/'
#
# If unset or empty, nothing runs.
#
# Notes:
# - SSH from a compute node to your laptop often fails (NAT/firewall). Prefer an NFS path,
#   a bastion you control, or run ./scripts/cluster_sync_pull_runs.sh from the login node.
# - Preserves the Slurm job exit code.

_benchmark_finalize_exit() {
  local _status=$?
  if [[ -n "${BENCHMARK_SYNC_BACK_DEST:-}" ]]; then
    if ! command -v rsync >/dev/null 2>&1; then
      echo "[job] BENCHMARK_SYNC_BACK_DEST set but rsync not found; skip sync-back" >&2
    else
      echo "[job] sync-back runs/ → ${BENCHMARK_SYNC_BACK_DEST}runs/"
      mkdir -p "${BENCHMARK_ROOT}/runs" 2>/dev/null || true
      # shellcheck disable=SC2086
      if rsync -a --partial "${BENCHMARK_ROOT}/runs/" "${BENCHMARK_SYNC_BACK_DEST%/}/runs/" 2>&1; then
        echo "[job] sync-back OK"
      else
        echo "[job] WARN: sync-back failed (exit ${_status} still reported for the job)" >&2
      fi
    fi
  fi
  exit "$_status"
}

trap '_benchmark_finalize_exit' EXIT
