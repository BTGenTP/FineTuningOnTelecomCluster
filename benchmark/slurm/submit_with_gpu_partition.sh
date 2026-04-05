#!/usr/bin/env bash
# Submit a benchmark Slurm job, optionally overriding GPU partition.
#
# Usage (from login node):
#   ./slurm/submit_with_gpu_partition.sh slurm/sft_lora.slurm
#
# Env:
#   GPU_PARTITION_ORDER — space-separated list tried in order (default: 3090 H100 P100)
#   First partition that exists in `sinfo` is used via: sbatch --partition=<p> <jobfile>
#
# If no partition matches, falls back to plain: sbatch <jobfile> (uses #SBATCH in file).
#
# Note: Slurm cannot switch partition at runtime; this wrapper only picks one at submit time.

set -euo pipefail

JOBFILE="${1:?usage: $0 path/to/job.slurm}"
if [[ ! -f "$JOBFILE" ]]; then
  echo "ERROR: not found: $JOBFILE"
  exit 1
fi

ORDER="${GPU_PARTITION_ORDER:-3090 H100 P100}"
for p in $ORDER; do
  if sinfo -p "$p" -h 2>/dev/null | grep -q .; then
    echo "[submit] sbatch --partition=$p $JOBFILE"
    exec sbatch --partition="$p" "$JOBFILE"
  fi
done

echo "[submit] no partition in [$ORDER] seen in sinfo; submitting without --partition (job defaults)"
exec sbatch "$JOBFILE"
