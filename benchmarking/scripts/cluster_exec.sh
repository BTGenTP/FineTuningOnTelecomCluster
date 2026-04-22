#!/bin/bash
# ============================================================================
# cluster_exec.sh — Execute SLURM commands on the remote cluster
# ============================================================================
# Convenience wrapper to submit jobs, check status, and read logs from local.
#
# Usage:
#   ./scripts/cluster_exec.sh submit train.sh                     # sbatch
#   ./scripts/cluster_exec.sh submit array_models.sh METHOD=sft   # with env vars
#   ./scripts/cluster_exec.sh status                              # squeue --me
#   ./scripts/cluster_exec.sh status 772878                       # sacct for job
#   ./scripts/cluster_exec.sh logs 772878                         # cat stdout
#   ./scripts/cluster_exec.sh logs 772878 err                     # cat stderr
#   ./scripts/cluster_exec.sh cancel 772878                       # scancel
#   ./scripts/cluster_exec.sh shell                               # interactive SSH
#   ./scripts/cluster_exec.sh cmd "ls -la runs/"                  # arbitrary command
#
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

# ── Functions ───────────────────────────────────────────────────────────────
usage() {
    cat <<'USAGE'
Usage: cluster_exec.sh <command> [args...]

Commands:
  submit <script> [VAR=val ...]   Submit a SLURM job (sbatch)
  status [job_id]                 Show job queue (squeue) or job details (sacct)
  logs <job_id> [out|err]         Read SLURM output/error logs
  cancel <job_id>                 Cancel a SLURM job (scancel)
  shell                           Open interactive SSH session
  cmd "<command>"                 Execute arbitrary command on cluster
  vastai [args...]                Launch on vast.ai (delegates to scripts/vastai_run.sh)

Examples:
  cluster_exec.sh submit train.sh METHOD=grpo MODEL=llama3_8b
  cluster_exec.sh submit array_models.sh METHOD=zero_shot
  cluster_exec.sh status
  cluster_exec.sh status 772878
  cluster_exec.sh logs 772878
  cluster_exec.sh cancel 772878
  cluster_exec.sh cmd "nvidia-smi"
  cluster_exec.sh vastai --model gemma2_9b --method zero_shot
  cluster_exec.sh vastai --dry-run
USAGE
    exit 0
}

do_submit() {
    local script="$1"
    shift

    # Build env export string from remaining args (VAR=val)
    local env_exports=""
    for var in "$@"; do
        env_exports+="export ${var}; "
    done

    # Resolve script path: look in scripts/slurm/ if not an absolute path
    local remote_script
    if [[ "$script" == /* ]]; then
        remote_script="$script"
    else
        remote_script="${REMOTE_PATH}/scripts/slurm/${script}"
    fi

    # ── Sync .env to cluster so _common.sh can source it ────────────────────
    # Paths: this script is in scripts/, so benchmarking/ is one level up.
    local bench_dir
    bench_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    if [ -r "${bench_dir}/.env" ]; then
        echo "Syncing .env to ${REMOTE_HOST}:${REMOTE_PATH}/.env"
        # --chmod=600 keeps the token private on the shared cluster FS.
        rsync -az --chmod=F600 "${bench_dir}/.env" "${REMOTE_HOST}:${REMOTE_PATH}/.env"
    else
        echo "Note: no ${bench_dir}/.env found — relying on cluster-side credentials."
    fi

    echo "Submitting: ${remote_script}"
    [ -n "$env_exports" ] && echo "Env: $env_exports"

    ssh "${REMOTE_HOST}" "cd ${REMOTE_PATH} && ${env_exports}sbatch ${remote_script}"
}

do_status() {
    local job_id="${1:-}"

    if [ -z "$job_id" ]; then
        echo "=== Active jobs (squeue --me) ==="
        ssh "${REMOTE_HOST}" "squeue --me --format='%.10i %.12P %.20j %.8u %.2t %.10M %.6D %R'"
        echo ""
        echo "=== Recent jobs (sacct, last 24h) ==="
        ssh "${REMOTE_HOST}" "sacct --starttime=now-1day --format=JobID,JobName%20,Partition,State,ExitCode,Elapsed,MaxRSS,NodeList --noheader 2>/dev/null | head -30"
    else
        echo "=== Job ${job_id} details ==="
        ssh "${REMOTE_HOST}" "sacct -j ${job_id} --format=JobID,JobName%20,Partition,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,NodeList"
    fi
}

do_logs() {
    local job_id="$1"
    local stream="${2:-out}"  # out or err

    echo "=== Searching for logs of job ${job_id} ==="

    # Find log files matching the job ID pattern
    local ext
    if [ "$stream" = "err" ]; then
        ext="err"
    else
        ext="out"
    fi

    ssh "${REMOTE_HOST}" "
        cd ${REMOTE_PATH}
        # Search for SLURM log files matching this job ID
        logs=\$(find runs/slurm/ -name \"*${job_id}*.${ext}\" 2>/dev/null | sort)
        if [ -z \"\$logs\" ]; then
            echo 'No log files found for job ${job_id} (*.${ext})'
            echo 'Available log dirs:'
            ls -d runs/slurm/nav4rail_*${job_id}* 2>/dev/null || echo '  (none)'
        else
            for f in \$logs; do
                echo \"--- \$f ---\"
                cat \"\$f\"
                echo ''
            done
        fi
    "
}

do_cancel() {
    local job_id="$1"
    echo "Cancelling job ${job_id}..."
    ssh "${REMOTE_HOST}" "scancel ${job_id}"
    echo "Done."
}

do_shell() {
    echo "Opening SSH session to ${REMOTE_HOST}..."
    echo "(Working directory: ${REMOTE_PATH})"
    ssh -t "${REMOTE_HOST}" "cd ${REMOTE_PATH} && exec \$SHELL -l"
}

do_cmd() {
    local cmd="$1"
    ssh "${REMOTE_HOST}" "cd ${REMOTE_PATH} && ${cmd}"
}

do_vastai() {
    # Resolve this script's directory to find vastai_run.sh
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local vastai_script="${script_dir}/vastai_run.sh"

    if [ ! -x "$vastai_script" ]; then
        echo "Error: ${vastai_script} not found or not executable"
        exit 1
    fi

    exec "$vastai_script" "$@"
}

# ── Main ────────────────────────────────────────────────────────────────────
if [ $# -lt 1 ]; then
    usage
fi

COMMAND="$1"
shift

case "$COMMAND" in
    submit)
        [ $# -lt 1 ] && { echo "Error: submit requires a script name"; exit 1; }
        do_submit "$@"
        ;;
    status)
        do_status "${1:-}"
        ;;
    logs)
        [ $# -lt 1 ] && { echo "Error: logs requires a job ID"; exit 1; }
        do_logs "$@"
        ;;
    cancel)
        [ $# -lt 1 ] && { echo "Error: cancel requires a job ID"; exit 1; }
        do_cancel "$1"
        ;;
    shell)
        do_shell
        ;;
    cmd)
        [ $# -lt 1 ] && { echo "Error: cmd requires a command string"; exit 1; }
        do_cmd "$*"
        ;;
    vastai)
        do_vastai "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown command: $COMMAND"
        usage
        ;;
esac
