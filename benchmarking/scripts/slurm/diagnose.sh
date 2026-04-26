#!/bin/bash
# ============================================================================
# diagnose.sh — Run on a compute node to verify the environment
# ============================================================================
# Submit this as a quick SLURM job to check that Python, venv, and all
# dependencies are correctly set up on the actual compute nodes.
#
# Usage:
#   sbatch scripts/slurm/diagnose.sh
#   # Then: cat runs/slurm/nav4rail_diagnose_*/slurm_*.out

#SBATCH --job-name=nav4rail_diagnose
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=runs/slurm/nav4rail_diagnose_%j/slurm_%j.out
#SBATCH --error=runs/slurm/nav4rail_diagnose_%j/slurm_%j.err

echo "=========================================="
echo "  NAV4RAIL Environment Diagnostic"
echo "=========================================="
echo ""

echo "--- System ---"
echo "Hostname:    $(hostname)"
echo "Date:        $(date)"
echo "Kernel:      $(uname -r)"
echo "User:        $(whoami)"
echo "Home:        $HOME"
echo "PWD:         $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo ""

echo "--- Module system ---"
module avail python 2>&1 | head -20 || echo "  module command not available"
echo ""
echo "Loading python/3.11.13..."
module load python/3.11.13 2>&1 || echo "  FAILED to load python/3.11.13"
echo "  After module load:"
echo "    which python3: $(which python3 2>/dev/null || echo NOT_FOUND)"
echo "    python3 --version: $(python3 --version 2>&1 || echo FAILED)"
echo "    PYTHONHOME: ${PYTHONHOME:-<unset>}"
echo "    PYTHONPATH: ${PYTHONPATH:-<unset>}"
echo ""

echo "--- GPU ---"
nvidia-smi 2>/dev/null || echo "  nvidia-smi not available"
echo ""

echo "--- Venv ---"
VENV_DIR="$HOME/.venvs/nav4rail_bench"
# Match the partition-scoped naming used by _common.sh
if [ -n "${SLURM_JOB_PARTITION:-}" ]; then
    VENV_DIR="${VENV_DIR}_${SLURM_JOB_PARTITION}"
fi
echo "VENV_DIR: $VENV_DIR"
if [ -d "$VENV_DIR" ]; then
    echo "  EXISTS"
    echo "  pyvenv.cfg:"
    cat "$VENV_DIR/pyvenv.cfg" 2>/dev/null | sed 's/^/    /'
    echo ""
    echo "  bin/python -> $(readlink -f "$VENV_DIR/bin/python" 2>/dev/null || echo BROKEN_SYMLINK)"
    echo "  bin/python version: $("$VENV_DIR/bin/python" --version 2>&1 || echo FAILED)"
else
    echo "  DOES NOT EXIST"
fi
echo ""

echo "--- Activate venv ---"
source "$VENV_DIR/bin/activate" 2>&1 || echo "  FAILED to activate"
unset PYTHONHOME 2>/dev/null || true
echo "  After activate + unset PYTHONHOME:"
echo "    which python: $(which python)"
echo "    python --version: $(python --version 2>&1)"
echo "    PYTHONHOME: ${PYTHONHOME:-<unset>}"
echo ""

echo "--- Key packages ---"
python -c "
packages = ['yaml', 'torch', 'transformers', 'peft', 'trl', 'bitsandbytes', 'lxml', 'zss']
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'OK')
        print(f'  {pkg:20s} {ver}')
    except ImportError as e:
        print(f'  {pkg:20s} MISSING ({e})')
" 2>&1
echo ""

echo "--- PYTHONPATH check ---"
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${BENCH_DIR}"
echo "  BENCH_DIR: $BENCH_DIR"
echo "  PYTHONPATH: $PYTHONPATH"
python -c "from src.utils.config import load_config; print('  src.utils.config: OK')" 2>&1
python -c "from src.data.skills_loader import SkillsCatalog; print('  src.data.skills_loader: OK')" 2>&1
python -c "from src.eval.validate_bt import validate_bt; print('  src.eval.validate_bt: OK')" 2>&1
echo ""

echo "=========================================="
echo "  Diagnostic complete"
echo "=========================================="
