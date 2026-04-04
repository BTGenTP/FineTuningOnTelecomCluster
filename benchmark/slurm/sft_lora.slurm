#!/bin/bash
#SBATCH --job-name=nav4rail_sft_lora
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

cd "$(dirname "$0")/.."

source .venv/bin/activate || true

python3 -c "print('TODO: wire config loader + sft runner (src/methods/sft.py) into a CLI entrypoint.')"

