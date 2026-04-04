#!/bin/bash
#SBATCH --job-name=nav4rail_prompt_eval
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

cd "$(dirname "$0")/.."

source .venv/bin/activate || true

python3 -c "print('TODO: implement CLI entrypoint for prompt benchmark; use src/evaluation/runner.py from a small driver script.')"

