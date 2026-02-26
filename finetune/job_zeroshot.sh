#!/bin/bash
#SBATCH --job-name=nav4rail_zeroshot
#SBATCH --output=nav4rail_zeroshot_%j.out
#SBATCH --error=nav4rail_zeroshot_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --mem=32G

module load python/3.11.13 cuda/12.4.1

source ~/venv_nav4rail/bin/activate

cd ~/code/nav4rail_finetune

echo "=== Zero-shot eval (Mistral-7B sans adapter) ==="
python3 finetune_lora_xml.py --model mistral --zero-shot
