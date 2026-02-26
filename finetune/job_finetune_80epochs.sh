#!/bin/bash
#SBATCH --job-name=nav4rail_80ep
#SBATCH --output=nav4rail_finetune_80ep_%j.out
#SBATCH --error=nav4rail_finetune_80ep_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mem=32G

module load python/3.11.13 cuda/12.4.1

# Venv partagé avec les runs précédents
VENV=~/venv_nav4rail
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers==4.41.2 peft==0.11.1 trl==0.9.4 \
                bitsandbytes==0.43.1 accelerate==0.30.1 datasets \
                lm-format-enforcer rich
else
    source "$VENV/bin/activate"
    # Patch : installe les dépendances manquantes si le venv est incomplet
    pip install rich lm-format-enforcer --quiet
fi

echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Nœud     : $(hostname)"
echo "GPU      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date     : $(date)"
echo "========================================"

cd ~/code/nav4rail_finetune

echo "[RUN] Fine-tuning Mistral-7B — 80 epochs — dataset v3 (550 ex.)"
python3 finetune_lora_xml.py --model mistral --epochs 80
