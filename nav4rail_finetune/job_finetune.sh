#!/bin/bash
# ─── SLURM — NAV4RAIL Fine-Tuning QLoRA ────────────────────────────────────
#SBATCH --job-name=nav4rail_finetune
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1                  # 1x Tesla P100-PCIE-16GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00               # Mistral ~3-4h / TinyLlama ~30min

# ─── Infos job ───────────────────────────────────────────────────────────────
echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Nœud     : $(hostname)"
echo "GPU      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date     : $(date)"
echo "========================================"

# ─── Environnement ──────────────────────────────────────────────────────────
module load python/3.11.13 cuda/12.4.1

WORK_DIR="$HOME/code/nav4rail_finetune"
VENV_DIR="$HOME/venvs/nav4rail"

# Créer le venv si absent
if [ ! -d "$VENV_DIR" ]; then
    echo "[SETUP] Création du venv..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip --quiet

    echo "[SETUP] Installation des dépendances..."
    pip install --quiet \
        torch==2.3.0 \
        transformers==4.44.0 \
        peft==0.12.0 \
        trl==0.10.1 \
        bitsandbytes==0.43.3 \
        datasets==2.21.0 \
        accelerate==0.34.0 \
        scipy \
        sentencepiece \
        protobuf \
        rich
    echo "[SETUP] Dépendances installées."
else
    source "$VENV_DIR/bin/activate"
    echo "[SETUP] Venv existant chargé."
fi

echo ""
echo "Python   : $(python3 --version)"
echo "PyTorch  : $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA OK  : $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# ─── Lancement ──────────────────────────────────────────────────────────────
cd "$WORK_DIR"

# Modèle : mistral (défaut) ou tinyllama (rapide, pour debug)
MODEL="${MODEL:-mistral}"
echo "[RUN] Fine-tuning avec modèle : $MODEL"

# Cache HuggingFace dans le home cluster (espace suffisant)
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/hub"

python3 finetune_lora_xml.py --model "$MODEL"

echo ""
echo "========================================"
echo "Job terminé : $(date)"
echo "Outputs dans : $WORK_DIR/outputs/"
ls -lh "$WORK_DIR/outputs/" 2>/dev/null || echo "(dossier outputs vide)"
echo "========================================"