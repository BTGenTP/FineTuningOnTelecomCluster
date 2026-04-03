#!/bin/bash
# ─── SLURM — NAV4RAIL Fine-Tuning QLoRA v5 (multi-subtree) ─────────────────
#SBATCH --job-name=nav4rail_v5
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1                  # 1x RTX 3090 24GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                     # v5: séquences 3x plus longues → plus de RAM
#SBATCH --time=36:00:00               # v5: max_seq_len 5120 → ~8-12h estimé (max QOS: 36h)

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
        rich \
        lm-format-enforcer
    echo "[SETUP] Dépendances installées."
else
    source "$VENV_DIR/bin/activate"
    echo "[SETUP] Venv existant chargé."
fi

echo ""
echo "Python   : $(python3 --version)"
echo "PyTorch  : $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA OK  : $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU mem  : $(python3 -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB\")' 2>/dev/null || echo 'N/A')"
echo ""

# ─── Lancement ──────────────────────────────────────────────────────────────
cd "$WORK_DIR"

MODEL="${MODEL:-mistral}"
echo "[RUN] Fine-tuning v5 (multi-subtree) avec modèle : $MODEL"
echo "[RUN] Dataset  : dataset_nav4rail_v5.jsonl (2000 ex., multi-BT, ports blackboard)"
echo "[RUN] Params   : max_seq_len=5120, batch=2, grad_accum=32, epochs=15"

# Cache HuggingFace
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/hub"

python3 finetune_lora_xml.py --model "$MODEL"

echo ""
echo "========================================"
echo "Job terminé : $(date)"
echo "Outputs dans : $WORK_DIR/outputs/"
ls -lh "$WORK_DIR/outputs/" 2>/dev/null || echo "(dossier outputs vide)"
echo "========================================"
