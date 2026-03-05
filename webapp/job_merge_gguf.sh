#!/bin/bash
#SBATCH --job-name=nav4rail_merge_gguf
#SBATCH --output=merge_gguf_%j.out
#SBATCH --error=merge_gguf_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G

module load python/3.11.13 cuda/12.4.1

VENV=~/venv_nav4rail
source "$VENV/bin/activate"

cd ~/code/nav4rail_finetune

echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Date     : $(date)"
echo "========================================"

# Step 1: Merge LoRA adapter into base model
echo "[1/3] Merging LoRA adapter into base model..."
python3 merge_and_convert.py \
    --adapter-path outputs/nav4rail_mistral_lora/lora_adapter \
    --output-dir merged_model

echo "[1/3] Done. Merged model saved to merged_model/"

# Step 2: Install llama.cpp conversion dependencies
echo "[2/3] Setting up llama.cpp converter..."
if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp
fi
pip install -q gguf numpy sentencepiece protobuf

# Step 3: Convert to GGUF Q4_K_M
echo "[3/3] Converting to GGUF Q4_K_M..."
python3 llama.cpp/convert_hf_to_gguf.py merged_model \
    --outtype q4_k_m \
    --outfile nav4rail-mistral-7b-q4_k_m.gguf

echo "========================================"
echo "Done! GGUF file:"
ls -lh nav4rail-mistral-7b-q4_k_m.gguf
echo "========================================"
echo "Transfer with: scp gpu:~/code/nav4rail_finetune/nav4rail-mistral-7b-q4_k_m.gguf ~/Telecom_Projet_fil_rouge/webapp/models/"
