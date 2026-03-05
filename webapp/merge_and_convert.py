"""
Merge LoRA adapter into base model and save for GGUF conversion.

Run this on the cluster (needs GPU + ~16 GB RAM):
    python merge_and_convert.py \
        --adapter-path ../finetune/outputs/nav4rail_mistral_lora/lora_adapter \
        --output-dir ./merged_model

Then convert to GGUF with llama.cpp:
    git clone https://github.com/ggml-org/llama.cpp
    cd llama.cpp && pip install -r requirements.txt
    python convert_hf_to_gguf.py ../merged_model \
        --outtype q4_k_m \
        --outfile ../nav4rail-mistral-7b-q4_k_m.gguf

Transfer locally:
    scp gpu:~/code/nav4rail_finetune/nav4rail-mistral-7b-q4_k_m.gguf \
        ~/Telecom_Projet_fil_rouge/webapp/models/
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="HuggingFace model ID")
    parser.add_argument("--adapter-path", required=True,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--output-dir", default="./merged_model",
                        help="Output directory for merged model")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_dir}")
    model.save_pretrained(str(output_dir))

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    tokenizer.save_pretrained(str(output_dir))

    print("Done. Next step: convert to GGUF with llama.cpp convert_hf_to_gguf.py")


if __name__ == "__main__":
    main()
