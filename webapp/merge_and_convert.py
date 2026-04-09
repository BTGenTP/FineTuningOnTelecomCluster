"""
Merge LoRA adapter into base model and save for GGUF conversion.

Auto-detects the base model from adapter_config.json if --base-model is not
specified explicitly.

Usage:
    python merge_and_convert.py --adapter-path ./lora_adapter --output-dir ./merged
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def detect_base_model(adapter_path: str) -> str:
    """Read base_model_name_or_path from adapter_config.json."""
    cfg_path = Path(adapter_path) / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No adapter_config.json in {adapter_path}. "
            "Specify --base-model explicitly."
        )
    cfg = json.loads(cfg_path.read_text())
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError(
            "adapter_config.json has no base_model_name_or_path. "
            "Specify --base-model explicitly."
        )
    return base


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--base-model",
        default=None,
        help="HuggingFace model ID (auto-detected from adapter_config.json if omitted)",
    )
    parser.add_argument(
        "--adapter-path", required=True, help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--output-dir",
        default="./merged_model",
        help="Output directory for merged model",
    )
    args = parser.parse_args()

    base_model_id = args.base_model or detect_base_model(args.adapter_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
    )

    print(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_dir}")
    model.save_pretrained(str(output_dir))

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    tokenizer.save_pretrained(str(output_dir))

    print(f"Done. Base model was: {base_model_id}")


if __name__ == "__main__":
    main()
