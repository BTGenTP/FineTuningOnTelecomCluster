from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional, Tuple

from ..contracts import ModelConfig, PeftConfig


DTYPE_MAP = {
    "fp32": "float32",
    "float32": "float32",
    "fp16": "float16",
    "float16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
}


def _torch_dtype(name: str) -> Any:
    import torch

    return getattr(torch, DTYPE_MAP.get(name, "bfloat16"))


def build_quantization_config(mode: Optional[str]) -> Any:
    if not mode:
        return None
    from transformers import BitsAndBytesConfig
    import torch

    normalized = mode.lower()
    if normalized == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if normalized in {"4bit", "nf4"}:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(f"Unsupported quantization mode: {mode}")


def load_tokenizer(config: ModelConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name_or_path or config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(config: ModelConfig) -> Any:
    from transformers import AutoModelForCausalLM

    kwargs = {
        "device_map": config.device_map,
        "torch_dtype": _torch_dtype(config.dtype),
        "trust_remote_code": config.trust_remote_code,
    }
    quantization_config = build_quantization_config(config.quantization)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if config.use_flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"
    return AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **kwargs)


def apply_peft(model: Any, peft_config: PeftConfig) -> Any:
    if not peft_config.method:
        return model
    from peft import IA3Config, LoraConfig, PeftModel, TaskType, get_peft_model

    if peft_config.adapter_path:
        return PeftModel.from_pretrained(model, peft_config.adapter_path)

    task_type = getattr(TaskType, peft_config.task_type, TaskType.CAUSAL_LM)
    method = peft_config.method.lower()
    if method in {"lora", "qlora"}:
        cfg = LoraConfig(
            r=peft_config.r,
            lora_alpha=peft_config.alpha,
            lora_dropout=peft_config.dropout,
            target_modules=peft_config.target_modules or None,
            bias=peft_config.bias,
            task_type=task_type,
        )
        return get_peft_model(model, cfg)
    if method in {"adapter", "ia3"}:
        cfg = IA3Config(task_type=task_type, target_modules=peft_config.target_modules or None)
        return get_peft_model(model, cfg)
    raise ValueError(f"Unsupported PEFT method: {peft_config.method}")


def load_model_bundle(config: ModelConfig, peft_config: Optional[PeftConfig] = None) -> Tuple[Any, Any]:
    tokenizer = load_tokenizer(config)
    model = load_base_model(config)
    if peft_config is not None:
        model = apply_peft(model, peft_config)
    return model, tokenizer


def summarize_model_setup(config: ModelConfig, peft_config: Optional[PeftConfig] = None) -> dict[str, Any]:
    return {
        "model": asdict(config),
        "peft": asdict(peft_config) if peft_config else None,
    }
