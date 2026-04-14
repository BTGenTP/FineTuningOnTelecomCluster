"""
Unified model loader for NAV4RAIL benchmarking.
================================================
Loads models for training or inference with configurable PEFT methods.

Supports: LoRA, QLoRA, DoRA, OFT, and no-adapter (zero-shot).
Reuses patterns from finetune/finetune_llama3_nav4rail.py.

Usage:
    from src.utils.model_loader import load_for_training, load_for_inference
    model, tokenizer = load_for_training(cfg)
    model, tokenizer = load_for_inference(cfg, adapter_path="runs/.../checkpoint")
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


def _build_bnb_config(cfg: dict) -> BitsAndBytesConfig | None:
    """Build BitsAndBytes 4-bit quantization config."""
    quant = cfg.get("quantization", {})
    if not quant.get("load_in_4bit", False):
        return None

    # bitsandbytes CUDA kernels require compute capability >= 7.5 (sm_75).
    # On older GPUs (e.g. Tesla P100 = sm_60), skip quantization → fp16 fallback.
    if torch.cuda.is_available():
        cc_major, cc_minor = torch.cuda.get_device_capability()
        if cc_major < 7 or (cc_major == 7 and cc_minor < 5):
            logger.warning(
                "GPU compute capability %d.%d < 7.5: disabling 4-bit quantization "
                "(bitsandbytes requires sm_75+). Falling back to fp16.",
                cc_major, cc_minor,
            )
            return None

    compute_dtype_str = quant.get("bnb_4bit_compute_dtype", "float16")
    compute_dtype = getattr(torch, compute_dtype_str, torch.float16)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant.get("bnb_4bit_use_double_quant", True),
    )


def _build_peft_config(cfg: dict):
    """Build PEFT config (LoRA, DoRA, OFT, or None)."""
    peft_cfg = cfg.get("peft", {})
    method = peft_cfg.get("method", "qlora")

    if method == "none":
        return None

    target_modules = peft_cfg.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    r = peft_cfg.get("r", 16)
    alpha = peft_cfg.get("alpha", 32)
    dropout = peft_cfg.get("dropout", 0.05)
    bias = peft_cfg.get("bias", "none")

    if method in ("lora", "qlora"):
        from peft import LoraConfig, TaskType

        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias=bias,
            target_modules=target_modules,
        )
    elif method == "dora":
        from peft import LoraConfig, TaskType

        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias=bias,
            target_modules=target_modules,
            use_dora=True,
        )
    elif method == "oft":
        from peft import OFTConfig

        return OFTConfig(
            r=r,
            target_modules=target_modules,
        )
    else:
        raise ValueError(f"Unknown PEFT method: {method}")


def _get_model_cfg(cfg: dict) -> dict[str, Any]:
    """Get the active model's config from the registry."""
    model_key = cfg.get("model", {}).get("key", "mistral_7b")
    return cfg.get("models", {}).get(model_key, {})


def load_for_training(cfg: dict) -> tuple:
    """
    Load model and tokenizer for training.

    Returns:
        (model, tokenizer) tuple ready for SFTTrainer/DPOTrainer/etc.
    """
    model_cfg = _get_model_cfg(cfg)
    hf_id = model_cfg["hf_id"]
    bf16 = model_cfg.get("bf16", True)

    logger.info(f"Loading model for training: {hf_id}")

    # Quantization
    bnb_config = _build_bnb_config(cfg)

    # Model kwargs
    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    else:
        if bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            logger.warning("GPU does not support bf16, falling back to fp16")
            bf16 = False
        model_kwargs["torch_dtype"] = torch.bfloat16 if bf16 else torch.float16

    attn_impl = model_cfg.get("attn_implementation")
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # PEFT
    peft_config = _build_peft_config(cfg)
    if peft_config is not None:
        from peft import get_peft_model, prepare_model_for_kbit_training

        if bnb_config:
            model = prepare_model_for_kbit_training(model)

        # Verify LoRA targets exist
        if hasattr(peft_config, "target_modules"):
            available = {name.split(".")[-1] for name, _ in model.named_modules()}
            missing = set(peft_config.target_modules) - available
            if missing:
                logger.warning(f"LoRA targets not found in model: {missing}")

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer


def load_for_inference(
    cfg: dict, adapter_path: str | None = None
) -> tuple:
    """
    Load model and tokenizer for inference/evaluation.

    Args:
        cfg: Configuration dict
        adapter_path: Path to PEFT adapter checkpoint (optional)

    Returns:
        (model, tokenizer) tuple ready for generation
    """
    model_cfg = _get_model_cfg(cfg)
    hf_id = model_cfg["hf_id"]
    bf16 = model_cfg.get("bf16", True)

    logger.info(f"Loading model for inference: {hf_id}")

    bnb_config = _build_bnb_config(cfg)

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    else:
        if bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            logger.warning("GPU does not support bf16, falling back to fp16")
            bf16 = False
        model_kwargs["torch_dtype"] = torch.bfloat16 if bf16 else torch.float16

    attn_impl = model_cfg.get("attn_implementation")
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Load adapter if provided
    if adapter_path:
        from peft import PeftModel

        logger.info(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer
