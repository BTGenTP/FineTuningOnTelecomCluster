"""
ROS2 Nav2 inference pipeline for strict JSON steps generation.

This module intentionally keeps heavy ML imports lazy so the webapp can still
start on machines where the Nav2 HF/LoRA stack is not installed yet.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from nav2_pipeline import (
    build_xml_from_steps,
    load_nav2_catalog,
    parse_steps_payload,
    write_nav2_run_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_Nav2.train.model_registry import MODELS  # noqa: E402
from finetune_Nav2.train.prompting import (  # noqa: E402
    build_chat_messages,
    build_mistral_inst_prompt,
    build_phi2_prompt,
)

DEFAULT_ADAPTER_DIR = (
    PROJECT_ROOT
    / "finetune_Nav2"
    / "outputs"
    / "nav2_steps_mistral7b_lora_743587"
    / "lora_adapter"
)

TEST_MISSIONS_NAV2 = [
    "Navigue vers le goal (Nav2), puis attends 2.0 s.",
    "Navigue vers le goal (Nav2), puis tourne de 180° (3.14 rad).",
    "Navigue vers le goal (Nav2), puis recule de 0.3 m à 0.08 m/s.",
    "Attends 1.0 s puis tourne de 90° (1.57 rad).",
    "Navigue vers le goal (Nav2), puis efface la costmap locale, puis attends 0.5 s.",
]


def _make_prompt(model_key: str, tokenizer, catalog: Dict[str, Any], mission: str) -> str:
    spec = MODELS[model_key]
    if spec.chat_template:
        msgs = build_chat_messages(mission=mission, catalog=catalog)
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "\n### Steps JSON:\n"
    if model_key == "phi2":
        prompt, _ = build_phi2_prompt(mission=mission, catalog=catalog)
        return prompt
    prompt, _ = build_mistral_inst_prompt(mission=mission, catalog=catalog)
    return prompt


class Nav2Generator:
    def __init__(
        self,
        *,
        model_key: str = "mistral7b",
        adapter_dir: str | Path | None = None,
        load_in_4bit: bool = True,
    ) -> None:
        self.model_key = model_key
        self.adapter_dir = Path(adapter_dir or DEFAULT_ADAPTER_DIR).expanduser().resolve()
        self.load_in_4bit = bool(load_in_4bit)
        self.catalog = load_nav2_catalog()
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._load_error: Optional[str] = None

    @property
    def configured(self) -> bool:
        return self.adapter_dir.exists()

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _ensure_loaded(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            if not self.adapter_dir.exists():
                raise FileNotFoundError(f"Nav2 LoRA adapter not found: {self.adapter_dir}")

            try:
                import torch
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from transformers import BitsAndBytesConfig
            except Exception as exc:  # pragma: no cover
                self._load_error = str(exc)
                raise RuntimeError(
                    "Missing HF/LoRA dependencies for Nav2 mode. "
                    "Install torch, transformers, peft and bitsandbytes."
                ) from exc

            spec = MODELS[self.model_key]
            tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if self.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                base = AutoModelForCausalLM.from_pretrained(
                    spec.hf_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            else:
                device_map = "auto" if torch.cuda.is_available() else None
                base = AutoModelForCausalLM.from_pretrained(
                    spec.hf_id,
                    device_map=device_map,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                )

            model = PeftModel.from_pretrained(base, self.adapter_dir)
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._load_error = None

    def status(self) -> Dict[str, Any]:
        return {
            "configured": self.configured,
            "loaded": self.loaded,
            "model_key": self.model_key,
            "adapter_dir": str(self.adapter_dir),
            "load_error": self._load_error,
        }

    def generate(
        self,
        mission: str,
        *,
        constrained: str = "jsonschema",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        write_run: bool = False,
        strict_attrs: bool = True,
        strict_blackboard: bool = True,
    ) -> Dict[str, Any]:
        self._ensure_loaded()

        assert self._tokenizer is not None
        assert self._model is not None

        tokenizer = self._tokenizer
        model = self._model
        prompt = _make_prompt(self.model_key, tokenizer, self.catalog, mission)

        prefix_fn = None
        if constrained == "jsonschema":
            from finetune_Nav2.constraints.steps_prefix_fn import build_prefix_allowed_tokens_fn

            prefix_fn = build_prefix_allowed_tokens_fn(tokenizer, self.catalog)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with self._lock:
            t0 = time.perf_counter()
            output = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(float(temperature) > 0.0),
                temperature=float(temperature) if float(temperature) > 0.0 else 1.0,
                pad_token_id=tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_fn,
            )
            latency_ms = int((time.perf_counter() - t0) * 1000.0)

        gen_ids = output[0][inputs["input_ids"].shape[1] :]
        raw_steps = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        parsed_payload = parse_steps_payload(raw_steps, catalog=self.catalog)

        result: Dict[str, Any] = {
            "provider": "hf_local_peft",
            "model": MODELS[self.model_key].hf_id,
            "model_key": self.model_key,
            "mission": mission,
            "prompt": prompt,
            "raw_steps": raw_steps,
            "steps": parsed_payload["steps"],
            "steps_json": parsed_payload["steps_json"],
            "steps_valid": parsed_payload["ok"],
            "parse_error": parsed_payload["error_message"],
            "parse_error_counters": parsed_payload["error_counters"],
            "generation_time_s": round(latency_ms / 1000.0, 2),
            "constraint_mode": constrained,
        }

        parsed = parsed_payload["_parsed"]
        if parsed.ok and parsed.steps:
            xml_payload = build_xml_from_steps(
                parsed.steps,
                catalog=self.catalog,
                strict_attrs=strict_attrs,
                strict_blackboard=strict_blackboard,
            )
            result.update(xml_payload)
            if write_run:
                result["run_dir"] = write_nav2_run_artifacts(
                    mission=mission,
                    prompt=prompt,
                    llm_raw=raw_steps,
                    parsed=parsed,
                    provider="hf_local_peft",
                    model_name=MODELS[self.model_key].hf_id,
                    temperature=float(temperature),
                    constraints_kind=constrained,
                    strict_attrs=strict_attrs,
                    strict_blackboard=strict_blackboard,
                    latency_ms=latency_ms,
                    xml_payload=xml_payload,
                )
        else:
            result.update(
                {
                    "xml": "",
                    "validation_report": {"ok": False, "issues": []},
                    "valid": False,
                    "score": 0.0,
                    "errors": parsed_payload["errors"],
                    "warnings": [],
                    "summary": parsed_payload["error_message"] or "Steps JSON invalid",
                    "structure": {},
                }
            )
            if write_run:
                result["run_dir"] = write_nav2_run_artifacts(
                    mission=mission,
                    prompt=prompt,
                    llm_raw=raw_steps,
                    parsed=parsed,
                    provider="hf_local_peft",
                    model_name=MODELS[self.model_key].hf_id,
                    temperature=float(temperature),
                    constraints_kind=constrained,
                    strict_attrs=strict_attrs,
                    strict_blackboard=strict_blackboard,
                    latency_ms=latency_ms,
                )

        return result


def build_nav2_generator_from_env() -> Nav2Generator:
    model_key = os.getenv("NAV2_MODEL_KEY", "mistral7b")
    adapter_dir = os.getenv("NAV2_ADAPTER_DIR") or str(DEFAULT_ADAPTER_DIR)
    load_in_4bit = os.getenv("NAV2_LOAD_IN_4BIT", "1") not in {"0", "false", "False"}
    return Nav2Generator(model_key=model_key, adapter_dir=adapter_dir, load_in_4bit=load_in_4bit)
