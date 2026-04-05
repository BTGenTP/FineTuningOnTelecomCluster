from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


def _messages_to_text(messages: Sequence[dict[str, str]]) -> str:
    # Fallback if tokenizer has no chat template.
    chunks: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        chunks.append(f"[{role.upper()}]\n{content}".strip())
    return "\n\n".join(chunks) + "\n\n[ASSISTANT]\n"


@dataclass(slots=True)
class HFChatGenerator:
    model: Any
    tokenizer: Any
    device: str | None = None

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int = -1,
        do_sample: bool,
    ) -> str:
        import torch

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = _messages_to_text(messages)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        if top_k is not None and int(top_k) > 0:
            gen_kwargs["top_k"] = int(top_k)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        gen_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

