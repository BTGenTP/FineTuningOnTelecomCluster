"""
Program-of-Thoughts (PoT) agent.
================================

One-shot code generation agent. The LLM receives the MissionBuilder API docs
plus the mission text, generates a Python script, the sandbox executes it,
and the stdout/`xml` variable becomes the BT XML.

There is no loop, no reflexion — hence "one-shot". Use ReActAgent for the
iterative variant with error-driven refinement.

Interface:
    agent = PoTAgent(model, tokenizer, model_config, pot_cfg, catalog)
    result = agent.run(mission_text)
    # result.xml, result.success, result.code, result.latency_s, ...
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from src.agents.sandbox import ExecutionResult, extract_xml_from_result, run_sandboxed
from src.data.skills_loader import SafetyRulesLoader, SkillsCatalog

logger = logging.getLogger(__name__)


# ── Result type ──────────────────────────────────────────────────────────────


@dataclass
class AgentResult:
    """Outcome of a single agent run on one mission."""

    mission: str
    xml: str = ""
    success: bool = False  # True iff sandbox ran AND an <root> was extracted
    code: str = ""
    # Latency breakdown
    total_latency_s: float = 0.0
    llm_latency_s: float = 0.0
    sandbox_latency_s: float = 0.0
    # Token counts (from the LLM call)
    n_tokens: int = 0
    # Execution info
    n_iterations: int = 1
    execution_results: list[ExecutionResult] = field(default_factory=list)
    # Final error if success=False
    error_type: str | None = None
    error_message: str | None = None
    # Full trace for debugging
    trace: list[dict[str, Any]] = field(default_factory=list)


# ── Code extraction ──────────────────────────────────────────────────────────


_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)


def extract_code(completion: str) -> str:
    """
    Pull the Python code block out of an LLM completion.

    Priority:
      1. First ```python ... ``` fenced block.
      2. First ``` ... ``` block (language-agnostic).
      3. The completion as-is, if it starts with an import/def/other Python token.
      4. Empty string otherwise.
    """
    m = _CODE_FENCE_RE.search(completion)
    if m:
        return m.group(1).strip()

    # Generic fence
    m = re.search(r"```\s*\n(.*?)\n```", completion, re.DOTALL)
    if m:
        return m.group(1).strip()

    stripped = completion.strip()
    if stripped.startswith(("from ", "import ", "builder", "b ", "def ", "#")):
        return stripped

    return ""


# ── PoT Agent ────────────────────────────────────────────────────────────────


class PoTAgent:
    """
    Program-of-Thoughts agent. Single LLM call, single sandbox execution.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_config: dict,
        pot_cfg: dict | None = None,
        catalog: SkillsCatalog | None = None,
        safety_rules: SafetyRulesLoader | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config or {}
        self.pot_cfg = pot_cfg or {}
        self.catalog = catalog or SkillsCatalog()
        self.safety_rules = safety_rules
        self.temperature = self.pot_cfg.get("temperature", 0.2)
        self.top_p = self.pot_cfg.get("top_p", 0.9)
        self.max_new_tokens = self.pot_cfg.get("max_new_tokens", 2048)
        self.sandbox_timeout_s = self.pot_cfg.get("sandbox_timeout_s", 10.0)

    def _llm_generate(self, prompt) -> tuple[str, float, int]:
        """Generate a single completion. Returns (text, latency_s, n_tokens)."""
        import torch

        if isinstance(prompt, list):
            text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        latency = time.perf_counter() - t0

        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        n_tokens = int(new_tokens.shape[0])
        completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return completion, latency, n_tokens

    def run(self, mission: str) -> AgentResult:
        """Run one full PoT attempt on a mission."""
        from src.data.prompt_builder import build_prompt

        t_start = time.perf_counter()
        result = AgentResult(mission=mission)

        prompt = build_prompt(
            mode="program_of_thoughts",
            mission=mission,
            model_config=self.model_config,
            catalog=self.catalog,
            safety_rules=self.safety_rules,
        )

        completion, llm_latency, n_tokens = self._llm_generate(prompt)
        result.llm_latency_s = llm_latency
        result.n_tokens = n_tokens

        code = extract_code(completion)
        result.code = code
        result.trace.append(
            {"step": "generate", "completion": completion, "code": code}
        )

        if not code:
            result.error_type = "NoCodeExtracted"
            result.error_message = "Could not extract any Python code from the LLM completion"
            result.total_latency_s = time.perf_counter() - t_start
            return result

        exec_result = run_sandboxed(
            code, catalog=self.catalog, timeout_s=self.sandbox_timeout_s
        )
        result.execution_results.append(exec_result)
        result.sandbox_latency_s = exec_result.exec_time_s
        result.trace.append(
            {
                "step": "execute",
                "success": exec_result.success,
                "stdout_len": len(exec_result.stdout),
                "error": exec_result.error_summary,
            }
        )

        xml = extract_xml_from_result(exec_result)
        if xml:
            result.xml = xml
            result.success = True
        else:
            result.error_type = exec_result.error_type or "NoXMLExtracted"
            result.error_message = (
                exec_result.error_summary or "Execution produced no <root> XML"
            )

        result.total_latency_s = time.perf_counter() - t_start
        return result
