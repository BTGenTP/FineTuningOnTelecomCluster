"""
Common types shared by NAV4RAIL agents.
=======================================

Hosts ``AgentResult`` so that direct-XML agents (``react_base_agent``) and
code-as-reasoning agents (``pot_agent``, ``react_pot_agent``) do not have to
import from each other. Both modules import ``AgentResult`` from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agents.sandbox import ExecutionResult


@dataclass
class AgentResult:
    """Outcome of a single agent run on one mission.

    Used by every agent (PoT, ReAct PoT, ReAct Base). Fields not relevant to a
    given variant default to neutral values:

    - ``code`` is empty for direct-XML agents.
    - ``execution_results`` is empty when there is no sandbox stage.
    - ``sandbox_latency_s`` is 0.0 when there is no sandbox stage.
    """

    mission: str
    xml: str = ""
    success: bool = False  # True iff a valid <root> XML was produced
    code: str = ""

    # Latency breakdown
    total_latency_s: float = 0.0
    llm_latency_s: float = 0.0
    sandbox_latency_s: float = 0.0

    # Token counts (LLM-side)
    n_tokens: int = 0

    # Iteration count (1 for one-shot agents like PoT)
    n_iterations: int = 1

    # Sandbox execution log (empty for direct-XML agents)
    execution_results: list[ExecutionResult] = field(default_factory=list)

    # Final error if success=False
    error_type: str | None = None
    error_message: str | None = None

    # Full trace for debugging / W&B logging
    trace: list[dict[str, Any]] = field(default_factory=list)
