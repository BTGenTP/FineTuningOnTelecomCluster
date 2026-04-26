"""
ReAct / Reflexion PoT agent for NAV4RAIL — iterative code generation.
=====================================================================

Iterative Code-as-Reasoning loop (Program-of-Thoughts variant). The LLM writes
a Python script against the MissionBuilder API, the sandbox executes it, and
the validator scores the resulting XML. If the score is below the target
threshold (or execution failed), the error feedback is appended to the prompt
and the LLM retries until either the target score is reached or
`max_iterations` is hit.

For direct XML inference (no intermediate Python code), see
`src/agents/react_base_agent.py` (`ReActBaseAgent`).

Implementation: LangGraph state machine with four nodes:

    generate_code → execute_code → validate → should_retry ─┐
          ▲────────────────────────────────────────────────┘
                              (retry)

State flows through each node. `should_retry` is a conditional edge that
routes to END when done, back to generate_code otherwise.

LangGraph is an optional dependency. If unavailable, we fall back to the
same logic expressed as a plain Python loop — the node functions are
identical.
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

from src.agents.base_agent import AgentResult
from src.agents.pot_agent import extract_code
from src.agents.sandbox import extract_xml_from_result, run_sandboxed
from src.data.skills_loader import SafetyRulesLoader, SkillsCatalog

logger = logging.getLogger(__name__)


# ── State ────────────────────────────────────────────────────────────────────


class ReActState(TypedDict, total=False):
    """Mutable state shared across the LangGraph nodes."""

    # Inputs (set once at entry)
    mission: str
    max_iterations: int
    target_score: float

    # Per-iteration fields
    iteration: int
    completion: str
    code: str
    exec_result: Any  # ExecutionResult
    xml: str
    validator_score: float
    validator_valid: bool
    validator_errors: list[str]
    validator_warnings: list[str]

    # Accumulated
    history: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    llm_latency_s: float
    sandbox_latency_s: float
    n_tokens: int

    # Terminal
    done: bool
    last_error: str | None
    last_error_type: str | None


# ── Node functions (pure-ish; mutate state in place for LangGraph) ───────────


class ReActPoTAgent:
    """
    ReAct / Reflexion PoT agent. Iterates generate_code→execute→validate→reflect
    until the validation score hits `target_score` or `max_iterations` is
    exhausted. The LLM writes Python (PoT); the sandbox produces the XML.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_config: dict,
        react_cfg: dict | None = None,
        catalog: SkillsCatalog | None = None,
        safety_rules: SafetyRulesLoader | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config or {}
        self.react_cfg = react_cfg or {}
        self.catalog = catalog or SkillsCatalog()
        self.safety_rules = safety_rules

        self.temperature = self.react_cfg.get("temperature", 0.3)
        self.top_p = self.react_cfg.get("top_p", 0.9)
        self.max_new_tokens = self.react_cfg.get("max_new_tokens", 2048)
        self.sandbox_timeout_s = self.react_cfg.get("sandbox_timeout_s", 10.0)
        self.max_iterations = int(self.react_cfg.get("max_iterations", 3))
        self.target_score = float(self.react_cfg.get("target_score", 1.0))
        self.use_langgraph = bool(self.react_cfg.get("use_langgraph", True))

    # ── LLM call (shared with PoT) ───────────────────────────────────────────

    def _llm_generate(self, prompt) -> tuple[str, float, int]:
        """Single completion. Returns (text, latency_s, n_tokens)."""
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

    # ── Node: generate_code ──────────────────────────────────────────────────

    def generate_code(self, state: ReActState) -> ReActState:
        """Build a prompt (with history), call the LLM, extract code."""
        from src.data.prompt_builder import build_prompt

        state["iteration"] = state.get("iteration", 0) + 1

        prompt = build_prompt(
            mode="react_pot_agent",
            mission=state["mission"],
            model_config=self.model_config,
            catalog=self.catalog,
            safety_rules=self.safety_rules,
            history=state.get("history", []),
        )

        completion, latency, n_tokens = self._llm_generate(prompt)
        state["completion"] = completion
        state["llm_latency_s"] = state.get("llm_latency_s", 0.0) + latency
        state["n_tokens"] = state.get("n_tokens", 0) + n_tokens

        code = extract_code(completion)
        state["code"] = code

        trace = state.setdefault("trace", [])
        trace.append({
            "step": "generate",
            "iteration": state["iteration"],
            "completion_len": len(completion),
            "code_len": len(code),
        })

        return state

    # ── Node: execute_code ───────────────────────────────────────────────────

    def execute_code(self, state: ReActState) -> ReActState:
        """Run the extracted code in the sandbox and capture the result."""
        code = state.get("code", "")
        if not code:
            state["exec_result"] = None
            state["xml"] = ""
            state["last_error"] = "Could not extract any Python code"
            state["last_error_type"] = "NoCodeExtracted"
            state.setdefault("trace", []).append({
                "step": "execute",
                "iteration": state["iteration"],
                "skipped": True,
                "reason": "no code",
            })
            return state

        exec_result = run_sandboxed(
            code, catalog=self.catalog, timeout_s=self.sandbox_timeout_s
        )
        state["exec_result"] = exec_result
        state["sandbox_latency_s"] = (
            state.get("sandbox_latency_s", 0.0) + exec_result.exec_time_s
        )

        xml = extract_xml_from_result(exec_result) or ""
        state["xml"] = xml

        if not exec_result.success:
            state["last_error"] = exec_result.error_summary
            state["last_error_type"] = exec_result.error_type
        elif not xml:
            state["last_error"] = "Execution succeeded but produced no <root> XML"
            state["last_error_type"] = "NoXMLExtracted"
        else:
            state["last_error"] = None
            state["last_error_type"] = None

        state.setdefault("trace", []).append({
            "step": "execute",
            "iteration": state["iteration"],
            "success": exec_result.success,
            "xml_found": bool(xml),
            "error": exec_result.error_summary,
        })
        return state

    # ── Node: validate ───────────────────────────────────────────────────────

    def validate(self, state: ReActState) -> ReActState:
        """Validate the XML with the L1-L4 validator. Stores score + errors."""
        from src.eval.validate_bt import enrich_ports, validate_bt

        xml = state.get("xml", "")
        if not xml:
            state["validator_score"] = 0.0
            state["validator_valid"] = False
            state["validator_errors"] = ["No XML to validate"]
            state["validator_warnings"] = []
            state.setdefault("trace", []).append({
                "step": "validate",
                "iteration": state["iteration"],
                "score": 0.0,
                "valid": False,
                "skipped_reason": "no xml",
            })
            return state

        # Match benchmark behavior: enrich missing ports before validating.
        try:
            xml_enriched = enrich_ports(xml, self.catalog)
        except Exception as e:  # noqa: BLE001
            logger.debug("enrich_ports failed: %s", e)
            xml_enriched = xml

        result = validate_bt(xml_enriched, self.catalog)
        state["xml"] = xml_enriched
        state["validator_score"] = float(result.score)
        state["validator_valid"] = bool(result.valid)
        state["validator_errors"] = list(result.errors)
        state["validator_warnings"] = list(result.warnings)

        state.setdefault("trace", []).append({
            "step": "validate",
            "iteration": state["iteration"],
            "score": result.score,
            "valid": result.valid,
            "n_errors": len(result.errors),
            "n_warnings": len(result.warnings),
        })
        return state

    # ── Node: reflect (also conditional router) ──────────────────────────────

    def reflect(self, state: ReActState) -> ReActState:
        """
        Decide whether to retry. Always record the current attempt in history
        so the next prompt can reference it.
        """
        score = state.get("validator_score", 0.0)
        iteration = state.get("iteration", 1)
        max_iter = state.get("max_iterations", self.max_iterations)
        target = state.get("target_score", self.target_score)

        # Build a validator feedback string for the LLM.
        errors = state.get("validator_errors", [])
        warnings = state.get("validator_warnings", [])
        validator_parts = [f"score={score:.2f}, valid={state.get('validator_valid', False)}"]
        if errors:
            validator_parts.append("errors: " + "; ".join(errors[:3]))
        if warnings:
            validator_parts.append("warnings: " + "; ".join(warnings[:3]))
        validator_feedback = " | ".join(validator_parts)

        state.setdefault("history", []).append({
            "iteration": iteration,
            "code": state.get("code", ""),
            "error": state.get("last_error"),
            "validator": validator_feedback,
            "score": score,
        })

        reached_target = score >= target and state.get("last_error") is None
        exhausted = iteration >= max_iter
        state["done"] = reached_target or exhausted

        state.setdefault("trace", []).append({
            "step": "reflect",
            "iteration": iteration,
            "done": state["done"],
            "reason": (
                "target_reached" if reached_target
                else "max_iterations" if exhausted
                else "retry"
            ),
        })
        return state

    # ── Graph wiring ─────────────────────────────────────────────────────────

    def _build_langgraph(self):
        """Build the LangGraph StateGraph. Raises ImportError if unavailable."""
        from langgraph.graph import END, StateGraph

        graph = StateGraph(ReActState)
        graph.add_node("generate_code", self.generate_code)
        graph.add_node("execute_code", self.execute_code)
        graph.add_node("validate", self.validate)
        graph.add_node("reflect", self.reflect)

        graph.set_entry_point("generate_code")
        graph.add_edge("generate_code", "execute_code")
        graph.add_edge("execute_code", "validate")
        graph.add_edge("validate", "reflect")

        def _route(state: ReActState) -> str:
            return END if state.get("done") else "generate_code"

        graph.add_conditional_edges(
            "reflect", _route, {"generate_code": "generate_code", END: END}
        )
        return graph.compile()

    def _run_plain_loop(self, state: ReActState) -> ReActState:
        """Pure-Python fallback when LangGraph isn't installed."""
        while True:
            self.generate_code(state)
            self.execute_code(state)
            self.validate(state)
            self.reflect(state)
            if state.get("done"):
                return state

    # ── Public entrypoint ────────────────────────────────────────────────────

    def run(self, mission: str) -> AgentResult:
        """Run the iterative ReAct loop on a single mission."""
        t_start = time.perf_counter()
        state: ReActState = {
            "mission": mission,
            "max_iterations": self.max_iterations,
            "target_score": self.target_score,
            "iteration": 0,
            "history": [],
            "trace": [],
            "llm_latency_s": 0.0,
            "sandbox_latency_s": 0.0,
            "n_tokens": 0,
        }

        if self.use_langgraph:
            try:
                graph = self._build_langgraph()
                # Recursion limit: each iteration = 4 nodes, plus a safety margin.
                state = graph.invoke(
                    state, config={"recursion_limit": 4 * self.max_iterations + 8}
                )
            except ImportError:
                logger.info("LangGraph unavailable — falling back to plain loop")
                state = self._run_plain_loop(state)
        else:
            state = self._run_plain_loop(state)

        total_latency = time.perf_counter() - t_start

        exec_results = []
        if state.get("exec_result") is not None:
            exec_results.append(state["exec_result"])

        result = AgentResult(
            mission=mission,
            xml=state.get("xml", "") if state.get("validator_valid") else state.get("xml", ""),
            success=bool(state.get("validator_valid") and state.get("xml")),
            code=state.get("code", ""),
            total_latency_s=total_latency,
            llm_latency_s=state.get("llm_latency_s", 0.0),
            sandbox_latency_s=state.get("sandbox_latency_s", 0.0),
            n_tokens=state.get("n_tokens", 0),
            n_iterations=state.get("iteration", 0),
            execution_results=exec_results,
            error_type=state.get("last_error_type"),
            error_message=state.get("last_error"),
            trace=state.get("trace", []),
        )
        return result
