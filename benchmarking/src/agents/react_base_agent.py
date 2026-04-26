"""
ReAct base agent for NAV4RAIL — direct XML inference with iterative refinement.
================================================================================

Same loop shape as `ReActPoTAgent`, but the LLM emits the BehaviorTree XML
directly (no Python intermediate, no sandbox). Validation feedback is fed
back through the prompt for the next iteration.

Inner prompt mode is configurable: zero_shot, few_shot, schema_guided, or
chain_of_thought. Constrained decoding (GBNF / Outlines) is honoured when a
ConstraintHandle is passed at construction.

Implementation: LangGraph state machine — generate_xml → validate → reflect.
Falls back to a plain Python loop when LangGraph is unavailable.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, TypedDict

from src.agents.base_agent import AgentResult
from src.data.skills_loader import SafetyRulesLoader, SkillsCatalog

logger = logging.getLogger(__name__)


_ROOT_BLOCK_RE = re.compile(r"<root[\s\S]*?</root\s*>", re.IGNORECASE)


def extract_xml(completion: str) -> str:
    """Pull the first <root>...</root> block from an LLM completion.

    Falls back to the raw text if the model emitted XML without an explicit
    `<root>` (e.g. constrained outlines mode that already returns XML).
    """
    m = _ROOT_BLOCK_RE.search(completion)
    if m:
        return m.group(0).strip()
    stripped = completion.strip()
    if stripped.startswith("<"):
        return stripped
    return ""


class ReActBaseState(TypedDict, total=False):
    mission: str
    max_iterations: int
    target_score: float

    iteration: int
    completion: str
    xml: str
    validator_score: float
    validator_valid: bool
    validator_errors: list[str]
    validator_warnings: list[str]

    history: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    llm_latency_s: float
    n_tokens: int

    done: bool
    last_error: str | None
    last_error_type: str | None


class ReActBaseAgent:
    """Iterative direct-XML refinement agent (no Python intermediate)."""

    def __init__(
        self,
        model,
        tokenizer,
        model_config: dict,
        agent_cfg: dict | None = None,
        catalog: SkillsCatalog | None = None,
        safety_rules: SafetyRulesLoader | None = None,
        constraint=None,
        eval_cfg: dict | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = dict(model_config or {})
        self.agent_cfg = dict(agent_cfg or {})
        self.catalog = catalog or SkillsCatalog()
        self.safety_rules = safety_rules
        self.constraint = constraint
        self.eval_cfg = dict(eval_cfg or {})

        self.temperature = float(self.agent_cfg.get("temperature", 0.3))
        self.top_p = float(self.agent_cfg.get("top_p", 0.9))
        self.max_new_tokens = int(self.agent_cfg.get("max_new_tokens", 4096))
        self.max_iterations = int(self.agent_cfg.get("max_iterations", 3))
        self.target_score = float(self.agent_cfg.get("target_score", 1.0))
        self.use_langgraph = bool(self.agent_cfg.get("use_langgraph", True))
        self.use_constraint = bool(self.agent_cfg.get("use_constraint", True))
        self.inner_prompt_mode = str(self.agent_cfg.get("inner_prompt_mode", "chain_of_thought"))

        # Inject inner_prompt_mode into model_config so prompt_builder's
        # react_base_agent branch can read it.
        self.model_config["inner_prompt_mode"] = self.inner_prompt_mode

    def _llm_generate(self, prompt) -> tuple[str, float, int, str | None]:
        """Single completion with optional grammar constraint.

        Returns (text, latency_s, n_tokens, error_message). Mirrors
        `src.eval.benchmark._generate_xml` constraint-failure handling so
        grammar tokenizer/vocab mismatches don't kill the run.
        """
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

        active_constraint = (
            self.constraint
            if self.use_constraint and self.constraint is not None and self.constraint.is_active()
            else None
        )
        if active_constraint is not None:
            from src.eval.constrained import apply_to_generate_kwargs

            active_constraint.fresh_processor()
            apply_to_generate_kwargs(gen_kwargs, active_constraint)

        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
        except AssertionError as e:
            logger.warning("Grammar backend assertion failure: %s", e)
            return "", time.perf_counter() - t0, 0, f"grammar_assertion: {e}"
        except ValueError as e:
            msg = str(e)
            if "stacks are empty" in msg or "not accepted by the grammar" in msg:
                logger.warning("Grammar backend rejected a token: %s", msg)
                return "", time.perf_counter() - t0, 0, f"grammar_rejected: {msg[:160]}"
            raise
        latency = time.perf_counter() - t0

        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        n_tokens = int(new_tokens.shape[0])
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        if active_constraint is not None and active_constraint.post_process is not None:
            raw = active_constraint.post_process(raw)

        return raw, latency, n_tokens, None

    def generate_xml(self, state: ReActBaseState) -> ReActBaseState:
        from src.data.prompt_builder import build_prompt

        state["iteration"] = state.get("iteration", 0) + 1

        prompt = build_prompt(
            mode="react_base_agent",
            mission=state["mission"],
            model_config=self.model_config,
            catalog=self.catalog,
            safety_rules=self.safety_rules,
            history=state.get("history", []),
        )
        completion, latency, n_tokens, gen_error = self._llm_generate(prompt)
        state["completion"] = completion
        state["llm_latency_s"] = state.get("llm_latency_s", 0.0) + latency
        state["n_tokens"] = state.get("n_tokens", 0) + n_tokens

        xml = extract_xml(completion)
        state["xml"] = xml

        if gen_error:
            state["last_error"] = gen_error
            state["last_error_type"] = (
                "GrammarAssertion" if "assertion" in gen_error else "GrammarRejected"
            )
        elif not xml:
            state["last_error"] = "LLM produced no <root> XML block"
            state["last_error_type"] = "NoXMLExtracted"
        else:
            state["last_error"] = None
            state["last_error_type"] = None

        state.setdefault("trace", []).append(
            {
                "step": "generate_xml",
                "iteration": state["iteration"],
                "completion_len": len(completion),
                "xml_found": bool(xml),
                "gen_error": gen_error,
            }
        )
        return state

    def validate(self, state: ReActBaseState) -> ReActBaseState:
        from src.eval.validate_bt import enrich_ports, validate_bt

        xml = state.get("xml", "")
        if not xml:
            state["validator_score"] = 0.0
            state["validator_valid"] = False
            state["validator_errors"] = ["No XML to validate"]
            state["validator_warnings"] = []
            state.setdefault("trace", []).append(
                {
                    "step": "validate",
                    "iteration": state["iteration"],
                    "score": 0.0,
                    "valid": False,
                    "skipped_reason": "no xml",
                }
            )
            return state

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

        state.setdefault("trace", []).append(
            {
                "step": "validate",
                "iteration": state["iteration"],
                "score": result.score,
                "valid": result.valid,
                "n_errors": len(result.errors),
                "n_warnings": len(result.warnings),
            }
        )
        return state

    def reflect(self, state: ReActBaseState) -> ReActBaseState:
        score = state.get("validator_score", 0.0)
        iteration = state.get("iteration", 1)
        max_iter = state.get("max_iterations", self.max_iterations)
        target = state.get("target_score", self.target_score)

        state.setdefault("history", []).append(
            {
                "iteration": iteration,
                "xml": state.get("xml", ""),
                "score": score,
                "errors": list(state.get("validator_errors", [])),
                "warnings": list(state.get("validator_warnings", [])),
                "error": state.get("last_error"),
            }
        )

        reached_target = score >= target and state.get("last_error") is None
        exhausted = iteration >= max_iter
        state["done"] = reached_target or exhausted

        state.setdefault("trace", []).append(
            {
                "step": "reflect",
                "iteration": iteration,
                "done": state["done"],
                "reason": (
                    "target_reached" if reached_target
                    else "max_iterations" if exhausted
                    else "retry"
                ),
            }
        )
        return state

    def _build_langgraph(self):
        from langgraph.graph import END, StateGraph

        graph = StateGraph(ReActBaseState)
        graph.add_node("generate_xml", self.generate_xml)
        graph.add_node("validate", self.validate)
        graph.add_node("reflect", self.reflect)

        graph.set_entry_point("generate_xml")
        graph.add_edge("generate_xml", "validate")
        graph.add_edge("validate", "reflect")

        def _route(state: ReActBaseState) -> str:
            return END if state.get("done") else "generate_xml"

        graph.add_conditional_edges(
            "reflect", _route, {"generate_xml": "generate_xml", END: END}
        )
        return graph.compile()

    def _run_plain_loop(self, state: ReActBaseState) -> ReActBaseState:
        while True:
            self.generate_xml(state)
            self.validate(state)
            self.reflect(state)
            if state.get("done"):
                return state

    def run(self, mission: str) -> AgentResult:
        t_start = time.perf_counter()
        state: ReActBaseState = {
            "mission": mission,
            "max_iterations": self.max_iterations,
            "target_score": self.target_score,
            "iteration": 0,
            "history": [],
            "trace": [],
            "llm_latency_s": 0.0,
            "n_tokens": 0,
        }

        if self.use_langgraph:
            try:
                graph = self._build_langgraph()
                state = graph.invoke(
                    state, config={"recursion_limit": 3 * self.max_iterations + 6}
                )
            except ImportError:
                logger.info("LangGraph unavailable — falling back to plain loop")
                state = self._run_plain_loop(state)
        else:
            state = self._run_plain_loop(state)

        total_latency = time.perf_counter() - t_start

        return AgentResult(
            mission=mission,
            xml=state.get("xml", ""),
            success=bool(state.get("validator_valid") and state.get("xml")),
            code="",  # No intermediate Python in this agent
            total_latency_s=total_latency,
            llm_latency_s=state.get("llm_latency_s", 0.0),
            sandbox_latency_s=0.0,
            n_tokens=state.get("n_tokens", 0),
            n_iterations=state.get("iteration", 0),
            execution_results=[],
            error_type=state.get("last_error_type"),
            error_message=state.get("last_error"),
            trace=state.get("trace", []),
        )
