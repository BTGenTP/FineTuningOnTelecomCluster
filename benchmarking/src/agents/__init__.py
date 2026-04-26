"""
NAV4RAIL code-as-reasoning + direct-XML agents.

Public API:
    from src.agents import AgentResult, PoTAgent, ReActPoTAgent, ReActBaseAgent, run_sandboxed
"""

from src.agents.base_agent import AgentResult
from src.agents.sandbox import ExecutionResult, SandboxError, run_sandboxed

__all__ = [
    "AgentResult",
    "ExecutionResult",
    "SandboxError",
    "run_sandboxed",
]


def __getattr__(name):
    # Lazy imports so missing optional deps (e.g. langgraph) don't block
    # the rest of the package.
    if name == "PoTAgent":
        from src.agents.pot_agent import PoTAgent

        return PoTAgent
    if name == "ReActPoTAgent":
        from src.agents.react_pot_agent import ReActPoTAgent

        return ReActPoTAgent
    if name == "ReActBaseAgent":
        from src.agents.react_base_agent import ReActBaseAgent

        return ReActBaseAgent
    raise AttributeError(f"module 'src.agents' has no attribute {name!r}")
