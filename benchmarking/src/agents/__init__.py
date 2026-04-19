"""
NAV4RAIL code-as-reasoning agents.

Public API:
    from src.agents import PoTAgent, ReActAgent, run_sandboxed
"""

from src.agents.sandbox import ExecutionResult, SandboxError, run_sandboxed

__all__ = [
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
    if name == "ReActAgent":
        from src.agents.react_agent import ReActAgent

        return ReActAgent
    raise AttributeError(f"module 'src.agents' has no attribute {name!r}")
