"""
In-process sandbox for Program-of-Thoughts / ReAct agents.
==========================================================

Executes untrusted Python code in a restricted namespace whose only import
surface is `nav4rail_builder` (which re-exports `MissionBuilder` and the
builder exceptions). Enforcement happens in two stages:

  1. Static AST allowlist  — rejects imports, attribute access to dunder
     names, and calls to dangerous built-ins before execution.
  2. Restricted globals    — no `open`, `os`, `subprocess`, `eval`, ...

The sandbox is NOT a security boundary against a determined adversary —
bypasses are documented in the literature — but it is strong enough to
catch LLM-generated code that accidentally tries to import std-lib modules,
read files, or escape the namespace. For a benchmark setting where the
author of the code is a local LLM and the operator controls the process,
this is the pragmatic middle-ground between speed and safety.

Usage:
    from src.agents.sandbox import run_sandboxed
    result = run_sandboxed(code, timeout_s=10.0)
    if result.success:
        print(result.stdout)      # typically the XML
"""

from __future__ import annotations

import ast
import contextlib
import io
import signal
import time
import traceback
from dataclasses import dataclass, field
from typing import Any


# ── Public types ─────────────────────────────────────────────────────────────


class SandboxError(Exception):
    """Raised when the code violates the sandbox allowlist before execution."""


@dataclass
class ExecutionResult:
    """Outcome of a sandboxed execution."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    error_type: str | None = None  # Python exception class name (on failure)
    error_message: str | None = None
    traceback: str | None = None
    exec_time_s: float = 0.0
    local_vars: dict[str, Any] = field(default_factory=dict)

    @property
    def error_summary(self) -> str | None:
        """Compact one-line error description for feeding back to the LLM."""
        if self.success:
            return None
        if self.error_type and self.error_message:
            return f"{self.error_type}: {self.error_message}"
        return self.error_message or "Unknown error"


# ── AST allowlist ────────────────────────────────────────────────────────────

# Allowed built-ins exposed inside the sandbox (minus __import__, which is
# constructed per-call in _build_sandbox_globals so it can see the allowlist).
_ALLOWED_BUILTINS: dict[str, Any] = {
    "print": print,
    "range": range,
    "len": len,
    "enumerate": enumerate,
    "zip": zip,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "True": True,
    "False": False,
    "None": None,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "sorted": sorted,
    "reversed": reversed,
    "any": any,
    "all": all,
    "map": map,
    "filter": filter,
    "round": round,
    "isinstance": isinstance,
    "hasattr": hasattr,
    "getattr": getattr,
    # Exceptions useful for the agent to inspect errors
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "AttributeError": AttributeError,
    "RuntimeError": RuntimeError,
    "ImportError": ImportError,
}

# Imports that may appear in generated code — only `nav4rail_builder`.
_ALLOWED_IMPORT_MODULES = frozenset({"nav4rail_builder"})


def _check_node(node: ast.AST) -> None:
    """Recursively validate a single AST node. Raises SandboxError on violation."""
    # Block any non-allowlisted import
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        names = (
            [n.name for n in node.names]
            if isinstance(node, ast.Import)
            else [node.module]
        )
        for name in names:
            if name is None:
                continue
            top = name.split(".")[0]
            if top not in _ALLOWED_IMPORT_MODULES:
                raise SandboxError(
                    f"Import of '{name}' is not allowed. Only these modules are "
                    f"importable: {sorted(_ALLOWED_IMPORT_MODULES)}"
                )

    # Block access to dunder attributes (the usual exec() escape route)
    if isinstance(node, ast.Attribute):
        if node.attr.startswith("__") and node.attr.endswith("__"):
            # Tolerate __init__ and __class__-style expressions? No — reject.
            raise SandboxError(
                f"Access to dunder attribute '{node.attr}' is not allowed"
            )

    # Block lookup of __builtins__ / globals() / locals() etc.
    if isinstance(node, ast.Name):
        if node.id in {
            "__builtins__",
            "__import__",
            "exec",
            "eval",
            "compile",
            "globals",
            "locals",
            "vars",
            "open",
            "input",
            "exit",
            "quit",
            "help",
        }:
            raise SandboxError(
                f"Reference to forbidden name '{node.id}'"
            )

    # Block starargs to a call — too easy to smuggle things through
    # (disabled: used legitimately by `b.sequence(*nodes)` etc.)


def _audit_ast(tree: ast.AST) -> None:
    """Walk the AST and reject anything that violates the sandbox policy."""
    for node in ast.walk(tree):
        _check_node(node)


# ── Optional timeout (best-effort; POSIX only) ───────────────────────────────


@contextlib.contextmanager
def _timeout(seconds: float):
    """
    Enforce a wall-clock timeout via SIGALRM. On Windows (no SIGALRM) the
    context manager is a no-op — use an outer subprocess timeout if strict
    isolation is required.
    """
    use_alarm = hasattr(signal, "SIGALRM")
    if not use_alarm or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"Sandbox timeout after {seconds:g}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


# ── Public API ────────────────────────────────────────────────────────────────


def _build_sandbox_globals(catalog=None) -> dict[str, Any]:
    """
    Build the restricted globals dict.

    Exposes `nav4rail_builder` as a synthetic module whose attributes are the
    builder classes, bound to the supplied catalog (so the LLM doesn't need
    to instantiate SkillsCatalog).
    """
    import types

    from src.builder import (
        BuilderError,
        MissingRequiredSkillError,
        MissionBuilder,
        PortError,
        StructuralError,
        UnknownSkillError,
    )

    if catalog is not None:
        # Partial-apply the catalog so `MissionBuilder()` uses the shared one.
        class _BoundMissionBuilder(MissionBuilder):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("catalog", catalog)
                super().__init__(*args, **kwargs)

        bound_cls = _BoundMissionBuilder
    else:
        bound_cls = MissionBuilder

    module = types.ModuleType("nav4rail_builder")
    module.MissionBuilder = bound_cls
    module.BuilderError = BuilderError
    module.UnknownSkillError = UnknownSkillError
    module.PortError = PortError
    module.StructuralError = StructuralError
    module.MissingRequiredSkillError = MissingRequiredSkillError

    # The AST audit already rejects any import other than `nav4rail_builder`.
    # This custom `__import__` is a second line of defense AND the plumbing
    # that makes `from nav4rail_builder import ...` work when __builtins__
    # is a dict (Python needs a resolver to find the module).
    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name not in _ALLOWED_IMPORT_MODULES:
            raise ImportError(
                f"Import of '{name}' is not allowed in the NAV4RAIL sandbox. "
                f"Only importable module: 'nav4rail_builder'."
            )
        return module

    builtins = dict(_ALLOWED_BUILTINS)
    builtins["__import__"] = _restricted_import

    return {
        "__builtins__": builtins,
        "nav4rail_builder": module,
    }


def run_sandboxed(
    code: str,
    catalog=None,
    timeout_s: float = 10.0,
) -> ExecutionResult:
    """
    Execute `code` in the sandbox and return a structured result.

    Pre-execution: AST audit. Any violation → ExecutionResult(success=False)
    with error_type="SandboxError".

    Execution: exec() in a restricted namespace, stdout/stderr captured.
    Any Python exception → ExecutionResult(success=False) with the traceback.

    The sandbox captures `local_vars` after execution so the caller can pull
    specific variables (e.g. `builder`, `xml`) out of the namespace.
    """
    t0 = time.perf_counter()

    # Stage 1 — static check
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return ExecutionResult(
            success=False,
            error_type="SyntaxError",
            error_message=str(e),
            traceback=traceback.format_exc(),
            exec_time_s=time.perf_counter() - t0,
        )

    try:
        _audit_ast(tree)
    except SandboxError as e:
        return ExecutionResult(
            success=False,
            error_type="SandboxError",
            error_message=str(e),
            exec_time_s=time.perf_counter() - t0,
        )

    # Stage 2 — sandboxed exec
    g = _build_sandbox_globals(catalog=catalog)
    l: dict[str, Any] = {}
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            with _timeout(timeout_s):
                exec(compile(tree, "<sandbox>", "exec"), g, l)
    except BaseException as e:  # noqa: BLE001 — we really do want everything
        return ExecutionResult(
            success=False,
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            error_type=type(e).__name__,
            error_message=str(e),
            traceback=traceback.format_exc(),
            exec_time_s=time.perf_counter() - t0,
            local_vars={k: v for k, v in l.items() if not k.startswith("_")},
        )

    return ExecutionResult(
        success=True,
        stdout=stdout_buf.getvalue(),
        stderr=stderr_buf.getvalue(),
        exec_time_s=time.perf_counter() - t0,
        local_vars={k: v for k, v in l.items() if not k.startswith("_")},
    )


def extract_xml_from_result(result: ExecutionResult) -> str | None:
    """
    Extract the generated XML from an execution result.

    Strategy (in order):
      1. If `xml` is in `local_vars` and is a string — use it.
      2. Otherwise, search stdout for a <root ...>...</root> block.
      3. Otherwise, return None.
    """
    import re

    candidate = result.local_vars.get("xml")
    if isinstance(candidate, str) and "<root" in candidate:
        return candidate

    stdout = result.stdout
    match = re.search(r"<root[\s>].*?</root>", stdout, re.DOTALL)
    if match:
        return match.group(0)

    if "<root" in stdout:
        idx = stdout.index("<root")
        return stdout[idx:]

    return None
