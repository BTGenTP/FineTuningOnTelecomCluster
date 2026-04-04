from __future__ import annotations

from pathlib import Path


def benchmark_root() -> Path:
    # scripts/ is at BENCHMARK_ROOT/scripts
    return Path(__file__).resolve().parents[1]


def ensure_sys_path() -> Path:
    import sys

    root = benchmark_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root

