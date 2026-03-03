"""
Compatibility shim for cluster nodes where Python was built without a working
libbz2 runtime.

This file intentionally shadows the standard-library ``bz2`` module when
``finetune_lora_xml.py`` is executed from this directory. It first tries to
load the real stdlib implementation. If that fails because ``_bz2`` cannot be
loaded, it exposes placeholder symbols so optional imports from libraries like
``datasets`` and ``trl`` keep working.

Actual .bz2 compression/decompression remains unavailable in fallback mode.
"""

from __future__ import annotations

import importlib.util
import sysconfig
from pathlib import Path


def _load_stdlib_bz2():
    stdlib_bz2 = Path(sysconfig.get_path("stdlib")) / "bz2.py"
    if not stdlib_bz2.is_file():
        return None

    spec = importlib.util.spec_from_file_location("_stdlib_bz2", stdlib_bz2)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    _STDLIB_BZ2 = _load_stdlib_bz2()
except ImportError as exc:
    if exc.name != "_bz2" and "libbz2" not in str(exc):
        raise
    _STDLIB_BZ2 = None


if _STDLIB_BZ2 is not None:
    __all__ = getattr(_STDLIB_BZ2, "__all__", [])
    __doc__ = _STDLIB_BZ2.__doc__

    for _name in dir(_STDLIB_BZ2):
        if _name.startswith("__") and _name not in {"__all__", "__doc__"}:
            continue
        globals()[_name] = getattr(_STDLIB_BZ2, _name)
else:
    _ERROR = (
        "bz2 support is unavailable on this node because the Python runtime "
        "cannot load libbz2. This shim only keeps optional imports working. "
        "Reading or writing .bz2 files is not supported here."
    )

    class _UnavailableBz2:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(_ERROR)


    class BZ2File(_UnavailableBz2):
        pass


    class BZ2Compressor(_UnavailableBz2):
        pass


    class BZ2Decompressor(_UnavailableBz2):
        pass


    def open(*args, **kwargs):
        raise RuntimeError(_ERROR)


    def compress(data, compresslevel=9):
        raise RuntimeError(_ERROR)


    def decompress(data):
        raise RuntimeError(_ERROR)


    __all__ = [
        "BZ2File",
        "BZ2Compressor",
        "BZ2Decompressor",
        "open",
        "compress",
        "decompress",
    ]
