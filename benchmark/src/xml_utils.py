from __future__ import annotations

from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET


def extract_root_xml(text: str) -> Optional[str]:
    s = text or ""
    # Prefer the last complete <root ...> ... </root> block in case other text precedes it.
    end = s.rfind("</root>")
    if end < 0:
        return None
    end += len("</root>")
    start = s.rfind("<root", 0, end)
    if start < 0:
        return None
    return s[start:end].strip()


def pretty_print_xml(xml_text: str, *, indent: str = "  ", ensure_trailing_newline: bool = True) -> str:
    root = ET.fromstring(xml_text)
    # Python 3.9+ supports ET.indent
    ET.indent(root, space=indent)  # type: ignore[attr-defined]
    out = ET.tostring(root, encoding="unicode")
    if ensure_trailing_newline and not out.endswith("\n"):
        out += "\n"
    return out


def pretty_print_xml_file(path: str | Path, *, indent: str = "  ") -> str:
    p = Path(path).expanduser().resolve()
    return pretty_print_xml(p.read_text(encoding="utf-8"), indent=indent)

