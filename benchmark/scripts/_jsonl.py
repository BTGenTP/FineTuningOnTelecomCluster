from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return p

