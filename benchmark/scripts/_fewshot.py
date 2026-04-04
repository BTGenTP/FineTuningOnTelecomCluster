from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from src.data.formatting import PromptExample


_WORD_RE = re.compile(r"[A-Za-z0-9_\\-]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text or "")}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass(frozen=True, slots=True)
class FewShotRow:
    mission: str
    xml: str
    metadata: dict

    def as_prompt_example(self) -> PromptExample:
        return PromptExample(mission=self.mission, xml=self.xml, metadata=dict(self.metadata or {}))


def select_top_k(mission: str, pool: Sequence[FewShotRow], k: int) -> list[PromptExample]:
    if k <= 0 or not pool:
        return []
    q = _tokens(mission)
    scored = [(jaccard(q, _tokens(row.mission)), row) for row in pool]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [row.as_prompt_example() for _score, row in scored[:k]]


def load_pool(rows: Iterable[dict]) -> list[FewShotRow]:
    out: list[FewShotRow] = []
    for r in rows:
        mission = str(r.get("mission", "")).strip()
        xml = str(r.get("xml", "")).strip()
        if not mission or not xml:
            continue
        out.append(FewShotRow(mission=mission, xml=xml, metadata=dict(r.get("metadata", {}))))
    return out

