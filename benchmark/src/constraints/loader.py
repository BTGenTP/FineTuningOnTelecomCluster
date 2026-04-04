from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


@dataclass(frozen=True, slots=True)
class ConstraintsBundle:
    constraints_dir: Path
    patterns: Dict[str, Any]
    fsm: Dict[str, Any]
    dataflow: Dict[str, Any]
    recovery: Dict[str, Any]
    enums: Dict[str, Any]
    xml_format: Dict[str, Any]


def default_constraints_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "constraints"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def load_constraints(constraints_dir: Optional[str | Path] = None) -> ConstraintsBundle:
    base = Path(constraints_dir).expanduser().resolve() if constraints_dir else default_constraints_dir().resolve()
    return ConstraintsBundle(
        constraints_dir=base,
        patterns=_load_yaml(base / "patterns.yaml"),
        fsm=_load_yaml(base / "fsm.yaml"),
        dataflow=_load_yaml(base / "dataflow.yaml"),
        recovery=_load_yaml(base / "recovery.yaml"),
        enums=_load_yaml(base / "enums.yaml"),
        xml_format=_load_yaml(base / "xml_format.yaml"),
    )

