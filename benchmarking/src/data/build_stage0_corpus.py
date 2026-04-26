#!/usr/bin/env python3
"""
Stage 0 corpus builder — continued pretraining on BT XML grammar.
================================================================

Reads BTs from one or more roots (default: nav4rails_repo/behavior_trees +
optional Nav2 / BehaviorTree.CPP examples), normalises them, and emits a
JSONL where each record is a self-contained training "document":

    {"text": "<root ...> ... </root>", "domain": "nav4rail" | "generic", ...}

Two contamination-mitigation modes (see PLAN.md §2.2):

  - mode=mask     : skill IDs replaced by typed placeholders
                    <Action ID="ACTION_*"/> / <Condition ID="COND_*"/>
                    Loses lexical content; teaches grammar and shape only.
                    Use when mining open-source corpora outside the catalogue.

  - mode=tag      : skill IDs preserved, prepended with a domain tag header
                    [domain=nav4rail] or [domain=generic]
                    Use for the internal nav4rail corpus where the vocabulary
                    is already in the catalogue.

  - mode=raw      : no transformation (debugging only).

The script is idempotent: re-running with the same inputs produces the same
JSONL byte-for-byte.

Usage:
  python -m src.data.build_stage0_corpus \\
      --root /home/mlatoundji/studies/dev/nav4rails/nav4rails_repo/behavior_trees \\
      --domain nav4rail --mode tag \\
      --out data/stage0_corpus.jsonl

  # Multi-source: mix internal (tag) + generic mining (mask) into one file
  python -m src.data.build_stage0_corpus \\
      --root /home/.../nav4rails_repo/behavior_trees --domain nav4rail --mode tag \\
      --extra-root /home/.../nav2_examples --extra-domain generic --extra-mode mask \\
      --out data/stage0_corpus.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


_ID_ATTR_RE = re.compile(r'\bID="([^"]+)"')


def _mask_skill_ids(xml_text: str) -> str:
    """Replace ID="SkillName" by ACTION_* / COND_* / SUB_* placeholders.

    Distinguishes <Action>, <Condition>, <SubTree>/<SubTreePlus> by the
    surrounding tag. The replacement is per-occurrence so duplicate skill
    references are coalesced (an Action used twice gets the same placeholder
    only if it's literally the same ID — preserves intra-tree consistency).
    """
    # First-pass: collect (tag, id) pairs in document order to assign indexes
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return xml_text  # malformed — leave unchanged for downstream filtering

    seen: dict[tuple[str, str], str] = {}
    counters = {"action": 0, "condition": 0, "subtree": 0}

    def assign(tag: str, sid: str) -> str:
        key = (tag, sid)
        if key in seen:
            return seen[key]
        if tag == "Action":
            counters["action"] += 1
            ph = f"ACTION_{counters['action']}"
        elif tag == "Condition":
            counters["condition"] += 1
            ph = f"COND_{counters['condition']}"
        elif tag in {"SubTree", "SubTreePlus"}:
            counters["subtree"] += 1
            ph = f"SUB_{counters['subtree']}"
        else:
            ph = sid  # leave unknown tags alone
        seen[key] = ph
        return ph

    # Walk and rewrite IDs on Action/Condition/SubTree nodes
    for elem in root.iter():
        if elem.tag in {"Action", "Condition", "SubTree", "SubTreePlus"}:
            sid = elem.attrib.get("ID")
            if sid:
                elem.attrib["ID"] = assign(elem.tag, sid)
            # Mask the human-readable name too — leaks the skill identity
            elem.attrib.pop("name", None)

    return ET.tostring(root, encoding="unicode")


def _add_domain_tag(xml_text: str, domain: str) -> str:
    """Prepend a single-line domain tag as an XML comment (preserves parsing)."""
    tag_line = f"<!-- [domain={domain}] -->"
    return f"{tag_line}\n{xml_text.strip()}"


def _normalise_xml(xml_text: str) -> str:
    """Light normalisation: strip BOM, collapse trailing whitespace, drop XML decl."""
    text = xml_text.lstrip("﻿").strip()
    if text.startswith("<?xml"):
        end = text.find("?>")
        if end != -1:
            text = text[end + 2 :].lstrip()
    return text


def _iter_bt_files(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    for ext in exts:
        yield from sorted(root.rglob(f"*{ext}"))


def _process(
    root: Path,
    domain: str,
    mode: str,
    exts: Iterable[str],
) -> Iterable[dict]:
    seen_hashes: set[str] = set()
    for path in _iter_bt_files(root, exts):
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"WARN: read failed {path}: {e}", file=sys.stderr)
            continue

        # Quick parse check — skip malformed
        try:
            ET.fromstring(raw)
        except ET.ParseError as e:
            print(f"WARN: parse failed {path}: {e}", file=sys.stderr)
            continue

        text = _normalise_xml(raw)

        if mode == "mask":
            text = _mask_skill_ids(text)
        elif mode == "tag":
            text = _add_domain_tag(text, domain)
        elif mode == "raw":
            pass
        else:
            raise ValueError(f"Unknown mode: {mode}")

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)

        yield {
            "text": text,
            "domain": domain,
            "mode": mode,
            "source_path": str(path),
            "sha256": digest,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Primary corpus root")
    parser.add_argument("--domain", default="nav4rail", help="Domain label for primary root")
    parser.add_argument("--mode", choices=["mask", "tag", "raw"], default="tag")
    parser.add_argument("--extra-root", action="append", default=[],
                        help="Additional corpus root (can be passed multiple times)")
    parser.add_argument("--extra-domain", action="append", default=[],
                        help="Domain label for the i-th extra root")
    parser.add_argument("--extra-mode", action="append", default=[],
                        help="Mode for the i-th extra root")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--ext", nargs="+", default=[".behaviortreeschema", ".xml"],
                        help="File extensions to scan")
    args = parser.parse_args()

    if not (len(args.extra_root) == len(args.extra_domain) == len(args.extra_mode)):
        parser.error("--extra-root / --extra-domain / --extra-mode must have equal length")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    with out_path.open("w", encoding="utf-8") as fp:
        for record in _process(Path(args.root), args.domain, args.mode, args.ext):
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_total += 1
        for extra_root, extra_domain, extra_mode in zip(
            args.extra_root, args.extra_domain, args.extra_mode
        ):
            for record in _process(Path(extra_root), extra_domain, extra_mode, args.ext):
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_total += 1

    print(f"Wrote {n_total} unique BT documents to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
