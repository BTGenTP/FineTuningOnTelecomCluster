#!/usr/bin/env python3
"""
Inventaire d'un corpus de Behavior Trees (.behaviortreeschema / .xml) BTCPP-format.

Produit:
  - <out>.jsonl   : une ligne par BT (path, root attrs, subtrees, skills, ports, depth, branching, parse status)
  - <out>.md      : rapport human-readable groupé par dossier
  - <out>.summary.json : agrégats globaux (skills uniques, frequencies, depth distribution, etc.)

Usage:
  python -m scripts.inventory_bt_corpus --root /path/to/nav4rails_repo/behavior_trees \\
      --out runs/local/bt_inventory_$(date +%Y%m%d)
  python -m scripts.inventory_bt_corpus --root /path --catalog data/skills_catalog.yaml --out runs/local/bt_inventory
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

try:
    import yaml
except ImportError:
    yaml = None


CONTROL_TAGS = {
    "Sequence", "ReactiveSequence", "SequenceStar",
    "Fallback", "ReactiveFallback", "FallbackStar",
    "Parallel", "ParallelAll",
    "Repeat", "RetryUntilSuccessful", "Inverter", "ForceSuccess", "ForceFailure",
    "Switch", "WhileDoElse",
}
LEAF_TAGS = {"Action", "Condition", "Decorator", "SubTree", "SubTreePlus"}
SUBTREE_TAGS = {"SubTree", "SubTreePlus"}


@dataclass
class BTInventoryEntry:
    path: str
    relative_path: str
    file_size_bytes: int
    parse_ok: bool
    parse_error: str | None
    root_main_tree: str | None
    declared_subtrees: list[str]
    referenced_subtrees: list[str]
    skill_ids: list[str]                # all <Action ID> + <Condition ID>
    skill_id_counts: dict[str, int]
    control_node_counts: dict[str, int]
    n_actions: int
    n_conditions: int
    n_subtrees: int
    n_subtree_plus: int
    max_depth: int
    mean_branching: float
    has_autoremap: bool
    has_repeat_loop: bool
    n_top_level_trees: int
    ports_seen: dict[str, list[str]]    # skill_id -> [port_name, ...] (deduplicated)
    blackboard_refs: list[str]          # unique {var} placeholders


def _walk(elem: ET.Element, depth: int = 0):
    yield elem, depth
    for child in elem:
        yield from _walk(child, depth + 1)


def _classify(tag: str) -> str:
    if tag in CONTROL_TAGS:
        return "control"
    if tag in LEAF_TAGS:
        return "leaf"
    if tag in {"BehaviorTree", "root"}:
        return "structural"
    return "other"


def analyze_bt(path: Path, root_dir: Path) -> BTInventoryEntry:
    raw = path.read_bytes()
    entry = BTInventoryEntry(
        path=str(path),
        relative_path=str(path.relative_to(root_dir)),
        file_size_bytes=len(raw),
        parse_ok=False,
        parse_error=None,
        root_main_tree=None,
        declared_subtrees=[],
        referenced_subtrees=[],
        skill_ids=[],
        skill_id_counts={},
        control_node_counts={},
        n_actions=0,
        n_conditions=0,
        n_subtrees=0,
        n_subtree_plus=0,
        max_depth=0,
        mean_branching=0.0,
        has_autoremap=False,
        has_repeat_loop=False,
        n_top_level_trees=0,
        ports_seen={},
        blackboard_refs=[],
    )
    try:
        root = ET.fromstring(raw)
        entry.parse_ok = True
    except ET.ParseError as exc:
        entry.parse_error = f"{type(exc).__name__}: {exc}"
        return entry

    if root.tag == "root":
        entry.root_main_tree = root.attrib.get("main_tree_to_execute")
    behavior_trees = root.findall("BehaviorTree") if root.tag == "root" else [root]
    entry.n_top_level_trees = len(behavior_trees)
    entry.declared_subtrees = [bt.attrib.get("ID", "") for bt in behavior_trees if bt.attrib.get("ID")]

    skill_counter: Counter[str] = Counter()
    control_counter: Counter[str] = Counter()
    referenced_subtrees: set[str] = set()
    ports_seen: dict[str, set[str]] = defaultdict(set)
    blackboard_refs: set[str] = set()
    branching_samples: list[int] = []

    for bt in behavior_trees:
        for elem, depth in _walk(bt):
            kind = _classify(elem.tag)
            entry.max_depth = max(entry.max_depth, depth)
            n_children = len(list(elem))
            if kind == "control" and n_children > 0:
                branching_samples.append(n_children)
            if kind == "control":
                control_counter[elem.tag] += 1
                if elem.tag in {"Repeat", "RetryUntilSuccessful"}:
                    entry.has_repeat_loop = True
            elif elem.tag in {"Action", "Condition"}:
                sid = elem.attrib.get("ID", "")
                if sid:
                    skill_counter[sid] += 1
                    for k, v in elem.attrib.items():
                        if k in {"ID", "name"}:
                            continue
                        ports_seen[sid].add(k)
                        if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                            blackboard_refs.add(v[1:-1])
                if elem.tag == "Action":
                    entry.n_actions += 1
                else:
                    entry.n_conditions += 1
            elif elem.tag in SUBTREE_TAGS:
                if elem.tag == "SubTree":
                    entry.n_subtrees += 1
                else:
                    entry.n_subtree_plus += 1
                ref = elem.attrib.get("ID")
                if ref:
                    referenced_subtrees.add(ref)
                if elem.attrib.get("__autoremap", "").lower() == "true":
                    entry.has_autoremap = True

    entry.skill_id_counts = dict(skill_counter)
    entry.skill_ids = sorted(skill_counter.keys())
    entry.control_node_counts = dict(control_counter)
    entry.referenced_subtrees = sorted(referenced_subtrees)
    entry.mean_branching = round(sum(branching_samples) / len(branching_samples), 2) if branching_samples else 0.0
    entry.ports_seen = {k: sorted(v) for k, v in ports_seen.items()}
    entry.blackboard_refs = sorted(blackboard_refs)
    return entry


def load_catalog_skill_ids(catalog_path: Path) -> set[str]:
    if yaml is None:
        print("WARN: pyyaml not installed, skipping catalog comparison", file=sys.stderr)
        return set()
    with open(catalog_path) as f:
        data = yaml.safe_load(f)
    skill_ids: set[str] = set()
    for fam in data.get("families", {}).values():
        skills = fam.get("skills", {})
        if isinstance(skills, dict):
            skill_ids.update(skills.keys())
        elif isinstance(skills, list):
            for skill in skills:
                if isinstance(skill, dict) and "id" in skill:
                    skill_ids.add(skill["id"])
    return skill_ids


def build_summary(entries: list[BTInventoryEntry], catalog_skills: set[str]) -> dict[str, Any]:
    parsed = [e for e in entries if e.parse_ok]
    unparsed = [e for e in entries if not e.parse_ok]
    all_skills: Counter[str] = Counter()
    all_controls: Counter[str] = Counter()
    depth_distribution: list[int] = []
    branching_distribution: list[float] = []
    autoremap_count = 0
    repeat_count = 0
    for e in parsed:
        all_skills.update(e.skill_id_counts)
        all_controls.update(e.control_node_counts)
        depth_distribution.append(e.max_depth)
        branching_distribution.append(e.mean_branching)
        if e.has_autoremap:
            autoremap_count += 1
        if e.has_repeat_loop:
            repeat_count += 1

    skills_in_corpus = set(all_skills.keys())
    summary = {
        "n_files_total": len(entries),
        "n_files_parsed": len(parsed),
        "n_files_failed": len(unparsed),
        "n_files_with_autoremap": autoremap_count,
        "n_files_with_repeat_loop": repeat_count,
        "depth_min": min(depth_distribution) if depth_distribution else 0,
        "depth_max": max(depth_distribution) if depth_distribution else 0,
        "depth_mean": round(sum(depth_distribution) / len(depth_distribution), 2) if depth_distribution else 0.0,
        "branching_mean": round(sum(branching_distribution) / len(branching_distribution), 2) if branching_distribution else 0.0,
        "unique_skills": sorted(skills_in_corpus),
        "skill_frequency": dict(all_skills.most_common()),
        "control_node_frequency": dict(all_controls.most_common()),
    }
    if catalog_skills:
        summary["skills_in_catalog_used"] = sorted(skills_in_corpus & catalog_skills)
        summary["skills_in_corpus_not_in_catalog"] = sorted(skills_in_corpus - catalog_skills)
        summary["skills_in_catalog_unused"] = sorted(catalog_skills - skills_in_corpus)
        summary["catalog_coverage_pct"] = round(
            100 * len(skills_in_corpus & catalog_skills) / max(1, len(catalog_skills)), 1
        )
    return summary


def write_markdown_report(entries: list[BTInventoryEntry], summary: dict[str, Any], out_md: Path) -> None:
    lines: list[str] = []
    lines.append("# BT Corpus Inventory")
    lines.append("")
    lines.append(f"- Total files scanned: **{summary['n_files_total']}**")
    lines.append(f"- Parsed OK: **{summary['n_files_parsed']}** / Failed: {summary['n_files_failed']}")
    lines.append(f"- With autoremap=true: {summary['n_files_with_autoremap']}")
    lines.append(f"- With Repeat/Retry loop: {summary['n_files_with_repeat_loop']}")
    lines.append(f"- Depth: min={summary['depth_min']} mean={summary['depth_mean']} max={summary['depth_max']}")
    lines.append(f"- Mean branching: {summary['branching_mean']}")
    lines.append("")
    if "catalog_coverage_pct" in summary:
        lines.append(f"## Catalogue coverage: {summary['catalog_coverage_pct']}%")
        lines.append("")
        lines.append(f"- Skills used (in catalog): {len(summary['skills_in_catalog_used'])}")
        lines.append(f"- Skills NOT in catalog (potential contamination): {len(summary['skills_in_corpus_not_in_catalog'])}")
        if summary["skills_in_corpus_not_in_catalog"]:
            lines.append("  - " + ", ".join(summary["skills_in_corpus_not_in_catalog"]))
        lines.append(f"- Catalog skills unused: {len(summary['skills_in_catalog_unused'])}")
        if summary["skills_in_catalog_unused"]:
            lines.append("  - " + ", ".join(summary["skills_in_catalog_unused"]))
        lines.append("")
    lines.append("## Control node frequency")
    for tag, count in summary["control_node_frequency"].items():
        lines.append(f"- `{tag}`: {count}")
    lines.append("")
    lines.append("## Top skills (by frequency)")
    for sid, count in list(summary["skill_frequency"].items())[:30]:
        lines.append(f"- `{sid}`: {count}")
    lines.append("")
    lines.append("## Per-file detail")
    by_dir: dict[str, list[BTInventoryEntry]] = defaultdict(list)
    for e in entries:
        d = str(Path(e.relative_path).parent)
        by_dir[d].append(e)
    for d in sorted(by_dir.keys()):
        lines.append(f"### {d}/")
        for e in sorted(by_dir[d], key=lambda x: x.relative_path):
            status = "OK" if e.parse_ok else "FAIL"
            short = Path(e.relative_path).name
            lines.append(
                f"- [{status}] **{short}** — depth={e.max_depth}, branching={e.mean_branching}, "
                f"trees={e.n_top_level_trees}, actions={e.n_actions}, conditions={e.n_conditions}, "
                f"subtree_plus={e.n_subtree_plus}"
                f"{' (autoremap)' if e.has_autoremap else ''}"
                f"{' (loop)' if e.has_repeat_loop else ''}"
            )
            if not e.parse_ok:
                lines.append(f"    - error: `{e.parse_error}`")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Corpus root directory")
    parser.add_argument("--out", required=True, help="Output prefix (no extension); writes .jsonl, .md, .summary.json")
    parser.add_argument("--catalog", default=None, help="Optional skills_catalog.yaml for coverage analysis")
    parser.add_argument(
        "--ext",
        nargs="+",
        default=[".behaviortreeschema", ".xml"],
        help="File extensions to consider (default: .behaviortreeschema .xml)",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        return 2
    out_prefix = Path(args.out).expanduser().resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    catalog_skills: set[str] = set()
    if args.catalog:
        catalog_skills = load_catalog_skill_ids(Path(args.catalog).expanduser().resolve())

    files: list[Path] = []
    for ext in args.ext:
        files.extend(root.rglob(f"*{ext}"))
    files = sorted(set(files))
    print(f"Scanning {len(files)} files under {root}", file=sys.stderr)

    entries: list[BTInventoryEntry] = [analyze_bt(f, root) for f in files]
    summary = build_summary(entries, catalog_skills)

    out_jsonl = out_prefix.with_suffix(".jsonl")
    out_md = out_prefix.with_suffix(".md")
    out_summary = out_prefix.with_suffix(".summary.json")

    with open(out_jsonl, "w", encoding="utf-8") as fp:
        for e in entries:
            fp.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(entries, summary, out_md)

    print(f"  jsonl    : {out_jsonl}")
    print(f"  markdown : {out_md}")
    print(f"  summary  : {out_summary}")
    print(f"Catalog coverage: {summary.get('catalog_coverage_pct', 'N/A')}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
