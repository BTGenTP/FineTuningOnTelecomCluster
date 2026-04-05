from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    # Allow direct execution: python src/uml/generate_nav4rail_catalog.py ...
    import sys

    this_file = Path(__file__).resolve()
    benchmark_root = this_file.parents[2]
    if str(benchmark_root) not in sys.path:
        sys.path.insert(0, str(benchmark_root))
    from src.uml.catalog_generator import build_catalog_from_uml, write_catalog  # type: ignore
    from src.uml.external_catalog_merge import merge_bt_navigator_port_semantics  # type: ignore
else:
    from .catalog_generator import build_catalog_from_uml, write_catalog
    from .external_catalog_merge import merge_bt_navigator_port_semantics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate NAV4RAIL skills catalog JSON from Papyrus UML.")
    p.add_argument("--nav4rails-repo", required=True, type=str, help="Path to nav4rails_repo root.")
    p.add_argument("--output", required=True, type=str, help="Output JSON path.")
    p.add_argument("--constraints-dir", default=None, type=str, help="Optional constraints directory for enums/tag overrides.")
    p.add_argument(
        "--bt-navigator-catalog",
        default=None,
        type=str,
        help="Optional BT_Navigator-style JSON (e.g. BT_Navigator/script/bt_nodes_catalog.json) to merge port_semantics into output.",
    )
    return p.parse_args()


def _looks_like_nav4rails_repo(repo: Path) -> bool:
    return (repo / "skills").is_dir() and (repo / "behavior_trees").is_dir()


def _autodetect_nav4rails_repo(start: Path, *, max_parents: int = 6) -> Path | None:
    """
    Best-effort helper for common path mistakes.
    Tries:
    - start/nav4rails_repo
    - any parent/nav4rails_repo (up to max_parents)
    """
    direct = start / "nav4rails_repo"
    if _looks_like_nav4rails_repo(direct):
        return direct
    for parent in [start, *list(start.parents)[:max_parents]]:
        candidate = parent / "nav4rails_repo"
        if _looks_like_nav4rails_repo(candidate):
            return candidate
    return None


def _fail_with_hint(repo: Path, *, skills: list[Path], schemas: list[Path]) -> None:
    resolved = repo.resolve()
    expected = [
        resolved / "skills" / "<skill_pkg>" / "models" / "skills" / "*.skills.uml",
        resolved / "behavior_trees" / "**" / "*.behaviortreeschema",
    ]
    msg = [
        "No UML/schemas found: check --nav4rails-repo path.",
        f"Resolved repo path: {resolved}",
        f"Found UML files: {len(skills)}",
        f"Found behaviortreeschema files: {len(schemas)}",
        "Expected patterns:",
        f"  - {expected[0]}",
        f"  - {expected[1]}",
        "",
        "Tip: from benchmark/ the workspace layout usually requires: --nav4rails-repo ../../../nav4rails_repo",
    ]
    raise SystemExit("\n".join(msg))


def main() -> int:
    args = parse_args()
    repo = Path(args.nav4rails_repo).expanduser().resolve()
    if not _looks_like_nav4rails_repo(repo):
        detected = _autodetect_nav4rails_repo(repo)
        if detected is not None:
            print(f"[warn] --nav4rails-repo does not look like nav4rails_repo: {repo}")
            print(f"[warn] Auto-detected nav4rails_repo at: {detected}")
            repo = detected
    skills = list(repo.glob("skills/*/models/skills/*.skills.uml"))
    schemas = list(repo.glob("behavior_trees/**/*.behaviortreeschema"))
    if not skills or not schemas:
        _fail_with_hint(repo, skills=skills, schemas=schemas)
    catalog = build_catalog_from_uml(
        skills_uml_paths=skills,
        behaviortreeschema_paths=schemas,
        constraints_dir=args.constraints_dir,
    )
    if args.bt_navigator_catalog:
        merge_bt_navigator_port_semantics(catalog, Path(args.bt_navigator_catalog))
    out = write_catalog(args.output, catalog)
    print(f"Wrote: {out}")
    print(f"Skills files: {len(skills)} | behaviortreeschema files: {len(schemas)} | skills in catalog: {len(catalog.get('atomic_skills', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

