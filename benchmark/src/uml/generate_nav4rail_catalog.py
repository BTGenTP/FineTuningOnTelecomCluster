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
else:
    from .catalog_generator import build_catalog_from_uml, write_catalog


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate NAV4RAIL skills catalog JSON from Papyrus UML.")
    p.add_argument("--nav4rails-repo", required=True, type=str, help="Path to nav4rails_repo root.")
    p.add_argument("--output", required=True, type=str, help="Output JSON path.")
    p.add_argument("--constraints-dir", default=None, type=str, help="Optional constraints directory for enums/tag overrides.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.nav4rails_repo).expanduser().resolve()
    skills = list(repo.glob("skills/*/models/skills/*.skills.uml"))
    schemas = list(repo.glob("behavior_trees/**/*.behaviortreeschema"))
    catalog = build_catalog_from_uml(
        skills_uml_paths=skills,
        behaviortreeschema_paths=schemas,
        constraints_dir=args.constraints_dir,
    )
    out = write_catalog(args.output, catalog)
    print(f"Wrote: {out}")
    print(f"Skills files: {len(skills)} | behaviortreeschema files: {len(schemas)} | skills in catalog: {len(catalog.get('atomic_skills', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

