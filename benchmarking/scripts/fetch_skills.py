#!/usr/bin/env python3
"""
Fetch skills from NAV4RAIL GitLab repositories.
=================================================
Clones/pulls the upstream repos and extracts skill definitions
to update the local skills catalog.

Usage:
    python scripts/fetch_skills.py
    python scripts/fetch_skills.py --output data/skills_catalog_upstream.yaml
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)

SKILLS_REPO = "https://gitlab.com/nav4rail/skills.git"
BT_REPO = "https://gitlab.com/nav4rail/behavior_trees.git"

REFERENCE_BTS_DIR = Path(__file__).parent.parent / "data" / "reference_bts"


def _clone_or_pull(repo_url: str, dest: Path) -> bool:
    """Clone repo or pull if already exists. Returns success."""
    try:
        if dest.exists():
            subprocess.run(
                ["git", "-C", str(dest), "pull", "--ff-only"],
                capture_output=True, timeout=60, check=True,
            )
            logger.info(f"Pulled {repo_url}")
        else:
            subprocess.run(
                ["git", "clone", "--depth=1", repo_url, str(dest)],
                capture_output=True, timeout=120, check=True,
            )
            logger.info(f"Cloned {repo_url}")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"Failed to fetch {repo_url}: {e}")
        return False


def _find_xml_files(directory: Path) -> list[Path]:
    """Find all XML files recursively."""
    return sorted(directory.rglob("*.xml"))


def _extract_skills_from_bt(xml_path: Path) -> set[str]:
    """Extract skill IDs from a BehaviorTree XML file."""
    skills = set()
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for elem in root.iter():
            if elem.tag in ("Action", "Condition"):
                skill_id = elem.get("ID", "")
                if skill_id:
                    skills.add(skill_id)
    except ET.ParseError:
        logger.warning(f"Failed to parse {xml_path}")
    return skills


def fetch_and_extract(output_dir: Path | None = None) -> dict:
    """
    Fetch repos and extract skill information.

    Returns:
        {"skills_found": set, "bt_files": list, "success": bool}
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        skills_dir = tmpdir / "skills"
        bt_dir = tmpdir / "behavior_trees"

        skills_ok = _clone_or_pull(SKILLS_REPO, skills_dir)
        bt_ok = _clone_or_pull(BT_REPO, bt_dir)

        if not (skills_ok or bt_ok):
            logger.error("Could not fetch any upstream repos")
            return {"skills_found": set(), "bt_files": [], "success": False}

        # Extract skills from BT XML files
        all_skills: set[str] = set()
        bt_files: list[str] = []

        if bt_ok:
            xml_files = _find_xml_files(bt_dir)
            for xml_path in xml_files:
                skills = _extract_skills_from_bt(xml_path)
                all_skills.update(skills)
                bt_files.append(xml_path.name)
                logger.info(f"  {xml_path.name}: {len(skills)} skills")

            # Copy reference BTs
            if output_dir:
                ref_dir = output_dir / "reference_bts"
                ref_dir.mkdir(parents=True, exist_ok=True)
                for xml_path in xml_files:
                    shutil.copy2(xml_path, ref_dir / xml_path.name)
                logger.info(f"Copied {len(xml_files)} reference BTs to {ref_dir}")

        # Try to parse skill definitions from skills repo
        if skills_ok:
            # Look for CMakeLists.txt, package.xml, or header files
            # that define skill interfaces
            for pattern in ["**/*.hpp", "**/*.h", "**/*.xml", "**/*.yaml"]:
                for f in skills_dir.rglob(pattern.split("/")[-1]):
                    logger.info(f"  Skills repo file: {f.relative_to(skills_dir)}")

        logger.info(f"Total unique skills found: {len(all_skills)}")
        if all_skills:
            logger.info(f"Skills: {sorted(all_skills)}")

        return {
            "skills_found": all_skills,
            "bt_files": bt_files,
            "success": skills_ok or bt_ok,
        }


def main():
    parser = argparse.ArgumentParser(description="Fetch NAV4RAIL skills from GitLab")
    parser.add_argument("--output", default=None, help="Output directory for reference BTs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path(args.output) if args.output else Path(__file__).parent.parent / "data"
    result = fetch_and_extract(output_dir)

    if result["success"]:
        print(f"\nFound {len(result['skills_found'])} skills from {len(result['bt_files'])} BT files")
        print("Update data/skills_catalog.yaml if new skills are found.")
    else:
        print("\nFailed to fetch upstream repos. Using cached local catalog.")


if __name__ == "__main__":
    main()
