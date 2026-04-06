"""
Validation batch des ports (L4) sur le dataset JSONL NAV4RAIL.

Usage :
    python validate_ports_dataset.py [dataset.jsonl]
    python validate_ports_dataset.py  # default: dataset_nav4rail_llm_2000.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from validate_bt import validate_ports  # noqa: E402


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = Path(__file__).parent / "dataset_nav4rail_llm_2000.jsonl"
        if not path.exists():
            path = Path(__file__).parent / "inspect_merged.jsonl"

    if not path.exists():
        print(f"Fichier introuvable : {path}", file=sys.stderr)
        sys.exit(1)

    samples = [
        json.loads(line)
        for line in path.read_text("utf-8").splitlines()
        if line.strip()
    ]
    total = len(samples)
    clean = 0
    all_issues: list[dict] = []
    issue_counter: Counter[str] = Counter()

    for i, s in enumerate(samples):
        issues = validate_ports(s.get("xml", ""))
        if not issues:
            clean += 1
        else:
            all_issues.append(
                {
                    "index": i,
                    "mission": s.get("mission", "")[:80],
                    "issues": issues,
                }
            )
            for iss in issues:
                # Extrait le type d'issue (ex: 'port requis "x" manquant')
                key = iss.split(" : ", 1)[1] if " : " in iss else iss
                issue_counter[key] += 1

    # ─── Résumé console ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Validation L4 (Ports) — {path.name}")
    print(f"{'=' * 60}")
    print(f"Samples analysés    : {total}")
    print(f"Samples OK          : {clean} ({clean / total * 100:.1f}%)")
    print(
        f"Samples avec issues : {total - clean} ({(total - clean) / total * 100:.1f}%)"
    )
    print(f"Issues totales      : {sum(issue_counter.values())}")

    if issue_counter:
        print(f"\nTop issues :")
        for issue_type, count in issue_counter.most_common(15):
            print(f"  {count:4d}×  {issue_type}")

    if all_issues:
        print(f"\nPremiers 10 samples problématiques :")
        for entry in all_issues[:10]:
            print(f"  #{entry['index']:4d}: {entry['mission']}")
            for iss in entry["issues"][:3]:
                print(f"         → {iss}")
            if len(entry["issues"]) > 3:
                print(f"         … +{len(entry['issues']) - 3} autres")

    # ─── Rapport JSON ────────────────────────────────────────────────────────
    report = {
        "total": total,
        "clean": clean,
        "issues_count": total - clean,
        "issues_total": sum(issue_counter.values()),
        "top_issues": dict(issue_counter.most_common(20)),
        "samples_with_issues": all_issues,
    }
    report_path = path.with_name(path.stem + ".ports_report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nRapport JSON : {report_path}")


if __name__ == "__main__":
    main()
