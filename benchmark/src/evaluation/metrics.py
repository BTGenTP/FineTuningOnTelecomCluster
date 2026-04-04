from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET

from ..data.catalog import allowed_skill_ids


def exact_match_ratio(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    matches = sum(1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip())
    return matches / max(1, len(predictions))


def _tree_signature(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    signature: list[str] = []
    for node in root.iter():
        if "ID" in node.attrib:
            signature.append(f"{node.tag}:{node.attrib['ID']}")
        else:
            signature.append(node.tag)
    return signature


def _levenshtein(left: Sequence[str], right: Sequence[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, token_left in enumerate(left, start=1):
        curr = [i]
        for j, token_right in enumerate(right, start=1):
            cost = 0 if token_left == token_right else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def tree_edit_distance(prediction_xml: str, reference_xml: str) -> int:
    return _levenshtein(_tree_signature(prediction_xml), _tree_signature(reference_xml))


def hallucination_rate(xml_text: str, catalog: Mapping[str, Any]) -> float:
    root = ET.fromstring(xml_text)
    allowed = allowed_skill_ids(catalog)
    skill_ids = [node.attrib["ID"] for node in root.iter() if "ID" in node.attrib and node.tag in {"Action", "Condition"}]
    if not skill_ids:
        return 0.0
    hallucinated = [skill_id for skill_id in skill_ids if skill_id not in allowed]
    return len(hallucinated) / len(skill_ids)


def throughput(tokens_generated: int, latency_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    return tokens_generated / (latency_ms / 1000.0)


def ms_per_token(tokens_generated: int, latency_ms: float) -> float:
    if tokens_generated <= 0:
        return 0.0
    return latency_ms / tokens_generated


def build_metrics_row(
    *,
    run_id: str,
    method: str,
    model_name: str,
    xml_valid: bool,
    latency_ms: float,
    tokens_generated: int,
    vram_mb: float | None,
    prediction_xml: str,
    reference_xml: str | None,
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "run_id": run_id,
        "method": method,
        "model": model_name,
        "xml_valid": xml_valid,
        "latency_ms": latency_ms,
        "tokens_generated": tokens_generated,
        "ms_per_token": ms_per_token(tokens_generated, latency_ms),
        "throughput_toks_per_s": throughput(tokens_generated, latency_ms),
        "peak_vram_mb": vram_mb,
        "hallucination_rate": hallucination_rate(prediction_xml, catalog),
        "exact_match": False,
        "tree_edit_distance": None,
    }
    if reference_xml is not None:
        row["exact_match"] = prediction_xml.strip() == reference_xml.strip()
        row["tree_edit_distance"] = tree_edit_distance(prediction_xml, reference_xml)
    return row


def dataframe_from_runs(rows: Sequence[Mapping[str, Any]]) -> Any:
    import pandas as pd

    return pd.DataFrame(list(rows))


def render_markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    try:
        df = dataframe_from_runs(rows)
        if df.empty:
            return "No results."
        return df.to_markdown(index=False)
    except Exception:
        if not rows:
            return "No results."
        headers = list(rows[0].keys())
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
        return "\n".join(lines)


def write_reports(output_dir: str | Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Path]:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark_results.csv"
    md_path = out_dir / "benchmark_results.md"
    png_path = out_dir / "latency_vs_validity.png"
    md_path.write_text(render_markdown_table(rows) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            handle.write("")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = dataframe_from_runs(rows)
        if not df.empty and {"method", "latency_ms"}.issubset(df.columns):
            plt.figure(figsize=(8, 4))
            sns.barplot(data=df, x="method", y="latency_ms")
            plt.tight_layout()
            plt.savefig(png_path, dpi=200)
            plt.close()
    except Exception:
        pass
    return {"csv": csv_path, "markdown": md_path, "plot": png_path}
