"""
NAV4RAIL BT Generator — Gradio Web Application.

Inference via Télécom Paris GPU cluster (on-demand SLURM provisioning via SSH).
Shares cluster provisioning logic with the FastAPI webapp.

Usage:
    python app.py
"""

import asyncio
import json
import logging
import re
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import gradio as gr
import httpx

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("nav4rail")

# ─── Import validation + test missions from existing webapp ─────────────────

APP_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = Path.home() / "nav4rail_webapp"  # shared inference, validation, grammar
FINETUNE_DIR = WEBAPP_DIR / "finetune"
sys.path.insert(0, str(WEBAPP_DIR))
sys.path.insert(0, str(FINETUNE_DIR))

from validate_bt import validate_bt, validate_ports  # noqa: E402
from inference import TEST_MISSIONS  # noqa: E402
from bt_visualizer import render_bt_html, render_bt_full_html  # noqa: E402

# ─── Cluster config (same as FastAPI webapp) ────────────────────────────────

TELECOM_URL = "http://localhost:8080"
DATAIA25_URL = "http://localhost:8081"
MAX_TIME_S = 900
GPU_HOST = "gpu"
DATAIA25_HOST = "192.168.80.211"
DATAIA25_PORT = 8080
REMOTE_USER = "blepourt-25"
CLUSTER_PORT = 8080

# Active cluster URL — updated dynamically
CLUSTER_URL = TELECOM_URL

MODEL_LABELS = {
    "mistral-7b-merged-fp16": "Mistral 7B",
    "llama3-8b-merged-fp16": "Llama 3.1 8B",
    "qwen-coder-7b-merged-fp16": "Qwen 2.5 Coder 7B",
    "gemma2-9b-merged-fp16": "Gemma 2 9B",
    "qwen-14b-merged-fp16": "Qwen 2.5 14B",
}

# Map model API key → SLURM model_key for job_serve_fp16.sh
MODEL_SLURM_KEY = {
    "mistral-7b-merged-fp16": "mistral_7b",
    "llama3-8b-merged-fp16": "llama3_8b",
    "qwen-coder-7b-merged-fp16": "qwen25_coder_7b",
    "gemma2-9b-merged-fp16": "gemma2_9b",
    "qwen-14b-merged-fp16": "qwen25_14b",
}

# dataia25 script key
MODEL_DATAIA25_KEY = {
    "mistral-7b-merged-fp16": "mistral_7b",
    "llama3-8b-merged-fp16": "llama3_8b",
    "qwen-coder-7b-merged-fp16": "qwen25_coder_7b",
    "gemma2-9b-merged-fp16": "gemma2_9b",
    "qwen-14b-merged-fp16": "qwen25_14b",
}

# GPU VRAM requirements (fp16): model → fits P100(16GB)?, fits 3090(24GB)?
MODEL_GPU_FIT = {
    "mistral-7b-merged-fp16": {"P100": True, "3090": True},
    "llama3-8b-merged-fp16": {"P100": False, "3090": True},
    "qwen-coder-7b-merged-fp16": {"P100": False, "3090": True},
    "gemma2-9b-merged-fp16": {"P100": False, "3090": True},
    "qwen-14b-merged-fp16": {"P100": False, "3090": False},  # needs 2×GPU
}

# SLURM partition parameters
SLURM_PARTITIONS = {
    "P100": {"partition": "P100", "gres": "gpu:1", "mem": "16G"},
    "3090": {"partition": "3090", "gres": "gpu:1", "mem": "32G"},
}

FAILOVER_WAIT_S = 30  # seconds to wait before falling back to dataia25

HISTORY_PATH = APP_DIR / "history.json"
ROOT_PATH = "/btgenerator_gradio"

# ─── Dataset loading & classification ───────────────────────────────────────

_INSPECTION_KW = re.compile(
    r"inspect|mesure|rail|ballast|caténaire|aiguillage|appareil de voie|traverse"
    r"|éclisse|tirefond|soudure|défaut",
    re.IGNORECASE,
)


def classify_mission(mission: str) -> str:
    m = mission.lower()
    if "simulation" in m:
        return "simulation"
    if "correction" in m or "anomalie" in m or "correcti" in m:
        return "correction_anomalie"
    if "arrêts multiples" in m or "arrets multiples" in m:
        return "transport_arrêts_multiples"
    if _INSPECTION_KW.search(mission):
        return "inspect-ctrl"
    if "autoris" in m:
        return "inspect+ctrl"
    return "transport_simple"


def _xml_metrics(xml_str: str) -> dict:
    """Extract complexity metrics from a BT XML string."""
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return {"nodes": 0, "subtrees": 0, "leaf_nodes": 0, "xml_size": len(xml_str)}
    all_nodes = list(root.iter())
    subtrees = root.findall("BehaviorTree")
    leaf_nodes = [n for n in all_nodes if len(list(n)) == 0]
    return {
        "nodes": len(all_nodes),
        "subtrees": len(subtrees),
        "leaf_nodes": len(leaf_nodes),
        "xml_size": len(xml_str),
    }


def _load_dataset() -> list[dict]:
    """Load inspect_merged.jsonl, add category + metrics to each sample."""
    candidates = [
        APP_DIR / "data" / "inspect_merged.jsonl",
        APP_DIR.parent / "finetune" / "inspect_merged.jsonl",
        FINETUNE_DIR / "inspect_merged.jsonl",
    ]
    path = None
    for p in candidates:
        if p.exists():
            path = p
            break
    if path is None:
        return []
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        s["category"] = classify_mission(s.get("mission", ""))
        m = _xml_metrics(s.get("xml", ""))
        s.update(m)
        port_issues = validate_ports(s.get("xml", ""))
        s["port_issues"] = port_issues
        s["port_ok"] = len(port_issues) == 0
        samples.append(s)
    return samples


DATASET: list[dict] = _load_dataset()

# Pre-computed stats
_CAT_ORDER = [
    "inspect-ctrl",
    "transport_arrêts_multiples",
    "simulation",
    "correction_anomalie",
    "transport_simple",
    "inspect+ctrl",
]
_CAT_LABELS = {
    "inspect-ctrl": "Inspection sans contrôle",
    "transport_arrêts_multiples": "Transport arrêts multiples",
    "simulation": "Simulation",
    "correction_anomalie": "Correction d'anomalie",
    "transport_simple": "Transport simple",
    "inspect+ctrl": "Transport avec autorisation",
}


def _compute_stats(samples: list[dict]) -> dict:
    if not samples:
        return {}
    from collections import Counter

    cats = Counter(s["category"] for s in samples)
    scores = Counter(round(s.get("score", 0), 1) for s in samples)
    nodes = [s["nodes"] for s in samples]
    subtrees = [s["subtrees"] for s in samples]
    leaf = [s["leaf_nodes"] for s in samples]
    sizes = [s["xml_size"] for s in samples]
    port_ok = sum(1 for s in samples if s.get("port_ok", False))
    port_issues_total = sum(len(s.get("port_issues", [])) for s in samples)
    return {
        "total": len(samples),
        "categories": {c: cats.get(c, 0) for c in _CAT_ORDER},
        "scores": dict(sorted(scores.items(), reverse=True)),
        "nodes": {
            "min": min(nodes),
            "avg": mean(nodes),
            "med": median(nodes),
            "max": max(nodes),
        },
        "subtrees": {
            "min": min(subtrees),
            "avg": mean(subtrees),
            "med": median(subtrees),
            "max": max(subtrees),
        },
        "leaf_nodes": {
            "min": min(leaf),
            "avg": mean(leaf),
            "med": median(leaf),
            "max": max(leaf),
        },
        "xml_size": {
            "min": min(sizes),
            "avg": mean(sizes),
            "med": median(sizes),
            "max": max(sizes),
        },
        "port_ok": port_ok,
        "port_issues_total": port_issues_total,
    }


DATASET_STATS = _compute_stats(DATASET)


# ─── Pipeline explanation (embedded constant) ──────────────────────────────

PIPELINE_MD = """\
### Comment ce dataset a été créé

**Problème cold-start** : un seul Behavior Tree de référence (`behavior_tree_example.xml`, \
173 lignes, 14 sous-arbres). Impossible de fine-tuner avec un unique exemple.

**Solution** : utiliser un **LLM puissant (Llama 3.3 70B AWQ INT4)** comme **proxy de \
génération** pour produire synthétiquement 2000 paires `(instruction, xml)` validées.

#### Architecture — Pipeline LangGraph à 3 nœuds

```
generate_instruction ──→ generate_xml ──→ validate_xml
         │                                     │
         │            ◄── self-correction ◄────┘ (si erreur, max 3 retries)
```

1. **generate_instruction** — Génère des missions uniques via 11 templates × 10 éléments \
d'inspection = 12 000+ combinaisons possibles. Déduplication par set.
2. **generate_xml** — Envoie au LLM 70B un prompt structuré (~2000 tokens) avec : \
règles de format, catalogue de 27 skills, exemple few-shot condensé, et une **classification \
sémantique** de la mission qui détermine les sous-arbres requis (transport, inspect+ctrl, \
inspect-ctrl).
3. **validate_xml** — Validation à 3 niveaux :
   - **L1 Syntaxique** : XML bien formé, `BTCPP_format="4"`, tags reconnus
   - **L2 Structurel** : pas de nœuds de contrôle vides, profondeur raisonnable
   - **L3 Sémantique** : ordre des skills, cohérence mission ↔ XML

#### Infrastructure

| | |
|---|---|
| **LLM** | Llama 3.3 70B AWQ INT4 via vLLM |
| **GPU** | 2× RTX 3090 sur Vast.ai |
| **Débit** | ~115 samples/heure |
| **Durée** | ~17h pour 2000 samples |
| **Coût** | ~6-7$ total |
| **Taux validation** | 100% au 1er essai (score ≥ 0.9) |

#### Catalogue de skills (27 skills, 4 familles)

| Famille | Skills | Count |
|---------|--------|-------|
| **Preparation** | LoadMission, MissionStructureValid, UpdateCurrentGeneratedActivity, \
ProjectPointOnNetwork, CreatePath, AgregatePath, MissionFullyTreated, PassAdvancedPath, \
PassMission, GenerateMissionSequence, GenerateCorrectiveSubSequence, \
InsertCorrectiveSubSequence | 12 |
| **Motion** | MissionTerminated, CheckCurrentStepType, PassMotionParameters, Move, \
UpdateCurrentExecutedStep, Deccelerate, MoveAndStop, SignalAndWaitForOrder, \
IsRobotPoseProjectionActive | 9 |
| **Inspection** | ManageMeasurements, AnalyseMeasurements, MeasurementsQualityValidated, \
PassDefectsLocalization, MeasurementsEnforcedValidated | 5 |
| **Simulation** | SimulationStarted | 1 |
"""


# ─── Statistics dashboard HTML ──────────────────────────────────────────────


def _build_stats_html(stats: dict) -> str:
    if not stats:
        return "<p>Dataset non chargé.</p>"
    total = stats["total"]

    # Category distribution
    cats_html = ""
    max_cat = max(stats["categories"].values()) if stats["categories"] else 1
    for cat in _CAT_ORDER:
        count = stats["categories"].get(cat, 0)
        pct = count / total * 100
        bar_w = count / max_cat * 100
        label = _CAT_LABELS.get(cat, cat)
        cats_html += f"""
        <tr>
            <td style="padding:4px 8px;white-space:nowrap;">{label}</td>
            <td style="padding:4px 8px;text-align:right;font-weight:600;">{count}</td>
            <td style="padding:4px 8px;text-align:right;color:#9ca3af;">{pct:.1f}%</td>
            <td style="padding:4px 8px;width:50%;">
                <div style="background:#374151;border-radius:4px;height:16px;overflow:hidden;">
                    <div style="background:#3b82f6;height:100%;width:{bar_w:.0f}%;border-radius:4px;"></div>
                </div>
            </td>
        </tr>"""

    # Score distribution
    scores_html = ""
    for score_val, count in stats["scores"].items():
        color = "#22c55e" if score_val >= 1.0 else "#eab308"
        scores_html += (
            f'<span style="background:{color};color:white;padding:2px 10px;'
            f'border-radius:4px;margin-right:8px;font-size:13px;">'
            f"Score {score_val} — {count} samples ({count / total * 100:.1f}%)</span>"
        )

    # Complexity table
    def _row(label, d):
        return (
            f"<tr><td style='padding:4px 8px;'>{label}</td>"
            f"<td style='padding:4px 8px;text-align:right;'>{d['min']}</td>"
            f"<td style='padding:4px 8px;text-align:right;'>{d['avg']:.1f}</td>"
            f"<td style='padding:4px 8px;text-align:right;'>{d['med']:.0f}</td>"
            f"<td style='padding:4px 8px;text-align:right;'>{d['max']}</td></tr>"
        )

    complexity_html = (
        "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        "<tr style='border-bottom:1px solid #4b5563;'>"
        "<th style='padding:4px 8px;text-align:left;'>Métrique</th>"
        "<th style='padding:4px 8px;text-align:right;'>Min</th>"
        "<th style='padding:4px 8px;text-align:right;'>Moy</th>"
        "<th style='padding:4px 8px;text-align:right;'>Méd</th>"
        "<th style='padding:4px 8px;text-align:right;'>Max</th></tr>"
        + _row("Nœuds / sample", stats["nodes"])
        + _row("Sous-arbres / sample", stats["subtrees"])
        + _row("Feuilles / sample", stats["leaf_nodes"])
        + _row("Taille XML (chars)", stats["xml_size"])
        + "</table>"
    )

    return f"""
    <div style="font-family:system-ui,sans-serif;">
        <div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap;">
            <div style="background:#1e3a5f;padding:12px 20px;border-radius:8px;text-align:center;">
                <div style="font-size:28px;font-weight:bold;color:#60a5fa;">{total}</div>
                <div style="font-size:12px;color:#93c5fd;">samples</div>
            </div>
            <div style="background:#1e3a5f;padding:12px 20px;border-radius:8px;text-align:center;">
                <div style="font-size:28px;font-weight:bold;color:#4ade80;">100%</div>
                <div style="font-size:12px;color:#86efac;">valides (score ≥ 0.9)</div>
            </div>
            <div style="background:#1e3a5f;padding:12px 20px;border-radius:8px;text-align:center;">
                <div style="font-size:28px;font-weight:bold;color:#fbbf24;">{len(_CAT_ORDER)}</div>
                <div style="font-size:12px;color:#fde68a;">catégories</div>
            </div>
            <div style="background:#1e3a5f;padding:12px 20px;border-radius:8px;text-align:center;">
                <div style="font-size:28px;font-weight:bold;color:{"#4ade80" if stats.get("port_ok", 0) == total else "#fbbf24"};">{stats.get("port_ok", 0)}/{total}</div>
                <div style="font-size:12px;color:#93c5fd;">L4 Ports OK</div>
            </div>
        </div>

        <h4 style="margin:12px 0 6px;">Distribution des catégories</h4>
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
            {cats_html}
        </table>

        <h4 style="margin:16px 0 6px;">Scores de validation</h4>
        <div style="margin-bottom:12px;">{scores_html}</div>

        <h4 style="margin:16px 0 6px;">Complexité structurelle</h4>
        {complexity_html}

        <div style="margin-top:16px;padding:10px;background:#422006;border:1px solid #92400e;
                    border-radius:6px;font-size:12px;color:#fbbf24;">
            <strong>⚠ Points de vigilance :</strong><br>
            • Déséquilibre catégoriel : ~77% inspection sans contrôle, ~2% par catégorie mineure<br>
            • ~780 near-duplicates (39%) ne différant que par les km — diversité structurelle réelle ~1220 patterns
        </div>
    </div>"""


# ─── Dataset sample browser functions ───────────────────────────────────────


def _filter_dataset(
    category: str, score_filter: str, search_text: str, port_filter: str = "Tous"
) -> list[tuple[int, dict]]:
    """Return (original_index, sample) pairs matching filters."""
    results = []
    for i, s in enumerate(DATASET):
        if category != "Toutes" and s["category"] != category:
            continue
        if score_filter == "1.0" and round(s.get("score", 0), 1) != 1.0:
            continue
        if score_filter == "0.9" and round(s.get("score", 0), 1) != 0.9:
            continue
        if search_text and search_text.lower() not in s.get("mission", "").lower():
            continue
        if port_filter == "L4 OK" and not s.get("port_ok", True):
            continue
        if port_filter == "L4 Issues" and s.get("port_ok", True):
            continue
        results.append((i, s))
    return results


def _render_sample(
    sample: dict, idx_in_filtered: int, total_filtered: int, original_index: int = -1
) -> tuple:
    """Return (status_md, mission_html, score_html, port_html, xml_code, prompt_code, bt_viz_html)."""
    mission = sample.get("mission", "")
    xml = sample.get("xml", "")
    prompt = sample.get("prompt", "")
    # Strip LLM response: keep only the instruction part before [/INST]
    if "[/INST]" in prompt:
        prompt = prompt[: prompt.index("[/INST]") + len("[/INST]")]
    score = sample.get("score", 0)
    cat = sample.get("category", "")
    cat_label = _CAT_LABELS.get(cat, cat)
    nodes = sample.get("nodes", 0)
    subtrees = sample.get("subtrees", 0)

    ds_id = f" · **ID #{original_index + 1}**" if original_index >= 0 else ""
    status = (
        f"**Sample {idx_in_filtered + 1} / {total_filtered}**{ds_id} — "
        f"Catégorie : {cat_label} — Score : {score} — "
        f"{nodes} nœuds, {subtrees} sous-arbres"
    )

    mission_html = (
        f'<div style="font-size:16px;padding:12px;background:#1e293b;border-radius:8px;'
        f'border-left:4px solid #3b82f6;margin-bottom:8px;">'
        f'<span style="color:#94a3b8;font-size:12px;">MISSION</span><br>'
        f'<span style="color:#f1f5f9;font-weight:500;">{mission}</span></div>'
    )

    badge_color = "#22c55e" if score >= 1.0 else "#eab308"
    score_html = (
        f'<span style="background:{badge_color};color:white;padding:3px 10px;'
        f'border-radius:5px;font-size:13px;font-weight:600;">Score {score}</span>'
        f' <span style="color:#9ca3af;font-size:12px;margin-left:8px;">'
        f"{nodes} nœuds · {subtrees} sous-arbres · {sample.get('xml_size', 0)} chars"
        f"</span>"
    )

    # Port validation (L4)
    port_issues = sample.get("port_issues", [])
    if port_issues:
        port_html = (
            '<div style="margin-top:8px;padding:8px;background:#422006;'
            'border:1px solid #92400e;border-radius:6px;font-size:12px;">'
            f'<span style="color:#fbbf24;font-weight:600;">'
            f"⚠ L4 Ports — {len(port_issues)} issue(s)</span><br>"
        )
        for iss in port_issues[:5]:
            iss_escaped = iss.replace("<", "&lt;").replace(">", "&gt;")
            port_html += f'<span style="color:#fde68a;">{iss_escaped}</span><br>'
        if len(port_issues) > 5:
            port_html += (
                f'<span style="color:#9ca3af;">… +{len(port_issues) - 5} autres</span>'
            )
        port_html += "</div>"
    else:
        port_html = (
            '<div style="margin-top:8px;padding:8px;background:#052e16;'
            'border:1px solid #166534;border-radius:6px;font-size:12px;">'
            '<span style="color:#4ade80;font-weight:600;">'
            "✓ L4 Ports — OK</span></div>"
        )

    bt_html = render_bt_full_html(xml) if xml else ""

    return status, mission_html, score_html, port_html, xml, prompt, bt_html


def ds_apply_filters(category, score_filter, search_text, port_filter="Tous"):
    """Apply filters, return updated state + first sample display + slider update."""
    filtered = _filter_dataset(category, score_filter, search_text, port_filter)
    if not filtered:
        empty = (
            "**0 résultats**",
            "<p>Aucun sample ne correspond aux filtres.</p>",
            "",
            "",
            "",
            "",
            "",
        )
        return (
            filtered,
            0,
            gr.update(minimum=1, maximum=1, value=1, label="Sample 0/0"),
            *empty,
        )

    orig_idx, sample = filtered[0]
    display = _render_sample(sample, 0, len(filtered), original_index=orig_idx)
    slider_update = gr.update(
        minimum=1,
        maximum=len(filtered),
        value=1,
        label=f"Sample (1–{len(filtered)})",
    )
    return filtered, 0, slider_update, *display


def ds_navigate(filtered, slider_val):
    """Navigate to sample at slider position."""
    if not filtered:
        return 0, "**0 résultats**", "", "", "", "", "", ""
    idx = max(0, min(int(slider_val) - 1, len(filtered) - 1))
    orig_idx, sample = filtered[idx]
    display = _render_sample(sample, idx, len(filtered), original_index=orig_idx)
    return idx, *display


def ds_prev(filtered, current_idx):
    """Go to previous sample."""
    if not filtered:
        return 0, gr.update(), "**0 résultats**", "", "", "", "", "", ""
    idx = max(0, current_idx - 1)
    orig_idx, sample = filtered[idx]
    display = _render_sample(sample, idx, len(filtered), original_index=orig_idx)
    return idx, gr.update(value=idx + 1), *display


def ds_next(filtered, current_idx):
    """Go to next sample."""
    if not filtered:
        return 0, gr.update(), "**0 résultats**", "", "", "", "", "", ""
    idx = min(len(filtered) - 1, current_idx + 1)
    orig_idx, sample = filtered[idx]
    display = _render_sample(sample, idx, len(filtered), original_index=orig_idx)
    return idx, gr.update(value=idx + 1), *display


# ─── Compare two samples ────────────────────────────────────────────────────


def _render_compare_card(sample: dict, label: str) -> str:
    """Render a compact HTML card for one side of the comparison."""
    mission = sample.get("mission", "")
    score = sample.get("score", 0)
    cat = sample.get("category", "")
    cat_label = _CAT_LABELS.get(cat, cat)
    nodes = sample.get("nodes", 0)
    subtrees = sample.get("subtrees", 0)
    xml_size = sample.get("xml_size", 0)
    port_issues = sample.get("port_issues", [])

    badge_color = "#22c55e" if score >= 1.0 else "#eab308"
    port_color = "#4ade80" if not port_issues else "#fbbf24"
    port_text = "✓ Ports OK" if not port_issues else f"⚠ {len(port_issues)} issue(s)"

    return f"""
    <div style="font-family:system-ui,sans-serif;">
        <div style="font-size:11px;color:#64748b;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:6px;">{label}</div>
        <div style="padding:10px;background:#1e293b;border-radius:8px;
                    border-left:4px solid #3b82f6;margin-bottom:8px;">
            <span style="color:#f1f5f9;font-size:14px;">{mission}</span>
        </div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:6px;">
            <span style="background:{badge_color};color:white;padding:2px 8px;
                         border-radius:4px;font-size:12px;font-weight:600;">Score {score}</span>
            <span style="color:#9ca3af;font-size:11px;">{cat_label}</span>
            <span style="color:#9ca3af;font-size:11px;">|</span>
            <span style="color:#9ca3af;font-size:11px;">{nodes} nœuds · {subtrees} sous-arbres · {xml_size} chars</span>
            <span style="color:{port_color};font-size:11px;">{port_text}</span>
        </div>
    </div>"""


def _render_diff_summary(s1: dict, s2: dict) -> str:
    """Render an HTML summary highlighting differences between two samples."""
    rows = ""

    def _diff_row(label, v1, v2, fmt=str):
        color = "#4ade80" if v1 == v2 else "#fbbf24"
        return (
            f"<tr>"
            f"<td style='padding:3px 8px;color:#94a3b8;font-size:12px;'>{label}</td>"
            f"<td style='padding:3px 8px;text-align:right;font-size:12px;'>{fmt(v1)}</td>"
            f"<td style='padding:3px 8px;text-align:center;color:{color};font-size:12px;'>"
            f"{'=' if v1 == v2 else '≠'}</td>"
            f"<td style='padding:3px 8px;text-align:right;font-size:12px;'>{fmt(v2)}</td>"
            f"</tr>"
        )

    rows += _diff_row("Score", s1.get("score", 0), s2.get("score", 0))
    rows += _diff_row(
        "Catégorie",
        _CAT_LABELS.get(s1.get("category", ""), s1.get("category", "")),
        _CAT_LABELS.get(s2.get("category", ""), s2.get("category", "")),
    )
    rows += _diff_row("Nœuds", s1.get("nodes", 0), s2.get("nodes", 0))
    rows += _diff_row("Sous-arbres", s1.get("subtrees", 0), s2.get("subtrees", 0))
    rows += _diff_row("Feuilles", s1.get("leaf_nodes", 0), s2.get("leaf_nodes", 0))
    rows += _diff_row(
        "Taille XML",
        s1.get("xml_size", 0),
        s2.get("xml_size", 0),
        lambda x: f"{x} chars",
    )
    rows += _diff_row(
        "Ports OK",
        len(s1.get("port_issues", [])) == 0,
        len(s2.get("port_issues", [])) == 0,
        lambda x: "✓" if x else "✗",
    )

    return f"""
    <div style="margin:8px 0;">
        <div style="font-size:12px;color:#64748b;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:4px;">Comparaison</div>
        <table style="width:100%;border-collapse:collapse;font-family:system-ui,sans-serif;">
            <tr style="border-bottom:1px solid #374151;">
                <th style="padding:3px 8px;text-align:left;font-size:11px;color:#64748b;">Métrique</th>
                <th style="padding:3px 8px;text-align:right;font-size:11px;color:#3b82f6;">Sample A</th>
                <th style="padding:3px 8px;text-align:center;font-size:11px;color:#64748b;"></th>
                <th style="padding:3px 8px;text-align:right;font-size:11px;color:#f59e0b;">Sample B</th>
            </tr>
            {rows}
        </table>
    </div>"""


def cmp_load_sample(sample_num: int, side: str):
    """Load a sample by its 1-based index. Returns (card_html, xml, bt_viz_html)."""
    idx = max(0, min(int(sample_num) - 1, len(DATASET) - 1))
    if not DATASET:
        return f"<p>Dataset vide</p>", "", ""
    sample = DATASET[idx]
    card = _render_compare_card(sample, f"Sample {side} (#{idx + 1})")
    xml = sample.get("xml", "")
    bt_html = render_bt_full_html(xml) if xml else ""
    return card, xml, bt_html


def _render_unified_diff_html(
    xml_a: str, xml_b: str, label_a: str, label_b: str
) -> str:
    """Render a unified diff of two XML strings as syntax-highlighted HTML."""
    import difflib

    lines_a = xml_a.splitlines(keepends=True)
    lines_b = xml_b.splitlines(keepends=True)
    diff = difflib.unified_diff(
        lines_a, lines_b, fromfile=label_a, tofile=label_b, lineterm=""
    )

    html_lines = []
    for line in diff:
        escaped = (
            line.rstrip("\n")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        if line.startswith("@@"):
            html_lines.append(
                f'<div style="background:#1e3a5f;color:#93c5fd;padding:1px 8px;'
                f'font-size:11px;margin:4px 0 0;">{escaped}</div>'
            )
        elif line.startswith("---"):
            html_lines.append(
                f'<div style="color:#f87171;padding:1px 8px;font-weight:600;'
                f'font-size:12px;border-bottom:1px solid #374151;">{escaped}</div>'
            )
        elif line.startswith("+++"):
            html_lines.append(
                f'<div style="color:#4ade80;padding:1px 8px;font-weight:600;'
                f'font-size:12px;border-bottom:1px solid #374151;">{escaped}</div>'
            )
        elif line.startswith("-"):
            html_lines.append(
                f'<div style="background:#3b1219;color:#fca5a5;padding:1px 8px;">{escaped}</div>'
            )
        elif line.startswith("+"):
            html_lines.append(
                f'<div style="background:#052e16;color:#86efac;padding:1px 8px;">{escaped}</div>'
            )
        else:
            html_lines.append(
                f'<div style="color:#d1d5db;padding:1px 8px;">{escaped}</div>'
            )

    if not html_lines:
        return (
            '<div style="padding:12px;background:#1e293b;border-radius:8px;'
            'font-family:monospace;font-size:12px;color:#4ade80;text-align:center;">'
            "✓ Les deux XML sont identiques</div>"
        )

    content = "\n".join(html_lines)
    return (
        f'<div style="background:#0f172a;border:1px solid #334155;border-radius:8px;'
        f"font-family:ui-monospace,monospace;font-size:11px;overflow-x:auto;"
        f'max-height:600px;overflow-y:auto;padding:4px 0;">'
        f"{content}</div>"
    )


def cmp_update_diff(num_a: int, num_b: int):
    """Compute the diff summary between two samples."""
    if not DATASET:
        return ""
    idx_a = max(0, min(int(num_a) - 1, len(DATASET) - 1))
    idx_b = max(0, min(int(num_b) - 1, len(DATASET) - 1))
    return _render_diff_summary(DATASET[idx_a], DATASET[idx_b])


def cmp_load_both(num_a: int, num_b: int):
    """Load both samples and compute diff. Returns all outputs."""
    card_a, xml_a, viz_a = cmp_load_sample(num_a, "A")
    card_b, xml_b, viz_b = cmp_load_sample(num_b, "B")
    diff_summary = cmp_update_diff(num_a, num_b)
    diff_xml = _render_unified_diff_html(
        xml_a,
        xml_b,
        f"Sample A (#{int(num_a)})",
        f"Sample B (#{int(num_b)})",
    )
    return card_a, viz_a, card_b, viz_b, diff_summary, diff_xml


# ─── History persistence ────────────────────────────────────────────────────


def load_history() -> list[dict]:
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text())
    return []


def save_history(entries: list[dict]):
    HISTORY_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2))


def append_history(entry: dict):
    entries = load_history()
    entries.insert(0, entry)
    save_history(entries)


# ─── Shell helper ───────────────────────────────────────────────────────────


def _run(cmd: str, timeout: float = 30) -> tuple[int, str]:
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:
        return -1, str(e)


def _run_bg(cmd: str, timeout: float = 30) -> int:
    """Run a command without capturing output (fire-and-forget SSH).

    Using capture_output=True with ssh -f or remote & causes subprocess.run
    to block until the background child closes inherited pipe FDs — which
    never happens until the remote process exits.  DEVNULL avoids this.
    """
    try:
        r = subprocess.run(
            cmd,
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        return r.returncode
    except subprocess.TimeoutExpired:
        return -1
    except Exception:
        return -1


# ─── Cluster provisioning (sync version for Gradio) ────────────────────────


def _check_health(url: str, timeout: float = 5) -> dict | None:
    """Fetch /health from a URL, return dict or None."""
    try:
        r = httpx.get(f"{url}/health", timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _check_cluster() -> bool:
    return _check_health(TELECOM_URL) is not None


def _check_dataia25() -> bool:
    return _check_health(DATAIA25_URL) is not None


def _ensure_vpn_and_gateway(progress=gr.Progress(), pct_start=0.05) -> bool:
    """Ensure VPN + route + SSH to gpu-gw are up. Shared by all Télécom provisioning."""
    progress(pct_start, desc="Vérification du VPN...")
    rc, _ = _run("nmcli -t -f NAME connection show --active | grep -q telecom-paris")
    if rc != 0:
        progress(pct_start + 0.02, desc="Activation du VPN Télécom Paris...")
        rc, out = _run("sudo nmcli connection up telecom-paris", timeout=15)
        if rc != 0:
            return False
        time.sleep(2)

    # Route (VPN peut être tun0 ou tun1)
    rc, _ = _run("ip route show | grep -q '137.194'")
    if rc != 0:
        tun = "tun1" if _run("ip link show tun1 2>/dev/null")[0] == 0 else "tun0"
        _run(f"sudo ip route add 137.194.0.0/16 dev {tun}")

    # Gateway
    progress(pct_start + 0.05, desc="Connexion à la passerelle GPU...")
    rc, _ = _run(f"ssh -o ConnectTimeout=5 {GPU_HOST} echo ok", timeout=10)
    return rc == 0


def _check_slurm_partition_free(partition: str) -> bool:
    """Check if a SLURM partition has idle/mix nodes with free GPUs."""
    rc, out = _run(
        f"ssh -o ConnectTimeout=5 {GPU_HOST} "
        f"\"sinfo -p {partition} -h -o '%t' | grep -qE 'idle|mix'\"",
        timeout=10,
    )
    return rc == 0


def _provision_telecom(
    model_choice: str, partition: str, progress=gr.Progress(), pct_start: float = 0.1
) -> bool:
    """Submit SLURM job for a specific model on a specific partition.

    Returns True if server ends up healthy on TELECOM_URL.
    """
    slurm_key = MODEL_SLURM_KEY.get(model_choice, "mistral_7b")
    part_cfg = SLURM_PARTITIONS[partition]

    # Check if our job is already running with the right model
    progress(pct_start, desc=f"Vérification job SLURM ({partition})...")
    rc, running = _run(
        f"ssh -o ConnectTimeout=5 {GPU_HOST} "
        f"'squeue -u $(whoami) -h -t R -n nav4rail-serve --format=\"%j\"'",
        timeout=15,
    )

    if running.strip():
        # Job running — check if it's serving the right model
        health = _check_health(TELECOM_URL, timeout=5)
        if health and health.get("model") == model_choice:
            return True
        # Wrong model or not reachable yet — cancel and resubmit
        _run(
            f"ssh {GPU_HOST} 'scancel -u $(whoami) -n nav4rail-serve 2>/dev/null'",
            timeout=10,
        )
        time.sleep(3)

    # Cancel any pending jobs
    _run(
        f"ssh {GPU_HOST} 'scancel -u $(whoami) -n nav4rail-serve -t PD 2>/dev/null'",
        timeout=10,
    )

    # Submit new job
    progress(pct_start + 0.05, desc=f"Soumission job {partition} ({slurm_key})...")
    rc, job_out = _run(
        f"ssh {GPU_HOST} 'cd ~/nav4rail_serve && "
        f"MODEL_KEY={slurm_key} sbatch --parsable "
        f"--partition={part_cfg['partition']} --gres={part_cfg['gres']} "
        f"--mem={part_cfg['mem']} job_serve_fp16.sh'",
        timeout=15,
    )
    job_id = job_out.strip()
    if rc != 0 or not job_id.isdigit():
        return False

    # Wait for GPU allocation
    node_name = ""
    for attempt in range(18):
        progress(
            pct_start + 0.1 + attempt * 0.02,
            desc=f"En attente d'un GPU {partition}... ({(attempt + 1) * 10}s)",
        )
        time.sleep(10)
        rc, state = _run(
            f"ssh -o ConnectTimeout=5 {GPU_HOST} "
            f"'squeue -j {job_id} -h --format=\"%T %N\"'",
            timeout=10,
        )
        parts = state.strip().split()
        state_val = parts[0] if parts else ""
        if state_val == "RUNNING":
            node_name = parts[1] if len(parts) > 1 else ""
            log.info(f"[provision] Job {job_id} running on {node_name}")
            progress(pct_start + 0.45, desc=f"Job GPU démarré sur {node_name} !")
            time.sleep(8)
            break
        if state_val == "" or "FAILED" in state_val or "CANCELLED" in state_val:
            return False
    else:
        return False

    # Wait for server ready (curl via node name, not localhost on gw)
    health_target = (
        f"http://{node_name}:{CLUSTER_PORT}"
        if node_name
        else f"http://localhost:{CLUSTER_PORT}"
    )
    progress(pct_start + 0.5, desc="Attente du serveur d'inférence...")
    for _ in range(48):  # up to ~240s (first run needs merge on disk)
        rc, health = _run(
            f"ssh -o ConnectTimeout=5 {GPU_HOST} "
            f"'curl -s --max-time 3 {health_target}/health'",
            timeout=10,
        )
        if '"status":"ok"' in health:
            break
        time.sleep(5)
    else:
        return False

    # SSH tunnel (through gw, targeting the compute node)
    _run(f"pkill -f 'ssh.*-L {CLUSTER_PORT}:' 2>/dev/null")
    time.sleep(1)
    tunnel_target = (
        f"{node_name}:{CLUSTER_PORT}" if node_name else f"localhost:{CLUSTER_PORT}"
    )
    progress(pct_start + 0.7, desc="Ouverture du tunnel SSH...")
    rc, _ = _run(
        f"ssh -fN -L {CLUSTER_PORT}:{tunnel_target} {GPU_HOST} "
        f"-o ServerAliveInterval=30 -o ServerAliveCountMax=3 "
        f"-o ExitOnForwardFailure=yes",
        timeout=10,
    )
    if rc != 0:
        _run(
            f"nohup ssh -N -L {CLUSTER_PORT}:{tunnel_target} {GPU_HOST} "
            f"-o ServerAliveInterval=30 -o ServerAliveCountMax=3 "
            f"-o ExitOnForwardFailure=yes </dev/null >/dev/null 2>&1 &",
            timeout=5,
        )
        time.sleep(3)

    # Final check
    for _ in range(6):
        if _check_cluster():
            return True
        time.sleep(2)
    return False


def _smart_provision(
    model_choice: str, progress=gr.Progress()
) -> tuple[str | None, str]:
    """Smart failover provisioning. Returns (target_url, cluster_label) or (None, error_msg).

    Strategy:
      - Mistral 7B: P100 → 3090 → dataia25
      - Llama/Qwen Coder/Gemma: 3090 → dataia25
      - Qwen 14B: dataia25 only
    """
    log.info(f"[provision] Starting smart provision for {model_choice}")
    fits = MODEL_GPU_FIT.get(model_choice, {})

    # 1) Fast path: check both clusters for already-serving model
    health_telecom = _check_health(TELECOM_URL, timeout=3)
    if health_telecom and health_telecom.get("model") == model_choice:
        return TELECOM_URL, "Télécom Paris"

    health_dataia = _check_health(DATAIA25_URL, timeout=3)
    if health_dataia and health_dataia.get("model") == model_choice:
        log.info(f"[provision] Fast path: DataIA25 already serving {model_choice}")
        return DATAIA25_URL, "DataIA25"

    log.info(
        f"[provision] No cluster serving {model_choice} — telecom={health_telecom is not None} dataia25={health_dataia is not None}"
    )

    # 2) Determine partition order for Télécom
    partitions_to_try = []
    if fits.get("P100"):
        partitions_to_try.append("P100")
    if fits.get("3090"):
        partitions_to_try.append("3090")

    # 3) Try Télécom partitions
    log.info(f"[provision] Télécom partitions to try: {partitions_to_try}")
    if partitions_to_try:
        progress(0.05, desc="Connexion à Télécom Paris...")
        vpn_ok = _ensure_vpn_and_gateway(progress, pct_start=0.05)
        if vpn_ok:
            for partition in partitions_to_try:
                progress(0.12, desc=f"Vérification partition {partition}...")
                if _check_slurm_partition_free(partition):
                    ok = _provision_telecom(
                        model_choice, partition, progress, pct_start=0.15
                    )
                    if ok:
                        return TELECOM_URL, f"Télécom Paris ({partition})"

            # No partition had free GPUs — wait FAILOVER_WAIT_S then try dataia25
            progress(0.5, desc=f"Télécom Paris occupé, attente {FAILOVER_WAIT_S}s...")
            time.sleep(FAILOVER_WAIT_S)

            # Re-check after wait
            for partition in partitions_to_try:
                if _check_slurm_partition_free(partition):
                    ok = _provision_telecom(
                        model_choice, partition, progress, pct_start=0.55
                    )
                    if ok:
                        return TELECOM_URL, f"Télécom Paris ({partition})"

    # 4) Fallback to dataia25
    log.info(f"[provision] Falling back to DataIA25 for {model_choice}")
    progress(0.7, desc="Basculement vers DataIA25...")
    dataia25_ok = _provision_dataia25(model_choice, progress)
    if dataia25_ok:
        log.info(f"[provision] DataIA25 provisioning succeeded for {model_choice}")
        return DATAIA25_URL, "DataIA25"

    log.error(f"[provision] ALL clusters failed for {model_choice}")
    return None, "Aucun cluster disponible"


# ─── Generation logic ──────────────────────────────────────────────────────


def generate_bt(
    mission: str,
    use_grammar: bool,
    model_choice: str,
    progress=gr.Progress(),
) -> tuple[str, str, str, str]:
    """Generate BT XML via smart failover. Returns (xml, validation_html, status, bt_viz_html)."""
    if not mission.strip():
        return "", "", "Veuillez entrer une mission.", ""

    # Resolve model
    if not model_choice or model_choice == "auto":
        model_choice = "mistral-7b-merged-fp16"

    # Smart provisioning with failover
    target_url, cluster_label = _smart_provision(model_choice, progress)
    if target_url is None:
        return "", "", f"Erreur : {cluster_label}", ""

    payload = {"mission": mission, "use_grammar": use_grammar}
    if model_choice and model_choice != "auto":
        payload["model"] = model_choice

    progress(0.95, desc=f"Génération en cours sur {cluster_label}...")
    try:
        r = httpx.post(
            f"{target_url}/generate",
            json=payload,
            timeout=httpx.Timeout(timeout=MAX_TIME_S),
        )
        r.raise_for_status()
        result = r.json()
    except Exception as e:
        return "", "", f"Erreur cluster : {e}", ""

    xml = result.get("xml", "")
    valid = result.get("valid", False)
    score = result.get("score", 0)
    errors = result.get("errors", [])
    warnings = result.get("warnings", [])
    gen_time = result.get("generation_time_s", 0)

    now = datetime.now()
    append_history(
        {
            "mode": "generate",
            "mission": mission,
            "xml": xml,
            "score": round(score, 2),
            "valid": valid,
            "backend": "cluster",
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
        }
    )

    model_used = result.get("model", "unknown")
    model_label = MODEL_LABELS.get(model_used, model_used)
    port_issues = validate_ports(xml) if xml else []
    validation_html = _build_validation_html(
        valid, score, errors, warnings, port_issues
    )
    status = f"{'Valide' if valid else 'Invalide'} | Score: {score:.1f} | {gen_time}s | {model_label} | {cluster_label}"

    return xml, validation_html, status, render_bt_full_html(xml)


# ─── Validation logic ──────────────────────────────────────────────────────


def validate_xml_input(xml_str: str) -> tuple[str, str, str]:
    if not xml_str.strip():
        return "", "Veuillez coller du XML à valider.", ""

    vr = validate_bt(xml_str)
    port_issues = validate_ports(xml_str)

    now = datetime.now()
    append_history(
        {
            "mode": "validate",
            "mission": None,
            "xml": xml_str,
            "score": round(vr.score, 2),
            "valid": vr.valid,
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
        }
    )

    validation_html = _build_validation_html(
        vr.valid, vr.score, vr.errors, vr.warnings, port_issues
    )
    status = f"{'Valide' if vr.valid else 'Invalide'} | Score: {vr.score:.1f}"

    return validation_html, status, render_bt_full_html(xml_str)


# ─── Validation display helper ─────────────────────────────────────────────


def _build_validation_html(
    valid: bool,
    score: float,
    errors: list,
    warnings: list,
    port_issues: list | None = None,
) -> str:
    badge_color = "#22c55e" if valid else "#ef4444"
    badge_text = "VALIDE" if valid else "INVALIDE"

    if score >= 0.9:
        bar_color = "#22c55e"
    elif score >= 0.7:
        bar_color = "#eab308"
    else:
        bar_color = "#ef4444"

    html = f"""
    <div style="font-family: system-ui, sans-serif;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
            <span style="background: {badge_color}; color: white; padding: 4px 12px;
                         border-radius: 6px; font-weight: bold; font-size: 14px;">
                {badge_text}
            </span>
            <span style="font-size: 18px; font-weight: bold;">Score: {score:.1f}</span>
        </div>
        <div style="background: #374151; border-radius: 8px; height: 12px; overflow: hidden; margin-bottom: 16px;">
            <div style="background: {bar_color}; height: 100%; width: {score * 100:.0f}%;
                        transition: width 0.3s;"></div>
        </div>
    """

    if errors:
        html += '<div style="margin-bottom: 10px;">'
        for e in errors:
            html += f'<div style="color: #ef4444; padding: 4px 0;">&#x2716; {e}</div>'
        html += "</div>"

    if warnings:
        html += '<div style="margin-bottom: 10px;">'
        for w in warnings:
            html += f'<div style="color: #eab308; padding: 4px 0;">&#x26A0; {w}</div>'
        html += "</div>"

    if not errors and not warnings:
        html += '<div style="color: #22c55e; padding: 4px 0;">&#x2714; L1+L2+L3 — aucun problème</div>'

    # L4 Port validation
    if port_issues is not None:
        if port_issues:
            html += (
                '<div style="margin-top: 12px; padding: 8px; background: #422006; '
                'border: 1px solid #92400e; border-radius: 6px;">'
                f'<div style="color: #fbbf24; font-weight: 600; margin-bottom: 4px;">'
                f"⚠ L4 Ports — {len(port_issues)} issue(s)</div>"
            )
            for iss in port_issues[:10]:
                iss_escaped = iss.replace("<", "&lt;").replace(">", "&gt;")
                html += f'<div style="color: #fde68a; padding: 2px 0; font-size: 13px;">{iss_escaped}</div>'
            if len(port_issues) > 10:
                html += f'<div style="color: #9ca3af;">… +{len(port_issues) - 10} autres</div>'
            html += "</div>"
        else:
            html += (
                '<div style="margin-top: 12px; padding: 8px; background: #052e16; '
                'border: 1px solid #166534; border-radius: 6px;">'
                '<span style="color: #4ade80; font-weight: 600;">'
                "✓ L4 Ports — OK</span></div>"
            )

    html += "</div>"
    return html


# ─── History display ───────────────────────────────────────────────────────


def history_to_dataframe():
    """Return history as a list of rows for gr.Dataframe, plus raw entries."""
    import pandas as pd

    entries = load_history()
    if not entries:
        return pd.DataFrame(
            columns=["Type", "Mission", "Score", "Résultat", "Date", "Heure"]
        )
    rows = []
    for e in entries:
        rows.append(
            [
                "Génération" if e.get("mode") == "generate" else "Validation",
                e.get("mission") or "—",
                e.get("score", 0),
                "✅ Valide" if e.get("valid") else "❌ Invalide",
                e.get("date", ""),
                e.get("time", ""),
            ]
        )
    return pd.DataFrame(
        rows, columns=["Type", "Mission", "Score", "Résultat", "Date", "Heure"]
    )


def get_xml_for_row(evt: gr.SelectData) -> str:
    entries = load_history()
    idx = evt.index[0]
    if 0 <= idx < len(entries):
        return entries[idx].get("xml", "")
    return ""


# ─── Cluster status ───────────────────────────────────────────────────────

CLUSTER_LABELS = {
    "telecom-paris": "Télécom Paris",
    "dataia25": "DataIA25",
}


def _get_cluster_info_from(url: str) -> dict | None:
    """Fetch /health from a cluster URL, return dict or None."""
    try:
        r = httpx.get(f"{url}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _get_cluster_info() -> dict | None:
    """Fetch /health from active cluster."""
    return _get_cluster_info_from(CLUSTER_URL)


def _get_all_clusters_info() -> list[tuple[str, str, dict | None]]:
    """Return [(label, url, health_dict_or_None), ...] for all clusters."""
    results = []
    for label, url in [("telecom-paris", TELECOM_URL), ("dataia25", DATAIA25_URL)]:
        info = _get_cluster_info_from(url)
        results.append((label, url, info))
    return results


# ─── Cached cluster status (background thread) ────────────────────────────

_cached_banner_html: str = ""
_banner_lock = threading.Lock()


def _banner_updater():
    """Background thread: refresh cluster status HTML every 5s."""
    global _cached_banner_html
    while True:
        try:
            html = _build_cluster_banner_html()
        except Exception:
            html = ""
        with _banner_lock:
            _cached_banner_html = html
        time.sleep(5)


def get_cached_banner_html() -> str:
    with _banner_lock:
        return _cached_banner_html


_banner_thread = threading.Thread(target=_banner_updater, daemon=True)
_banner_thread.start()


def _get_available_models() -> list[str]:
    """Fetch available model keys from active cluster."""
    try:
        r = httpx.get(f"{CLUSTER_URL}/models", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return list(data.get("available", {}).keys())
    except Exception:
        pass
    return list(MODEL_LABELS.keys())


def _ensure_dataia25_tunnel() -> bool:
    """Ensure SSH tunnel to dataia25 is up (localhost:8081 -> dataia25:8080)."""
    info = _get_cluster_info_from(DATAIA25_URL)
    if info:
        return True

    _run(f"pkill -f 'ssh.*-L 8081:localhost:{DATAIA25_PORT}' 2>/dev/null")
    time.sleep(0.5)

    rc, _ = _run(
        f"ssh -fN -L 8081:localhost:{DATAIA25_PORT} dataia25 "
        f"-o ConnectTimeout=5 -o ServerAliveInterval=30 "
        f"-o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes",
        timeout=10,
    )
    if rc != 0:
        _run(
            f"nohup ssh -N -L 8081:{DATAIA25_HOST}:{DATAIA25_PORT} localhost "
            f"-o ServerAliveInterval=30 -o ServerAliveCountMax=3 "
            f"-o ExitOnForwardFailure=yes </dev/null >/dev/null 2>&1 &",
            timeout=5,
        )
        time.sleep(2)

    for _ in range(3):
        if _get_cluster_info_from(DATAIA25_URL):
            return True
        time.sleep(1)
    return False


def _provision_dataia25(
    model_choice: str = "mistral-7b-merged-fp16", progress=gr.Progress()
) -> bool:
    """Full auto-provisioning: tunnel + start server if needed + wait for ready."""
    dataia25_key = MODEL_DATAIA25_KEY.get(model_choice, "mistral_7b")

    # 1) Fast path: server already running with right model and tunnel up
    health = _check_health(DATAIA25_URL, timeout=3)
    if health and health.get("model") == model_choice:
        return True

    # Also check via SSH in case tunnel is down but server is running
    if not health:
        rc, health_str = _run(
            f"ssh -o ConnectTimeout=5 dataia25 "
            f"'curl -s --max-time 3 http://localhost:{DATAIA25_PORT}/health'",
            timeout=10,
        )
        if '"status":"ok"' in health_str:
            import json as _json

            try:
                health = _json.loads(health_str)
            except Exception:
                health = None

    if health and health.get("model") == model_choice:
        # Server ready but tunnel down — just fix the tunnel
        progress(0.85, desc="Reconnexion tunnel DataIA25...")
        if _ensure_dataia25_tunnel():
            return True

    # If server running with wrong model, kill and restart
    if health and health.get("model") != model_choice:
        progress(0.1, desc=f"Changement de modèle DataIA25 → {dataia25_key}...")
        _run(
            "ssh -o ConnectTimeout=5 dataia25 "
            "'fuser -k 8080/tcp 2>/dev/null; sleep 1; "
            "pkill -9 -f run_serve_hf 2>/dev/null; "
            'pkill -9 -f "python3.*FastAPI" 2>/dev/null; '
            'pkill -9 -f "python3.*uvicorn" 2>/dev/null; '
            'pkill -9 -f "python3.*-c.*import" 2>/dev/null\'',
            timeout=15,
        )
        # Wait for CUDA VRAM to be released (GPU processes take time to free)
        time.sleep(8)
        # Verify GPU is free
        rc, gpu_mem = _run(
            "ssh -o ConnectTimeout=3 dataia25 "
            "'nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits'",
            timeout=8,
        )
        try:
            mem_mb = int(gpu_mem.strip())
            if mem_mb > 1000:
                log.warning(
                    f"[dataia25] GPU still has {mem_mb} MiB in use after kill, retrying kill..."
                )
                _run(
                    "ssh -o ConnectTimeout=5 dataia25 "
                    "'fuser -k 8080/tcp 2>/dev/null; "
                    'pkill -9 -f "python3.*-c.*import" 2>/dev/null\'',
                    timeout=10,
                )
                time.sleep(10)
        except (ValueError, AttributeError):
            pass

    # 2) SSH to dataia25 and check if serve process is running
    progress(0.1, desc="Vérification du serveur DataIA25...")
    rc, out = _run(
        "ssh -o ConnectTimeout=5 dataia25 "
        "'pgrep -af run_serve_hf | grep -v pgrep || echo NO_PROCESS'",
        timeout=10,
    )
    server_running = rc == 0 and "NO_PROCESS" not in out

    # 3) Start the serve script if not running
    if not server_running:
        model_label = MODEL_LABELS.get(model_choice, dataia25_key)
        progress(0.15, desc=f"Démarrage {model_label} fp16 sur DataIA25...")
        # ssh -f: fork after auth so parent exits immediately.
        # _run_bg (DEVNULL): prevents pipe deadlock that killed capture_output.
        rc = _run_bg(
            f"ssh -f -o ConnectTimeout=5 dataia25 "
            f"'cd ~/nav4rail && nohup bash run_serve_hf_dataia25.sh {dataia25_key} "
            f"> serve_{dataia25_key}.log 2>&1 </dev/null &'",
            timeout=15,
        )
        if rc != 0:
            log.warning(f"[dataia25] ssh -f launch failed (rc={rc}), retrying...")
            rc = _run_bg(
                f"ssh -o ConnectTimeout=5 dataia25 "
                f"'cd ~/nav4rail && setsid nohup bash run_serve_hf_dataia25.sh {dataia25_key} "
                f"> serve_{dataia25_key}.log 2>&1 </dev/null &'",
                timeout=20,
            )
            if rc != 0:
                return False
        time.sleep(3)  # give the script time to parse args and start

    # 4) Wait for the server to become ready
    #    First-time download can take ~400-600s (18GB model + LoRA merge + GPU load)
    max_wait = 120  # 120 × 5s = 600s max
    progress(0.2, desc="Chargement du modèle sur le GPU...")
    for attempt in range(max_wait):
        pct = 0.2 + (attempt / max_wait) * 0.65

        # Read last log line for progress feedback
        rc_log, log_tail = _run(
            f"ssh -o ConnectTimeout=3 dataia25 "
            f"'tail -1 ~/nav4rail/serve_{dataia25_key}.log 2>/dev/null'",
            timeout=8,
        )
        if "Downloading" in log_tail or "Fetching" in log_tail:
            progress(pct, desc=f"Téléchargement du modèle... ({(attempt + 1) * 5}s)")
        elif "LoRA" in log_tail or "merge" in log_tail.lower():
            progress(pct, desc=f"Fusion LoRA en mémoire... ({(attempt + 1) * 5}s)")
        elif "Moving model" in log_tail or "to GPU" in log_tail:
            progress(pct, desc=f"Transfert vers le GPU... ({(attempt + 1) * 5}s)")
        else:
            progress(pct, desc=f"Chargement du modèle... ({(attempt + 1) * 5}s)")

        time.sleep(5)

        # Check health directly via SSH (tunnel may not be up yet)
        rc, health_str = _run(
            f"ssh -o ConnectTimeout=3 dataia25 "
            f"'curl -s --max-time 3 http://localhost:{DATAIA25_PORT}/health'",
            timeout=10,
        )
        if (
            '"status":"ok"' in health_str
            and f'"model":"{model_choice}"' in health_str.replace(" ", "")
        ):
            break

        # Every 30s, check if process is still alive (fail fast on crash)
        if attempt > 0 and attempt % 6 == 0:
            rc_proc, proc_out = _run(
                "ssh -o ConnectTimeout=3 dataia25 "
                "'pgrep -af run_serve_hf | grep -v pgrep || echo NO_PROCESS'",
                timeout=8,
            )
            if "NO_PROCESS" in proc_out:
                return False
    else:
        return False

    # 5) Ensure tunnel is up
    progress(0.9, desc="Ouverture du tunnel SSH...")
    if _ensure_dataia25_tunnel():
        progress(1.0, desc="DataIA25 prêt !")
        return True

    return False


def _build_cluster_banner_html() -> str:
    """Build HTML banner (called by background thread only)."""
    clusters = _get_all_clusters_info()

    items_html = ""
    for key, url, info in clusters:
        label = CLUSTER_LABELS.get(key, key)
        if info:
            model_key = info.get("model", "unknown")
            model_label = MODEL_LABELS.get(model_key, model_key)
            gpu_name = info.get("gpu_name", "GPU")
            backend = info.get("backend", "")
            backend_tag = f" ({backend})" if backend else ""
            dot = "🟢"
            color = "#22c55e"

            detail = f"{gpu_name}{backend_tag} — {model_label}"
            remaining = info.get("remaining_s")
            if remaining is not None:
                hrs = remaining // 3600
                mins = (remaining % 3600) // 60
                secs = remaining % 60
                if hrs > 0:
                    detail += f" — ⏱ {hrs}h{mins:02d}m{secs:02d}s"
                else:
                    detail += f" — ⏱ {mins}m{secs:02d}s"

            items_html += (
                f'<div style="display:inline-flex;align-items:center;gap:8px;padding:4px 12px;'
                f'background:#052e16;border:1px solid #166534;border-radius:8px;">'
                f'<span style="font-size:14px;">{dot}</span>'
                f'<span style="color:{color};font-weight:600;font-size:13px;">{label}</span>'
                f'<span style="color:#9ca3af;font-size:12px;">{detail}</span>'
                f"</div> "
            )
        else:
            dot = "🔴"
            color = "#6b7280"
            items_html += (
                f'<div style="display:inline-flex;align-items:center;gap:8px;padding:4px 12px;'
                f'background:#1f2937;border:1px solid #374151;border-radius:8px;">'
                f'<span style="font-size:14px;">{dot}</span>'
                f'<span style="color:{color};font-weight:600;font-size:13px;">{label}</span>'
                f'<span style="color:#6b7280;font-size:12px;">hors ligne</span>'
                f"</div> "
            )

    return (
        f'<div id="cluster-banner" style="display:flex;gap:10px;flex-wrap:wrap;'
        f'min-height:36px;align-items:center;">'
        f"{items_html}"
        f"</div>"
    )


def get_cluster_status_html() -> str:
    """Return cached banner HTML (instant, never blocks)."""
    return get_cached_banner_html()


# ─── Build Gradio UI ───────────────────────────────────────────────────────

THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

CSS = """
    .main-title { text-align: center; margin-bottom: 4px; }
    .subtitle { text-align: center; color: #9ca3af; margin-top: 0; font-size: 14px; }
    footer { display: none !important; }
    .copy-btn {
        background: #374151; border: 1px solid #4b5563; color: #d1d5db;
        padding: 3px 10px; border-radius: 4px; font-size: 11px; cursor: pointer;
        float: right; margin-top: -2px;
    }
    .copy-btn:hover { background: #4b5563; }
    #history-df .wrap { max-height: 280px; overflow-y: auto; }
    /* Fix: prevent banner refresh from shifting layout */
    #cluster-status-box { min-height: 44px; }
    #cluster-status-box .prose { min-height: 36px; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="NAV4RAIL BT Generator", theme=THEME, css=CSS) as app:
        gr.Markdown("# NAV4RAIL — Behavior Tree Generator", elem_classes="main-title")
        gr.Markdown(
            "Génération via GPU cluster (Télécom Paris / DataIA25) | Validation locale",
            elem_classes="subtitle",
        )

        cluster_status = gr.HTML(
            value=get_cluster_status_html(),
            elem_id="cluster-status-box",
        )
        # Periodic refresh using gr.Timer (Gradio 5.x compatible)
        _cluster_timer = gr.Timer(value=3)
        _cluster_timer.tick(fn=get_cluster_status_html, outputs=[cluster_status])

        with gr.Tabs():
            # ─── Tab 1: Génération ──────────────────────────────────────
            with gr.Tab("Génération", id="generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        mission_input = gr.Textbox(
                            label="Mission (langage naturel)",
                            placeholder="Ex: Inspection des rails entre le km 12 et le km 45",
                            lines=3,
                        )
                        with gr.Row():
                            model_selector = gr.Dropdown(
                                choices=[
                                    (label, key) for key, label in MODEL_LABELS.items()
                                ],
                                value="mistral-7b-merged-fp16",
                                label="Modèle",
                                scale=3,
                            )
                            use_grammar = gr.Checkbox(
                                label="Grammaire GBNF",
                                value=False,
                                scale=1,
                                visible=False,
                            )
                        generate_btn = gr.Button(
                            "Générer le Behavior Tree",
                            variant="primary",
                            size="lg",
                        )
                        gen_status = gr.Markdown("")

                        gr.Markdown("### Exemples de missions")
                        examples = gr.Dataset(
                            components=[gr.Textbox(visible=False)],
                            samples=[[m] for m in TEST_MISSIONS],
                            label="Cliquez pour utiliser un exemple",
                        )

                    with gr.Column(scale=1):
                        xml_output = gr.Code(
                            label="XML généré",
                            language="html",
                            lines=20,
                        )
                        gen_validation = gr.HTML(label="Validation")

                gen_bt_viz = gr.HTML(label="Behavior Tree", elem_id="gen-bt-viz")

                generate_btn.click(
                    fn=generate_bt,
                    inputs=[
                        mission_input,
                        use_grammar,
                        model_selector,
                    ],
                    outputs=[xml_output, gen_validation, gen_status, gen_bt_viz],
                )

                examples.click(
                    fn=lambda x: x[0],
                    inputs=[examples],
                    outputs=[mission_input],
                )

            # ─── Tab 2: Validation ──────────────────────────────────────
            with gr.Tab("Validation", id="validate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        xml_input = gr.Code(
                            label="XML à valider",
                            language="html",
                            lines=15,
                        )
                        validate_btn = gr.Button(
                            "Valider le XML",
                            variant="primary",
                            size="lg",
                        )
                        val_status = gr.Markdown("")

                    with gr.Column(scale=1):
                        val_result = gr.HTML(label="Résultat de validation")

                val_bt_viz = gr.HTML(label="Behavior Tree", elem_id="val-bt-viz")

                validate_btn.click(
                    fn=validate_xml_input,
                    inputs=[xml_input],
                    outputs=[val_result, val_status, val_bt_viz],
                )

            # ─── Tab 3: Historique ──────────────────────────────────────
            with gr.Tab("Historique", id="history"):
                refresh_btn = gr.Button("↻ Rafraîchir", size="sm")
                history_df = gr.Dataframe(
                    value=history_to_dataframe,
                    interactive=False,
                    label="Entrées (cliquez une ligne pour voir le XML)",
                    elem_id="history-df",
                )
                history_xml = gr.Code(
                    label="XML — cliquez une ligne ci-dessus",
                    language="html",
                    lines=15,
                    interactive=False,
                )
                refresh_btn.click(fn=history_to_dataframe, outputs=[history_df])
                history_df.select(fn=get_xml_for_row, outputs=[history_xml])

            # ─── Tab 4: Dataset Explorer ────────────────────────────────
            with gr.Tab("Dataset proxy", id="dataset"):
                # --- Pipeline explanation ---
                with gr.Accordion("Comment ce dataset a été généré", open=False):
                    gr.Markdown(PIPELINE_MD)

                # --- Statistics dashboard ---
                with gr.Accordion("Statistiques du dataset", open=True):
                    gr.HTML(value=_build_stats_html(DATASET_STATS))

                gr.Markdown("---")

                with gr.Tabs():
                    # ──── Sub-tab: Explorateur ──────────────────────────
                    with gr.Tab("Explorateur", id="ds-explore"):
                        # --- Filters ---
                        with gr.Row():
                            ds_cat_filter = gr.Dropdown(
                                choices=["Toutes"] + [f"{c}" for c in _CAT_ORDER],
                                value="Toutes",
                                label="Catégorie",
                                scale=2,
                            )
                            ds_score_filter = gr.Radio(
                                choices=["Tous", "1.0", "0.9"],
                                value="Tous",
                                label="Score",
                                scale=1,
                            )
                            ds_port_filter = gr.Radio(
                                choices=["Tous", "L4 OK", "L4 Issues"],
                                value="Tous",
                                label="Ports (L4)",
                                scale=1,
                            )
                            ds_search = gr.Textbox(
                                label="Recherche dans la mission",
                                placeholder="Ex: km 42, ballast, simulation…",
                                scale=2,
                            )
                            ds_filter_btn = gr.Button(
                                "Filtrer", variant="primary", scale=0
                            )

                        # --- State ---
                        ds_filtered_state = gr.State(
                            [(i, s) for i, s in enumerate(DATASET)]
                        )
                        ds_idx_state = gr.State(0)

                        # --- Navigator ---
                        with gr.Row():
                            ds_prev_btn = gr.Button("◀ Précédent", size="sm", scale=0)
                            ds_slider = gr.Slider(
                                minimum=1,
                                maximum=max(len(DATASET), 1),
                                value=1,
                                step=1,
                                label=f"Sample (1–{len(DATASET)})",
                                scale=3,
                            )
                            ds_next_btn = gr.Button("Suivant ▶", size="sm", scale=0)

                        ds_status = gr.Markdown("Chargement…")

                        # --- Sample display ---
                        ds_mission_html = gr.HTML()
                        with gr.Row():
                            ds_score_html = gr.HTML()
                        ds_port_html = gr.HTML()

                        with gr.Row():
                            with gr.Column(scale=1):
                                ds_xml = gr.Code(
                                    label="XML du Behavior Tree",
                                    language="html",
                                    lines=18,
                                )
                            with gr.Column(scale=1):
                                with gr.Accordion(
                                    "Prompt envoyé au LLM 70B", open=False
                                ):
                                    ds_prompt = gr.Code(
                                        label="Prompt complet",
                                        language=None,
                                        lines=18,
                                    )

                        ds_bt_viz = gr.HTML(
                            label="Behavior Tree",
                            elem_id="ds-bt-viz",
                        )

                        # --- Wire filter callback ---
                        _ds_filter_inputs = [
                            ds_cat_filter,
                            ds_score_filter,
                            ds_search,
                            ds_port_filter,
                        ]
                        _ds_filter_outputs = [
                            ds_filtered_state,
                            ds_idx_state,
                            ds_slider,
                            ds_status,
                            ds_mission_html,
                            ds_score_html,
                            ds_port_html,
                            ds_xml,
                            ds_prompt,
                            ds_bt_viz,
                        ]
                        ds_filter_btn.click(
                            fn=ds_apply_filters,
                            inputs=_ds_filter_inputs,
                            outputs=_ds_filter_outputs,
                        )
                        ds_search.submit(
                            fn=ds_apply_filters,
                            inputs=_ds_filter_inputs,
                            outputs=_ds_filter_outputs,
                        )

                        # --- Wire slider navigation ---
                        ds_slider.change(
                            fn=ds_navigate,
                            inputs=[ds_filtered_state, ds_slider],
                            outputs=[
                                ds_idx_state,
                                ds_status,
                                ds_mission_html,
                                ds_score_html,
                                ds_port_html,
                                ds_xml,
                                ds_prompt,
                                ds_bt_viz,
                            ],
                        )

                        # --- Wire prev/next buttons ---
                        ds_prev_btn.click(
                            fn=ds_prev,
                            inputs=[ds_filtered_state, ds_idx_state],
                            outputs=[
                                ds_idx_state,
                                ds_slider,
                                ds_status,
                                ds_mission_html,
                                ds_score_html,
                                ds_port_html,
                                ds_xml,
                                ds_prompt,
                                ds_bt_viz,
                            ],
                        )
                        ds_next_btn.click(
                            fn=ds_next,
                            inputs=[ds_filtered_state, ds_idx_state],
                            outputs=[
                                ds_idx_state,
                                ds_slider,
                                ds_status,
                                ds_mission_html,
                                ds_score_html,
                                ds_port_html,
                                ds_xml,
                                ds_prompt,
                                ds_bt_viz,
                            ],
                        )

                        # --- Initial load: display first sample ---
                        app.load(
                            fn=ds_apply_filters,
                            inputs=_ds_filter_inputs,
                            outputs=_ds_filter_outputs,
                        )

                    # ──── Sub-tab: Comparateur ──────────────────────────
                    with gr.Tab("Comparateur", id="ds-compare"):
                        gr.Markdown(
                            "Sélectionnez deux samples par leur numéro pour les comparer côte à côte."
                        )

                        with gr.Row():
                            with gr.Column(scale=1):
                                cmp_num_a = gr.Number(
                                    label="Sample A (n°)",
                                    value=1,
                                    minimum=1,
                                    maximum=max(len(DATASET), 1),
                                    precision=0,
                                )
                            with gr.Column(scale=1):
                                cmp_num_b = gr.Number(
                                    label="Sample B (n°)",
                                    value=min(2, len(DATASET)),
                                    minimum=1,
                                    maximum=max(len(DATASET), 1),
                                    precision=0,
                                )
                            cmp_btn = gr.Button("Comparer", variant="primary", scale=0)

                        # --- Diff summary ---
                        cmp_diff_html = gr.HTML()

                        # --- Unified XML diff ---
                        cmp_diff_xml = gr.HTML()

                        # --- Side by side BT viz ---
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1):
                                cmp_card_a = gr.HTML()
                                cmp_viz_a = gr.HTML(label="BT — Sample A")
                            with gr.Column(scale=1):
                                cmp_card_b = gr.HTML()
                                cmp_viz_b = gr.HTML(label="BT — Sample B")

                        # --- Wire compare ---
                        _cmp_outputs = [
                            cmp_card_a,
                            cmp_viz_a,
                            cmp_card_b,
                            cmp_viz_b,
                            cmp_diff_html,
                            cmp_diff_xml,
                        ]
                        cmp_btn.click(
                            fn=cmp_load_both,
                            inputs=[cmp_num_a, cmp_num_b],
                            outputs=_cmp_outputs,
                        )
                        # Also load on number change
                        cmp_num_a.change(
                            fn=cmp_load_both,
                            inputs=[cmp_num_a, cmp_num_b],
                            outputs=_cmp_outputs,
                        )
                        cmp_num_b.change(
                            fn=cmp_load_both,
                            inputs=[cmp_num_a, cmp_num_b],
                            outputs=_cmp_outputs,
                        )
                        # Initial load
                        app.load(
                            fn=cmp_load_both,
                            inputs=[cmp_num_a, cmp_num_b],
                            outputs=_cmp_outputs,
                        )

    return app


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=8778,
        root_path=ROOT_PATH,
    )
