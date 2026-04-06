"""
NAV4RAIL BT Generator — Gradio Web Application.

Inference via Télécom Paris GPU cluster (on-demand SLURM provisioning via SSH).
Shares cluster provisioning logic with the FastAPI webapp.

Usage:
    python app.py
"""

import asyncio
import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import gradio as gr
import httpx

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

CLUSTER_URL = "http://localhost:8080"
MAX_TIME_S = 900
GPU_HOST = "gpu"
REMOTE_USER = "blepourt-25"
REMOTE_MODEL = (
    f"/home/infres/{REMOTE_USER}/code/nav4rail_finetune/nav4rail-mistral-7b-q4_k_m.gguf"
)
CLUSTER_PORT = 8080

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
        "nodes": {"min": min(nodes), "avg": mean(nodes), "med": median(nodes), "max": max(nodes)},
        "subtrees": {"min": min(subtrees), "avg": mean(subtrees), "med": median(subtrees), "max": max(subtrees)},
        "leaf_nodes": {"min": min(leaf), "avg": mean(leaf), "med": median(leaf), "max": max(leaf)},
        "xml_size": {"min": min(sizes), "avg": mean(sizes), "med": median(sizes), "max": max(sizes)},
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
            f'Score {score_val} — {count} samples ({count/total*100:.1f}%)</span>'
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
                <div style="font-size:28px;font-weight:bold;color:{'#4ade80' if stats.get('port_ok', 0) == total else '#fbbf24'};">{stats.get('port_ok', 0)}/{total}</div>
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


def _render_sample(sample: dict, idx_in_filtered: int, total_filtered: int) -> tuple:
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

    status = (
        f"**Sample {idx_in_filtered + 1} / {total_filtered}** — "
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
        f'{nodes} nœuds · {subtrees} sous-arbres · {sample.get("xml_size", 0)} chars'
        f"</span>"
    )

    # Port validation (L4)
    port_issues = sample.get("port_issues", [])
    if port_issues:
        port_html = (
            '<div style="margin-top:8px;padding:8px;background:#422006;'
            'border:1px solid #92400e;border-radius:6px;font-size:12px;">'
            f'<span style="color:#fbbf24;font-weight:600;">'
            f'⚠ L4 Ports — {len(port_issues)} issue(s)</span><br>'
        )
        for iss in port_issues[:5]:
            iss_escaped = iss.replace("<", "&lt;").replace(">", "&gt;")
            port_html += f'<span style="color:#fde68a;">{iss_escaped}</span><br>'
        if len(port_issues) > 5:
            port_html += (
                f'<span style="color:#9ca3af;">'
                f"… +{len(port_issues) - 5} autres</span>"
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
        empty = ("**0 résultats**", "<p>Aucun sample ne correspond aux filtres.</p>", "", "", "", "", "")
        return filtered, 0, gr.update(minimum=1, maximum=1, value=1, label="Sample 0/0"), *empty

    _, sample = filtered[0]
    display = _render_sample(sample, 0, len(filtered))
    slider_update = gr.update(
        minimum=1, maximum=len(filtered), value=1,
        label=f"Sample (1–{len(filtered)})",
    )
    return filtered, 0, slider_update, *display


def ds_navigate(filtered, slider_val):
    """Navigate to sample at slider position."""
    if not filtered:
        return 0, "**0 résultats**", "", "", "", "", "", ""
    idx = max(0, min(int(slider_val) - 1, len(filtered) - 1))
    _, sample = filtered[idx]
    display = _render_sample(sample, idx, len(filtered))
    return idx, *display


def ds_prev(filtered, current_idx):
    """Go to previous sample."""
    if not filtered:
        return 0, gr.update(), "**0 résultats**", "", "", "", "", "", ""
    idx = max(0, current_idx - 1)
    _, sample = filtered[idx]
    display = _render_sample(sample, idx, len(filtered))
    return idx, gr.update(value=idx + 1), *display


def ds_next(filtered, current_idx):
    """Go to next sample."""
    if not filtered:
        return 0, gr.update(), "**0 résultats**", "", "", "", "", "", ""
    idx = min(len(filtered) - 1, current_idx + 1)
    _, sample = filtered[idx]
    display = _render_sample(sample, idx, len(filtered))
    return idx, gr.update(value=idx + 1), *display


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


# ─── Cluster provisioning (sync version for Gradio) ────────────────────────


def _check_cluster() -> bool:
    try:
        r = httpx.get(f"{CLUSTER_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _provision_cluster(progress=gr.Progress()) -> bool:
    """Ensure VPN + route + SLURM job + SSH tunnel are up."""
    if _check_cluster():
        return True

    progress(0.1, desc="Vérification du VPN...")
    rc, _ = _run("nmcli -t -f NAME connection show --active | grep -q telecom-paris")
    if rc != 0:
        progress(0.15, desc="Activation du VPN Télécom Paris...")
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
    progress(0.2, desc="Connexion à la passerelle GPU...")
    rc, _ = _run(f"ssh -o ConnectTimeout=5 {GPU_HOST} echo ok", timeout=10)
    if rc != 0:
        return False

    # SLURM job
    progress(0.3, desc="Vérification du job SLURM...")
    rc, running = _run(
        f"ssh -o ConnectTimeout=5 {GPU_HOST} "
        f"'squeue -u $(whoami) -h -t R -n nav4rail-serve --format=\"%j\"'",
        timeout=15,
    )
    if not running.strip():
        _run(
            f"ssh {GPU_HOST} 'scancel -u $(whoami) -n nav4rail-serve -t PD 2>/dev/null'",
            timeout=10,
        )
        progress(0.35, desc="Soumission du job GPU...")
        rc, job_out = _run(
            f"ssh {GPU_HOST} 'cd ~/nav4rail_serve && "
            f"MODEL_PATH={REMOTE_MODEL} sbatch --parsable job_serve_final.sh'",
            timeout=15,
        )
        job_id = job_out.strip()
        if rc != 0 or not job_id.isdigit():
            return False

        # Wait for GPU
        for attempt in range(18):
            progress(
                0.4 + attempt * 0.02,
                desc=f"En attente d'un GPU... ({(attempt + 1) * 10}s)",
            )
            time.sleep(10)
            rc, state = _run(
                f"ssh -o ConnectTimeout=5 {GPU_HOST} "
                f"'squeue -j {job_id} -h --format=\"%T\"'",
                timeout=10,
            )
            state = state.strip()
            if state == "RUNNING":
                progress(0.75, desc="Job GPU démarré ! Chargement du modèle...")
                time.sleep(8)
                break
            if state == "" or "FAILED" in state or "CANCELLED" in state:
                return False
        else:
            return False

        # Wait for server ready
        progress(0.8, desc="Chargement du modèle sur le GPU...")
        for _ in range(12):
            rc, health = _run(
                f"ssh -o ConnectTimeout=5 {GPU_HOST} "
                f"'curl -s --max-time 3 http://localhost:{CLUSTER_PORT}/health'",
                timeout=10,
            )
            if '"status":"ok"' in health:
                break
            time.sleep(5)

    # SSH tunnel
    _run(f"pkill -f 'ssh.*-L {CLUSTER_PORT}:localhost:{CLUSTER_PORT}' 2>/dev/null")
    time.sleep(1)
    progress(0.9, desc="Ouverture du tunnel SSH...")
    rc, _ = _run(
        f"ssh -fN -L {CLUSTER_PORT}:localhost:{CLUSTER_PORT} {GPU_HOST} "
        f"-o ServerAliveInterval=30 -o ServerAliveCountMax=3 "
        f"-o ExitOnForwardFailure=yes",
        timeout=10,
    )
    if rc != 0:
        _run(
            f"nohup ssh -N -L {CLUSTER_PORT}:localhost:{CLUSTER_PORT} {GPU_HOST} "
            f"-o ServerAliveInterval=30 -o ServerAliveCountMax=3 "
            f"-o ExitOnForwardFailure=yes </dev/null >/dev/null 2>&1 &",
            timeout=5,
        )
        time.sleep(3)

    # Final check
    for _ in range(6):
        if _check_cluster():
            progress(1.0, desc="Cluster GPU prêt !")
            return True
        time.sleep(2)

    return False


# ─── Generation logic ──────────────────────────────────────────────────────


def generate_bt(
    mission: str, use_grammar: bool, progress=gr.Progress()
) -> tuple[str, str, str, str]:
    """Generate BT XML via cluster. Returns (xml, validation_html, status, bt_viz_html)."""
    if not mission.strip():
        return "", "", "Veuillez entrer une mission.", ""

    # Provision cluster
    progress(0.05, desc="Connexion au cluster GPU...")
    cluster_ok = _provision_cluster(progress)

    if not cluster_ok:
        return "", "", "Cluster GPU inaccessible. Vérifiez le VPN et le cluster.", ""

    progress(0.95, desc="Génération en cours sur le GPU...")
    try:
        r = httpx.post(
            f"{CLUSTER_URL}/generate",
            json={"mission": mission, "use_grammar": use_grammar},
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

    validation_html = _build_validation_html(valid, score, errors, warnings)
    status = f"{'Valide' if valid else 'Invalide'} | Score: {score:.1f} | {gen_time}s | GPU cluster"

    return xml, validation_html, status, render_bt_html(xml)


# ─── Validation logic ──────────────────────────────────────────────────────


def validate_xml_input(xml_str: str) -> tuple[str, str, str]:
    if not xml_str.strip():
        return "", "Veuillez coller du XML à valider.", ""

    vr = validate_bt(xml_str)

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

    validation_html = _build_validation_html(vr.valid, vr.score, vr.errors, vr.warnings)
    status = f"{'Valide' if vr.valid else 'Invalide'} | Score: {vr.score:.1f}"

    return validation_html, status, render_bt_html(xml_str)


# ─── Validation display helper ─────────────────────────────────────────────


def _build_validation_html(
    valid: bool, score: float, errors: list, warnings: list
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


def get_cluster_status_html() -> str:
    ok = _check_cluster()
    if ok:
        color, dot, label = "#22c55e", "🟢", "GPU Cluster Télécom Paris — accessible"
    else:
        color, dot, label = (
            "#ef4444",
            "🔴",
            "GPU Cluster Télécom Paris — hors ligne (provisioning à la demande)",
        )
    return (
        f'<div style="display:flex;align-items:center;gap:8px;padding:6px 0;font-size:13px;">'
        f'<span style="font-size:16px;">{dot}</span>'
        f'<span style="color:{color};font-weight:600;">{label}</span>'
        f"</div>"
    )


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
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="NAV4RAIL BT Generator", theme=THEME, css=CSS) as app:
        gr.Markdown("# NAV4RAIL — Behavior Tree Generator", elem_classes="main-title")
        gr.Markdown(
            "Génération via le cluster GPU Télécom Paris | Validation locale",
            elem_classes="subtitle",
        )

        with gr.Row():
            cluster_status = gr.HTML(value=get_cluster_status_html)
            refresh_status_btn = gr.Button("↻", size="sm", scale=0)
        refresh_status_btn.click(fn=get_cluster_status_html, outputs=[cluster_status])

        with gr.Tabs():
            # ─── Tab 1: Génération ──────────────────────────────────────
            with gr.Tab("Génération", id="generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        mission_input = gr.Textbox(
                            label="Mission (langage naturel)",
                            placeholder="Ex: Navigue jusqu'au km 42 depuis le km 10",
                            lines=3,
                        )
                        use_grammar = gr.Checkbox(
                            label="Grammaire GBNF (décodage contraint)",
                            value=True,
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
                    inputs=[mission_input, use_grammar],
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
                gr.Markdown("### Explorateur de samples")

                # --- Filters ---
                with gr.Row():
                    ds_cat_filter = gr.Dropdown(
                        choices=["Toutes"] + [
                            f"{c}" for c in _CAT_ORDER
                        ],
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
                    ds_filter_btn = gr.Button("Filtrer", variant="primary", scale=0)

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
                        with gr.Accordion("Prompt envoyé au LLM 70B", open=False):
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
                _ds_filter_inputs = [ds_cat_filter, ds_score_filter, ds_search, ds_port_filter]
                _ds_filter_outputs = [
                    ds_filtered_state, ds_idx_state, ds_slider,
                    ds_status, ds_mission_html, ds_score_html,
                    ds_port_html, ds_xml, ds_prompt, ds_bt_viz,
                ]
                ds_filter_btn.click(
                    fn=ds_apply_filters,
                    inputs=_ds_filter_inputs,
                    outputs=_ds_filter_outputs,
                )
                # Also trigger on Enter in search box
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
                        ds_status, ds_mission_html, ds_score_html,
                        ds_port_html, ds_xml, ds_prompt, ds_bt_viz,
                    ],
                )

                # --- Wire prev/next buttons ---
                ds_prev_btn.click(
                    fn=ds_prev,
                    inputs=[ds_filtered_state, ds_idx_state],
                    outputs=[
                        ds_idx_state, ds_slider,
                        ds_status, ds_mission_html, ds_score_html,
                        ds_port_html, ds_xml, ds_prompt, ds_bt_viz,
                    ],
                )
                ds_next_btn.click(
                    fn=ds_next,
                    inputs=[ds_filtered_state, ds_idx_state],
                    outputs=[
                        ds_idx_state, ds_slider,
                        ds_status, ds_mission_html, ds_score_html,
                        ds_port_html, ds_xml, ds_prompt, ds_bt_viz,
                    ],
                )

                # --- Initial load: display first sample ---
                app.load(
                    fn=ds_apply_filters,
                    inputs=_ds_filter_inputs,
                    outputs=_ds_filter_outputs,
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
