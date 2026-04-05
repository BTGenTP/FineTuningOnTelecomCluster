"""
NAV4RAIL BT Generator — Gradio Web Application.

Inference via Télécom Paris GPU cluster (on-demand SLURM provisioning via SSH).
Shares cluster provisioning logic with the FastAPI webapp.

Usage:
    python app.py
"""

import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import httpx

# ─── Import validation + test missions from existing webapp ─────────────────

APP_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = Path.home() / "nav4rail_webapp"  # shared inference, validation, grammar
FINETUNE_DIR = WEBAPP_DIR / "finetune"
sys.path.insert(0, str(WEBAPP_DIR))
sys.path.insert(0, str(FINETUNE_DIR))

from validate_bt import validate_bt  # noqa: E402
from inference import TEST_MISSIONS  # noqa: E402
from bt_visualizer import render_bt_html  # noqa: E402

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

    return app


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=8778,
        root_path=ROOT_PATH,
    )
