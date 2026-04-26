# Running LangGraph Agents on SLURM and vast.ai

Practical guide for executing `ReActPoTAgent` and `ReActBaseAgent` on Telecom-Paris SLURM and vast.ai, with **real-time tracing** and **post-hoc visualization** of LangGraph state transitions (incl. `ReActState` / `ReActBaseState`).

---

## 1. Decision matrix — where each compute target fits

| Scenario | Recommended target | Rationale |
|---|---|---|
| Phase-1 baselines (zero-shot, few-shot, CoT, schema) on 7-9B | SLURM P100 (16 GB) | Inference fits in 16 GB GGUF Q4_K_M; free; 100 missions ~1-3 h |
| ReAct + GBNF/Outlines on 7-9B | SLURM 3090 (24 GB) | Constraint backends raise memory pressure ~+4 GB |
| 14B (Qwen 2.5) inference + ReAct | vast.ai RTX 4090 / A100 | 14B is tight on 3090; rent on demand |
| SDPO Iterated DPO with N=8 candidates | SLURM 3090 with checkpointing | Long wall time; checkpoint every iteration |

---

## 2. Hard constraints to design around

### 2.1 Networking
- **SLURM compute nodes** at Telecom-Paris : **no outbound internet** (sometimes through proxy, never reliably). LangSmith / Weave streaming **WILL fail**. W&B works because it queues offline and syncs from the login node.
- **vast.ai** : outbound internet available. LangSmith / Weave / W&B all stream live.

### 2.2 Filesystem
- SLURM home is networked, slow on small writes. Always log traces to **`$SLURM_TMPDIR`** (local SSD) and rsync at job end.
- vast.ai : `/workspace` is the persistent volume; `/tmp` is ephemeral.

### 2.3 LangGraph specifics
- LangGraph holds the entire state in RAM. For `max_iterations=3` × 100 missions × ~5 KB / iteration ≈ 1.5 MB — negligible.
- The `recursion_limit` config field MUST be > nodes-per-iteration × max_iterations (already set to `4*max_iterations + 8` in `react_pot_agent.py` and `3*max_iterations + 6` in `react_base_agent.py`).
- LangGraph 0.2.x → 0.3.x changed the `compile()` signature. Pin in `requirements.txt`: `langgraph>=0.2.50,<0.4`.

---

## 3. Real-time tracing — the three layers

### Layer A — In-process JSONL trace (always on, no external dep)

`agent.run(mission) → AgentResult.trace` is already a `list[dict]` populated at every LangGraph node. To stream it to disk in real time, monkey-patch each node:

```python
# src/eval/trace_writer.py
import json, threading, time
from pathlib import Path

class JSONLTracer:
    """Append per-node-transition records to a JSONL file as they happen."""
    def __init__(self, path: Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8", buffering=1)  # line-buffered

    def emit(self, mission_id: str, iteration: int, step: str, payload: dict):
        rec = {
            "ts": time.time(),
            "run_id": self.run_id,
            "mission_id": mission_id,
            "iteration": iteration,
            "step": step,
            **payload,
        }
        with self._lock:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()  # critical: SLURM kills jobs without flush

    def close(self):
        self._fh.close()
```

Wire it into the agent at construction (`agent.tracer = JSONLTracer(...)`) and call `self.tracer.emit(...)` at the end of each `generate_xml`/`validate`/`reflect` node. The `trace` already contains the right fields; tracer just persists them line-by-line.

**SLURM job** sets `TRACE_PATH=$SLURM_TMPDIR/trace.jsonl`, then rsyncs back at exit.

### Layer B — W&B (offline + login-node sync)

W&B is the production tracker. For each agent run, log:
- `eval/agent_iterations` (per mission)
- `eval/agent_llm_latency_s` (per mission)
- `eval/per_iteration/{i}/score` (line plot of score vs iteration — visualizes the reflexion loop)
- An **artifact** `trace.jsonl` so the full ReActState lineage is preserved per run

```bash
# inside SLURM job
export WANDB_MODE=offline
export WANDB_DIR=$SLURM_TMPDIR/wandb
python -m src.eval.benchmark --config configs/base.yaml --prompt-mode react_base_agent

# at job end, on the login node
wandb sync $SLURM_TMPDIR/wandb
```

For vast.ai, use `WANDB_MODE=online` and skip the sync step.

### Layer C — LangGraph topology PNG (offline-friendly)

```python
# anywhere in the agent code
graph = self._build_langgraph()
mermaid = graph.get_graph().draw_mermaid()    # text representation
png_bytes = graph.get_graph().draw_mermaid_png()  # requires mermaid CLI or remote API
```

`draw_mermaid()` is text-only and works offline (write the .mermaid file to the run dir; render later with `mmdc` on the login node or in CI).
`draw_mermaid_png()` calls a remote service — DO NOT USE on SLURM compute nodes (silent hang on the proxy).

---

## 4. Visualizing ReActState transitions

### Option 1 — Plain Python timeline (what I'd recommend first)

`benchmarking/scripts/plot_agent_trace.py` (to be added when needed) reads `trace.jsonl` and produces:
- **Timeline plot** : x=mission_id, y=iteration, color=step, hover=score → matplotlib or plotly
- **Score-vs-iteration line** per mission, faceted by category — shows where the reflexion loop converges
- **Error-type histogram** — distribution of `last_error_type` at iteration 1 vs 2 vs 3

This is the lowest-overhead path; works on any machine that has the JSONL, no service dependency.

### Option 2 — W&B custom panel

In W&B run, log the full trace as a `Table`:
```python
import wandb
table = wandb.Table(columns=["mission_id", "iteration", "step", "score", "errors", "latency_s"])
for rec in trace_jsonl_records:
    table.add_data(rec["mission_id"], rec["iteration"], rec["step"], rec.get("score", 0), rec.get("errors", []), rec.get("latency_s", 0))
wandb.log({"agent_trace": table})
```
Then build a panel filtering by mission category, iteration, score range. Reusable across runs.

### Option 3 — Custom webapp (only if you need live streaming)

You already have a Gradio webapp (`benchmarking/webapp.py` per project memory). Add a `/trace` route that tails `trace.jsonl` and renders with `gradio.LinePlot`. Useful when watching a long ReAct run live (vast.ai) but **not** for SLURM (no inbound port open by default; tunnel via `ssh -L 7860:compute-node:7860` if you really need it).

For the inspection workflow you described in memory.md (cluster_sync_pull.sh), the cleaner path is:
1. Job runs on SLURM, writes `trace.jsonl` to `$SLURM_TMPDIR`.
2. End-of-job hook rsyncs to `~/runs/<run_id>/trace.jsonl`.
3. `cluster_sync_pull.sh` brings it local.
4. `streamlit run benchmarking/scripts/trace_viewer.py runs/<run_id>/trace.jsonl` shows it locally.

---

## 5. SLURM job template — `react_base_agent` with full tracing

```bash
#!/bin/bash
#SBATCH --job-name=nav4rail_react_base
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm/react_base_%j/slurm.out

set -euo pipefail
source scripts/slurm/_common.sh   # bz2 shim, venv bootstrap, .env loading

RUN_DIR="runs/slurm/react_base_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

export WANDB_MODE=offline
export WANDB_DIR="$SLURM_TMPDIR/wandb"
export TRACE_PATH="$SLURM_TMPDIR/trace.jsonl"
export PYTHONUNBUFFERED=1   # critical: see logs in real time, not at job end

python -m src.eval.benchmark \
  --config configs/base.yaml \
  --prompt-mode react_base_agent \
  --constraint gbnf \
  --output "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/run.log"

# Persist the trace + W&B offline run before SLURM clears $SLURM_TMPDIR
cp "$TRACE_PATH" "$RUN_DIR/trace.jsonl" || true
rsync -a "$WANDB_DIR/" "$RUN_DIR/wandb/" || true

# Sync W&B from login node:  wandb sync $RUN_DIR/wandb
```

---

## 6. vast.ai job template — same agent, live tracing

```bash
# ~/run_react_base.sh (executed inside the vast.ai container)
set -euo pipefail

cd /workspace/benchmarking
source .venv/bin/activate

export WANDB_MODE=online
export WANDB_PROJECT=nav4rail-bench
export WANDB_API_KEY=$(cat ~/.wandb_key)
export TRACE_PATH=/workspace/runs/react_base_$(date +%s)/trace.jsonl
export PYTHONUNBUFFERED=1

mkdir -p $(dirname "$TRACE_PATH")

python -m src.eval.benchmark \
  --config configs/base.yaml \
  --prompt-mode react_base_agent \
  --output $(dirname "$TRACE_PATH")
```

For vast.ai with internet:
- Set `LANGSMITH_API_KEY` and `LANGCHAIN_TRACING_V2=true` to get full LangGraph tracing in LangSmith automatically (LangGraph's official integration). **Do not** set this on SLURM — silent failures on proxy.

---

## 7. Quick-fire checklist

- [ ] `langgraph>=0.2.50,<0.4` pinned in `requirements.txt`
- [ ] Every agent run writes `$TRACE_PATH` (JSONL) — survive the job-clean-up via rsync
- [ ] `PYTHONUNBUFFERED=1` set in every SLURM script (otherwise logs only flush at job end)
- [ ] W&B `mode=offline` on SLURM, `online` on vast.ai
- [ ] `LANGSMITH_*` env vars NEVER set on SLURM (proxy hang)
- [ ] `recursion_limit` ≥ `nodes_per_iter × max_iterations + safety` (already wired)
- [ ] Trace viewer (`streamlit` or `wandb` Table) configured locally, not on cluster

---

## 8. Common failure modes seen in NAV4RAIL agents

| Symptom | Root cause | Fix |
|---|---|---|
| LangGraph hangs forever on `graph.invoke(...)` | `recursion_limit` too low for `max_iterations × nodes_per_iter` | Bump in `agent.run()` config |
| `ImportError: langgraph` mid-run | venv missing the optional dep | `use_langgraph: false` falls back to plain loop, OR `pip install langgraph` |
| Trace JSONL empty at job end | no `flush()` after each write | line-buffered file (`buffering=1`) + explicit `flush()` |
| W&B run shows score=0 across the board on Llama3 + GBNF | `grammar_assertion: vocab mismatch` (already caught in `_generate_xml`) | Mark Llama3 as GBNF-incompatible; switch to Outlines |
| Mistral GBNF crashes at ~60/100 missions | `All stacks are empty` (already caught) | benchmark continues; record as 0 for that mission |
