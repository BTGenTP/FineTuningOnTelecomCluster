"""
Build a clean W&B config dict from the benchmark cfg.
=====================================================
The raw `base.yaml` dumps a bunch of fields that are either:
  - identical for every run in the project (e.g. `experiment.wandb_project`)
  - a registry of all 5 models when only 1 is active (`models.<key>.*`)
  - the non-active compute backend (`compute.slurm.*` on a vast.ai run)

Those pollute the W&B runs table with columns where every row has the same
value or where the value contradicts what the run actually did.

`build_wandb_config(cfg, job_type)` produces a reduced, runtime-aware dict:
  - Only the *active* model appears, under `model.*`
  - Only the *active* compute backend appears, under `compute.backend.*`
  - `runtime.*` captures the actual execution context (SLURM job id, hostname,
    GPU name) so every row in the W&B table is traceable to a specific job.

Backend detection priority:
  1. `SLURM_JOB_ID` set        → `slurm`
  2. `VAST_CONTAINERLABEL`,
     `VAST_TCP_HOST`, or
     `/workspace/repo` on path → `vastai`
  3. otherwise                 → `local`
"""

from __future__ import annotations

import copy
import os
import socket
from typing import Any


def _detect_execution_backend() -> str:
    if os.environ.get("SLURM_JOB_ID"):
        return "slurm"
    if (
        os.environ.get("VAST_CONTAINERLABEL")
        or os.environ.get("VAST_TCP_HOST")
        or os.environ.get("VAST_CONTAINER_ID")
    ):
        return "vastai"
    cwd = os.getcwd()
    if "/workspace/repo" in cwd or cwd.startswith("/workspace"):
        return "vastai"
    return "local"


def _detect_gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return None


def _build_runtime_block(job_type: str) -> dict[str, Any]:
    backend = _detect_execution_backend()
    runtime: dict[str, Any] = {
        "job_type": job_type,  # "training" | "inference"
        "execution_backend": backend,
        "hostname": socket.gethostname(),
    }

    gpu_name = _detect_gpu_name()
    if gpu_name:
        runtime["gpu_name"] = gpu_name

    if backend == "slurm":
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
        slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        slurm_job_name = os.environ.get("SLURM_JOB_NAME")
        slurm_partition = os.environ.get("SLURM_JOB_PARTITION")

        if slurm_job_id:
            runtime["slurm_job_id"] = slurm_job_id
        if slurm_array_job_id and slurm_array_task_id:
            # Composite reference that matches the output directory suffix
            # `nav4rail_<method>_<constraint>_<model>_<ARRAY_JOB_ID>` when
            # launched via array_models.sh.
            runtime["slurm_array_ref"] = f"{slurm_array_job_id}_{slurm_array_task_id}"
            runtime["slurm_array_job_id"] = slurm_array_job_id
            runtime["slurm_array_task_id"] = slurm_array_task_id
        if slurm_job_name:
            runtime["slurm_job_name"] = slurm_job_name
        if slurm_partition:
            runtime["slurm_partition"] = slurm_partition

    if backend == "vastai":
        for key in ("VAST_CONTAINERLABEL", "VAST_CONTAINER_ID", "VAST_TCP_HOST"):
            val = os.environ.get(key)
            if val:
                runtime[key.lower()] = val

    return runtime


def build_wandb_config(cfg: dict, job_type: str) -> dict[str, Any]:
    """Return a reduced, runtime-aware config dict for `wandb.init(config=...)`.

    Args:
        cfg: The full benchmark config (parsed base.yaml + overrides).
        job_type: "training" or "inference".
    """
    out = copy.deepcopy(cfg)

    # ── Collapse model registry to active model ────────────────────────────
    active_key = out.get("model", {}).get("key")
    models_registry = out.pop("models", {}) or {}
    if active_key and active_key in models_registry:
        model_spec = copy.deepcopy(models_registry[active_key])
        model_spec["key"] = active_key
        out["model"] = model_spec
    elif active_key:
        out["model"] = {"key": active_key}

    # ── Collapse compute to the active backend ─────────────────────────────
    backend = _detect_execution_backend()
    compute_in = out.pop("compute", {}) or {}
    compute_out: dict[str, Any] = {"backend": backend}
    if backend in compute_in and isinstance(compute_in[backend], dict):
        compute_out["backend_config"] = compute_in[backend]
    out["compute"] = compute_out

    # ── Drop the dev-convenience default; detected backend wins ────────────
    # (`compute.default_backend` was identical on every row.)

    # ── Prune method-specific blocks that don't match the active method ────
    method = out.get("training", {}).get("method", "sft")
    for block, keep_when in (
        ("grpo", method == "grpo"),
        ("dpo", method == "dpo"),
        ("kto", method == "kto"),
        ("pot", method == "pot"),
        ("react_pot_agent", method == "react_pot_agent"),
        ("react_base_agent", method == "react_base_agent"),
    ):
        if not keep_when:
            out.pop(block, None)

    # ── Flatten constraint mode if present ─────────────────────────────────
    # (Already handled by callers adding `constraint_mode`; nothing to do.)

    # ── Attach runtime metadata ────────────────────────────────────────────
    out["runtime"] = _build_runtime_block(job_type)

    return out


def build_run_name_suffix() -> str:
    """Short suffix to append to W&B run names for traceability.

    Returns the SLURM array ref (e.g. ``123456_2``), the plain SLURM job id,
    a vast.ai container label, or an empty string — whatever actually
    identifies the run in the compute dashboard.
    """
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_job and array_task:
        return f"{array_job}_{array_task}"
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        return slurm_job_id
    vast_label = os.environ.get("VAST_CONTAINERLABEL") or os.environ.get("VAST_CONTAINER_ID")
    if vast_label:
        return f"vast{vast_label}"
    return ""
